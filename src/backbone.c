/* Qwen3.5-MoE hybrid backbone, following fish_speech fish_omni backbone.py /
 * layers.py / moe.py, including where the reference computes in fp32. */
#include <math.h>
#include <string.h>

#include "model.h"

/* ---- shared helpers ---- */

/* ZeroCenteredRMSNorm: fp32 rmsnorm times fp32 (1 + w), back to x's dtype. */
static mlx_array zc_norm(mlx_array x, mlx_array w1, float eps) {
  mlx_dtype dtype = m_dtype(x);
  return m_astype(m_rms_norm(m_astype(x, MLX_FLOAT32), w1, eps), dtype);
}

/* ---- gated delta rule Metal kernel ----
 * Adapted from the recurrent kernel in mlx-lm (mlx_lm/models/gated_delta.py,
 * Copyright (c) 2025 Apple Inc., MIT License), here with fp32 q/k/v/g/beta. */

static const char *GDN_SOURCE =
    "  auto n = thread_position_in_grid.z;\n"
    "  auto b_idx = n / Hv;\n"
    "  auto hv_idx = n % Hv;\n"
    "  auto hk_idx = hv_idx / (Hv / Hk);\n"
    "  constexpr int n_per_t = Dk / 32;\n"
    "  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;\n"
    "  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;\n"
    "  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;\n"
    "  y += b_idx * T * Hv * Dv + hv_idx * Dv;\n"
    "  auto dk_idx = thread_position_in_threadgroup.x;\n"
    "  auto dv_idx = thread_position_in_grid.y;\n"
    "  auto i_state = state_in + (n * Dv + dv_idx) * Dk;\n"
    "  auto o_state = state_out + (n * Dv + dv_idx) * Dk;\n"
    "  float state[n_per_t];\n"
    "  for (int i = 0; i < n_per_t; ++i) {\n"
    "    state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);\n"
    "  }\n"
    "  auto g_ = g + b_idx * T * Hv;\n"
    "  auto beta_ = beta + b_idx * T * Hv;\n"
    "  for (int t = 0; t < T; ++t) {\n"
    "    float kv_mem = 0.0f;\n"
    "    for (int i = 0; i < n_per_t; ++i) {\n"
    "      auto s_idx = n_per_t * dk_idx + i;\n"
    "      state[i] = state[i] * g_[hv_idx];\n"
    "      kv_mem += state[i] * k_[s_idx];\n"
    "    }\n"
    "    kv_mem = simd_sum(kv_mem);\n"
    "    auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];\n"
    "    float out = 0.0f;\n"
    "    for (int i = 0; i < n_per_t; ++i) {\n"
    "      auto s_idx = n_per_t * dk_idx + i;\n"
    "      state[i] = state[i] + k_[s_idx] * delta;\n"
    "      out += state[i] * q_[s_idx];\n"
    "    }\n"
    "    out = simd_sum(out);\n"
    "    if (thread_index_in_simdgroup == 0) {\n"
    "      y[dv_idx] = static_cast<float>(out);\n"
    "    }\n"
    "    q_ += Hk * Dk;\n"
    "    k_ += Hk * Dk;\n"
    "    v_ += Hv * Dv;\n"
    "    y += Hv * Dv;\n"
    "    g_ += Hv;\n"
    "    beta_ += Hv;\n"
    "  }\n"
    "  for (int i = 0; i < n_per_t; ++i) {\n"
    "    o_state[n_per_t * dk_idx + i] = state[i];\n"
    "  }\n";

static mlx_fast_metal_kernel gdn_kernel;
static bool gdn_kernel_ready;

/* Token-by-token gated delta rule over T steps, all fp32.
 * q, k [1, T, Hk, Dk] (normalized), v [1, T, Hv, Dv], g (decay) / beta [1, T, Hv],
 * state [1, Hv, Dv, Dk]. Returns y [1, T, Hv, Dv] and the new state. */
static void gated_delta(mlx_array q, mlx_array k, mlx_array v, mlx_array g,
                        mlx_array beta, mlx_array state, mlx_array *y, mlx_array *state_out) {
  if (!gdn_kernel_ready) {
    const char *in_names[] = {"q", "k", "v", "g", "beta", "state_in", "T"};
    const char *out_names[] = {"y", "state_out"};
    mlx_vector_string ins = mlx_vector_string_new_data(in_names, 7);
    mlx_vector_string outs = mlx_vector_string_new_data(out_names, 2);
    gdn_kernel = mlx_fast_metal_kernel_new("drama3_gated_delta", ins, outs, GDN_SOURCE, "",
                                           true, false);
    mlx_vector_string_free(ins);
    mlx_vector_string_free(outs);
    gdn_kernel_ready = true;
  }
  int T = m_dim(q, 1), Hk = m_dim(q, 2), Dk = m_dim(q, 3);
  int Hv = m_dim(v, 2), Dv = m_dim(v, 3);
  mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){1, T, Hv, Dv}, 4, MLX_FLOAT32);
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){1, Hv, Dv, Dk}, 4, MLX_FLOAT32);
  mlx_fast_metal_kernel_config_set_grid(cfg, 32, Dv, Hv);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dk", Dk);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dv", Dv);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hk", Hk);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hv", Hv);
  mlx_array inputs[] = {q, k, v, g, beta, state, m_int(T)};
  mlx_vector_array in_vec = mlx_vector_array_new_data(inputs, 7);
  mlx_vector_array out_vec = mlx_vector_array_new();
  if (mlx_fast_metal_kernel_apply(&out_vec, gdn_kernel, in_vec, cfg, g_stream))
    die("gated delta kernel failed");
  mlx_array a = mlx_array_new(), b = mlx_array_new();
  mlx_vector_array_get(&a, out_vec, 0);
  mlx_vector_array_get(&b, out_vec, 1);
  *y = track(a);
  *state_out = track(b);
  mlx_vector_array_free(in_vec);
  mlx_vector_array_free(out_vec);
  mlx_fast_metal_kernel_config_free(cfg);
}

static mlx_array l2norm_f32(mlx_array x) {
  mlx_array ss = m_sum(m_square(x), -1, true);
  return m_mul(x, m_rsqrt(m_add(ss, m_float(1e-6f))));
}

/* GatedDeltaNet: x [1, T, H] bf16 -> [1, T, H] bf16. */
static mlx_array gdn_forward(Model *m, int li, mlx_array x) {
  const Config *c = &m->cfg;
  BackboneLayer *L = &m->layers[li];
  LayerCache *C = &m->cache[li];
  int T = m_dim(x, 1);
  int key_dim = c->lin_n_key * c->lin_key_dim;
  int value_dim = c->lin_n_value * c->lin_value_dim;
  int hist = c->conv_kernel - 1;

  mlx_array qkv = m_linear(x, L->in_qkv); /* [1, T, conv_dim] */
  mlx_array padded = m_concat2(C->conv, qkv, 1);
  assign(&C->conv, m_slice_axis(padded, 1, T, T + hist));

  /* depthwise causal conv: out[t] = sum_j w[:, j] * padded[t + j], fp32 accumulation */
  mlx_array pf = m_astype(padded, MLX_FLOAT32);
  mlx_array wf = m_astype(L->conv_w, MLX_FLOAT32); /* [C, K] */
  mlx_array acc = null_array();
  for (int j = 0; j < c->conv_kernel; j++) {
    mlx_array wj = m_reshape(m_slice_axis(wf, 1, j, j + 1), SH(1, 1, -1));
    mlx_array term = m_mul(m_slice_axis(pf, 1, j, j + T), wj);
    acc = is_null(acc) ? term : m_add(acc, term);
  }
  mlx_array mixed = m_silu_f32(m_astype(acc, MLX_BFLOAT16));

  mlx_array parts[3];
  m_split(mixed, (int[]){key_dim, 2 * key_dim}, 2, -1, parts);
  mlx_array q = m_reshape(parts[0], SH(1, T, c->lin_n_key, c->lin_key_dim));
  mlx_array k = m_reshape(parts[1], SH(1, T, c->lin_n_key, c->lin_key_dim));
  mlx_array v = m_reshape(parts[2], SH(1, T, c->lin_n_value, c->lin_value_dim));
  mlx_array z = m_reshape(m_linear(x, L->in_z), SH(1, T, c->lin_n_value, c->lin_value_dim));
  mlx_array beta = m_sigmoid(m_astype(m_linear(x, L->in_b), MLX_FLOAT32));
  mlx_array a = m_astype(m_linear(x, L->in_a), MLX_FLOAT32);
  mlx_array decay = m_mul(m_neg(m_exp(m_astype(L->a_log, MLX_FLOAT32))),
                          m_softplus(m_add(a, m_astype(L->dt_bias, MLX_FLOAT32))));
  mlx_array g = m_exp(decay);

  mlx_array qf = m_mul(l2norm_f32(m_astype(q, MLX_FLOAT32)),
                       m_float(1.0f / sqrtf((float)c->lin_key_dim)));
  mlx_array kf = l2norm_f32(m_astype(k, MLX_FLOAT32));
  mlx_array vf = m_astype(v, MLX_FLOAT32);
  mlx_array y, new_state;
  gated_delta(qf, kf, vf, g, beta, C->rec, &y, &new_state);
  assign(&C->rec, new_state);

  /* GatedRMSNorm: weight * rmsnorm(core).to(bf16), then * silu(z) in fp32 */
  mlx_array core = m_astype(y, MLX_BFLOAT16);
  mlx_array normed =
      m_astype(m_rms_norm(m_astype(core, MLX_FLOAT32), null_array(), c->rms_eps), MLX_BFLOAT16);
  normed = m_mul(L->gdn_norm, normed);
  mlx_array zf = m_astype(z, MLX_FLOAT32);
  mlx_array gated = m_astype(m_mul(m_astype(normed, MLX_FLOAT32), m_mul(zf, m_sigmoid(zf))),
                             MLX_BFLOAT16);
  return m_linear(m_reshape(gated, SH(1, T, value_dim)), L->out_proj);
}

/* ---- gated full attention ---- */

static mlx_array attn_forward(Model *m, int li, mlx_array x, int pos) {
  const Config *c = &m->cfg;
  BackboneLayer *L = &m->layers[li];
  LayerCache *C = &m->cache[li];
  int T = m_dim(x, 1);
  int hd = c->head_dim;

  mlx_array qg = m_reshape(m_linear(x, L->q_proj), SH(1, T, c->n_heads, 2 * hd));
  mlx_array qparts[2];
  m_split(qg, (int[]){hd}, 1, -1, qparts);
  mlx_array gate = m_reshape(qparts[1], SH(1, T, c->n_heads * hd));
  mlx_array q = zc_norm(qparts[0], L->q_norm, c->rms_eps);
  mlx_array k = zc_norm(m_reshape(m_linear(x, L->k_proj), SH(1, T, c->n_kv_heads, hd)),
                        L->k_norm, c->rms_eps);
  mlx_array v = m_reshape(m_linear(x, L->v_proj), SH(1, T, c->n_kv_heads, hd));
  q = m_rope(m_transpose(q, SH(0, 2, 1, 3)), c->rotary_dim, false, c->rope_theta, pos);
  k = m_rope(m_transpose(k, SH(0, 2, 1, 3)), c->rotary_dim, false, c->rope_theta, pos);
  v = m_transpose(v, SH(0, 2, 1, 3));

  int end = pos + T;
  if (end > m_dim(C->k, 2)) die("backbone cache capacity exceeded");
  assign(&C->k, m_slice_update(C->k, k, (int[]){0, 0, pos, 0},
                               (int[]){1, c->n_kv_heads, end, hd}, 4));
  assign(&C->v, m_slice_update(C->v, v, (int[]){0, 0, pos, 0},
                               (int[]){1, c->n_kv_heads, end, hd}, 4));
  mlx_array keys = m_slice_axis(C->k, 2, 0, end);
  mlx_array values = m_slice_axis(C->v, 2, 0, end);
  mlx_array out = m_sdpa(q, keys, values, 1.0f / sqrtf((float)hd), T > 1 ? "causal" : "",
                         null_array());
  out = m_reshape(m_transpose(out, SH(0, 2, 1, 3)), SH(1, T, c->n_heads * hd));
  return m_linear(m_mul(out, m_sigmoid(gate)), L->o_proj);
}

/* ---- sparse MoE ---- */

static mlx_array moe_forward(Model *m, int li, mlx_array x) {
  const Config *c = &m->cfg;
  BackboneLayer *L = &m->layers[li];
  int T = m_dim(x, 1), H = c->hidden, K = c->top_k, E = c->n_experts;
  mlx_array flat = m_reshape(x, SH(T, H));

  mlx_array probs = m_softmax(m_astype(m_linear(flat, L->router), MLX_FLOAT32), -1);
  mlx_array inds = m_slice_axis(m_argpartition(probs, E - K, -1), -1, E - K, E);
  mlx_array scores = m_take_along_axis(probs, inds, -1);
  if (c->norm_topk) scores = m_div(scores, m_sum(scores, -1, true));
  mlx_array weights = m_astype(m_astype(scores, MLX_BFLOAT16), MLX_FLOAT32); /* [T, K] */

  int gs = c->expert_group, bits = c->expert_bits;
  int inter = c->moe_inter;
  mlx_array expert_out; /* [T, K, H] bf16 */
  if (T * K >= 64) {
    mlx_array flat_idx = m_reshape(inds, SH(T * K));
    mlx_array order = m_argsort(flat_idx, 0);
    mlx_array inv = m_argsort(order, 0);
    mlx_array rows = m_floor_divide(order, m_astype(m_int(K), m_dtype(order)));
    mlx_array xs = m_take(m_reshape(flat, SH(T, 1, H)), rows, 0); /* [T*K, 1, H] */
    mlx_array idx = m_take(flat_idx, order, 0);
    mlx_array gu = m_gather_qmm(xs, L->gu_w, L->gu_s, L->gu_b, idx, gs, bits, true);
    mlx_array gparts[2];
    m_split(gu, (int[]){inter}, 1, -1, gparts);
    mlx_array h = m_mul(m_silu_f32(gparts[0]), gparts[1]);
    mlx_array d = m_gather_qmm(h, L->down_w, L->down_s, L->down_b, idx, gs, bits, true);
    expert_out = m_reshape(m_take(d, inv, 0), SH(T, K, H));
  } else {
    mlx_array xs = m_reshape(flat, SH(T, 1, 1, H));
    mlx_array gu = m_gather_qmm(xs, L->gu_w, L->gu_s, L->gu_b, inds, gs, bits, false);
    mlx_array gparts[2];
    m_split(gu, (int[]){inter}, 1, -1, gparts);
    mlx_array h = m_mul(m_silu_f32(gparts[0]), gparts[1]);
    mlx_array d = m_gather_qmm(h, L->down_w, L->down_s, L->down_b, inds, gs, bits, false);
    expert_out = m_reshape(d, SH(T, K, H));
  }
  mlx_array routed = m_astype(
      m_sum(m_mul(m_astype(expert_out, MLX_FLOAT32), m_expand_dims(weights, -1)), 1, false),
      MLX_BFLOAT16);

  mlx_array shared = m_linear(
      m_mul(m_silu_f32(m_linear(flat, L->sh_gate)), m_linear(flat, L->sh_up)), L->sh_down);
  mlx_array shared_gate = m_sigmoid(m_linear(flat, L->sh_expert_gate));
  return m_reshape(m_add(routed, m_mul(shared_gate, shared)), SH(1, T, H));
}

/* ---- public ---- */

mlx_array backbone_embed_frames(Model *m, mlx_array frames) {
  const Config *c = &m->cfg;
  int T = m_dim(frames, 0);
  mlx_array tokens = m_reshape(m_slice_axis(frames, 1, 0, 1), SH(T));
  mlx_array text = m_take(m->embed_tokens, tokens, 0); /* [T, H] bf16 */
  int count = c->vq_backbone_codebooks;
  if (count == 0) return text;
  mlx_array extra = null_array();
  for (int j = 1; j <= count; j++) {
    mlx_array codes = m_reshape(m_slice_axis(frames, 1, 1 + j, 2 + j), SH(T));
    codes = m_add(codes, m_int(j * c->code_vocab));
    mlx_array e = m_astype(m_take(m->codebook_embeddings, codes, 0), MLX_FLOAT32);
    extra = is_null(extra) ? e : m_add(extra, e);
  }
  extra = m_astype(extra, MLX_BFLOAT16);
  mlx_array sum = m_astype(m_add(text, extra), MLX_FLOAT32);
  mlx_array mixed = m_astype(m_div(sum, m_float(sqrtf((float)(1 + count)))), MLX_BFLOAT16);
  mlx_array semantic = m_logical_and(m_greater_equal(tokens, m_int(c->semantic_start)),
                                     m_less_equal(tokens, m_int(c->semantic_end)));
  return m_where(m_expand_dims(semantic, -1), mixed, text);
}

mlx_array backbone_layer_forward(Model *m, int layer, mlx_array x, int pos);

mlx_array backbone_forward(Model *m, mlx_array embeds, int pos) {
  const Config *c = &m->cfg;
  mlx_array x = embeds;
  if (m_ndim(x) == 2) x = m_expand_dims(x, 0);
  for (int i = 0; i < c->n_layers; i++) x = backbone_layer_forward(m, i, x, pos);
  return zc_norm(x, m->final_norm, c->rms_eps);
}

mlx_array backbone_mixer_forward(Model *m, int layer, mlx_array x, int pos) {
  return m->cfg.is_linear[layer] ? gdn_forward(m, layer, x) : attn_forward(m, layer, x, pos);
}

mlx_array backbone_moe_forward(Model *m, int layer, mlx_array x) {
  return moe_forward(m, layer, x);
}

mlx_array backbone_layer_forward(Model *m, int layer, mlx_array x, int pos) {
  const Config *c = &m->cfg;
  BackboneLayer *L = &m->layers[layer];
  mlx_array h = zc_norm(x, L->input_norm, c->rms_eps);
  x = m_add(x, backbone_mixer_forward(m, layer, h, pos));
  return m_add(x, moe_forward(m, layer, zc_norm(x, L->post_norm, c->rms_eps)));
}

mlx_array backbone_audio_logits(Model *m, mlx_array hidden) {
  return m_linear(hidden, m->lm_head_audio);
}
