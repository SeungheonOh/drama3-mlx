/* Codebook-MTP audio decoder tower, following fish_speech fish_omni tower.py.
 *
 * Each tower layer is split around its KV cache: tower_pre (code embedding mix,
 * projections, RoPE) and tower_post (output, feed-forward, final norm) are pure
 * functions, compiled for single-row decode and run eagerly for prefill. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"

/* Repetition-aware sampling of the main token, fixed by the training recipe. */
#define RAS_TEMPERATURE 1.0f
#define RAS_TOP_K 30
#define RAS_TOP_P 0.95f

static mlx_array zc_norm(mlx_array x, mlx_array w1, float eps) {
  return m_rms_norm_f32(x, w1, eps, m_dtype(x));
}

typedef struct {
  mlx_array embedding, enorm, hnorm, eh_proj, attn_norm, wqkv, q_norm, k_norm;
} PreW;

typedef struct {
  mlx_array wo, ffn_norm, w1, w3, w2, final_norm;
} PostW;

#define PRE_W_COUNT 8
#define POST_W_COUNT 6

static PreW pre_weights(const TowerLayer *t) {
  return (PreW){t->embedding, t->enorm, t->hnorm, t->eh_proj,
                t->attn_norm, t->wqkv,  t->q_norm, t->k_norm};
}

static PostW post_weights(const TowerLayer *t) {
  return (PostW){t->wo, t->ffn_norm, t->w1, t->w3, t->w2, t->final_norm};
}

/* ---- fused input mix ----
 * One threadgroup per row: the code's embedding row (clamped code; zero for a
 * non-semantic row) plus the shared input, then the two zero-centered norms
 * (enorm of the embedding, hnorm of the hidden) with m_rms_norm_f32's
 * reduction, written side by side as eh_proj's input [L, 2 * dim]. */

static const char *MIX_SOURCE =
    "  uint row = threadgroup_position_in_grid.x;\n"
    "  uint tid = thread_position_in_threadgroup.x;\n"
    "  uint tg = threads_per_threadgroup.x;\n"
    "  int code = min(max(int(codes[row]), 0), V - 1);\n"
    "  bool keep = semantic[row];\n"
    "  auto er = embedding + (size_t)code * D;\n"
    "  auto hr = hidden + (size_t)row * D;\n"
    "  auto outr = out + (size_t)row * 2 * D;\n"
    "  /* MLX's rms_single_row for both norms: 4 contiguous elements per thread */\n"
    "  float acc_e = 0.0f, acc_h = 0.0f;\n"
    "  for (uint i = 4 * tid; i < 4 * tid + 4 && i < D; i++) {\n"
    "    T e = keep ? er[i] : static_cast<T>(0.0f);\n"
    "    if (SHARED) e = static_cast<T>(float(e) + float(shared[(size_t)row * D + i]));\n"
    "    outr[i] = e;\n"
    "    float v = float(e);\n"
    "    acc_e += v * v;\n"
    "    float u = float(hr[i]);\n"
    "    acc_h += u * u;\n"
    "  }\n"
    "  threadgroup float partial_e[32];\n"
    "  threadgroup float partial_h[32];\n"
    "  acc_e = simd_sum(acc_e);\n"
    "  acc_h = simd_sum(acc_h);\n"
    "  uint lane = tid % 32, sg = tid / 32;\n"
    "  if (lane == 0) { partial_e[sg] = acc_e; partial_h[sg] = acc_h; }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (sg == 0) {\n"
    "    float v = lane < (tg + 31) / 32 ? partial_e[lane] : 0.0f;\n"
    "    float u = lane < (tg + 31) / 32 ? partial_h[lane] : 0.0f;\n"
    "    v = simd_sum(v);\n"
    "    u = simd_sum(u);\n"
    "    if (lane == 0) { partial_e[0] = v; partial_h[0] = u; }\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  float inv_e = metal::precise::rsqrt(partial_e[0] / float(D) + EPS);\n"
    "  float inv_h = metal::precise::rsqrt(partial_h[0] / float(D) + EPS);\n"
    "  for (uint i = 4 * tid; i < 4 * tid + 4 && i < D; i++) {\n"
    "    float n = float(outr[i]) * inv_e;\n"
    "    n = n * enorm[i];\n"
    "    outr[i] = static_cast<T>(n);\n"
    "    float m = float(hr[i]) * inv_h;\n"
    "    m = m * hnorm[i];\n"
    "    outr[D + i] = static_cast<T>(m);\n"
    "  }\n";

static mlx_fast_metal_kernel mix_kernel;
static float mix_eps;
static bool mix_ready;

/* hidden [L, dim], codes [L] int32, semantic [L] bool, shared_input [L, dim] or
 * null: concat(enorm(e), hnorm(hidden)) [L, 2 * dim]. */
static mlx_array tower_mix(const Config *c, mlx_array embedding, mlx_array enorm,
                           mlx_array hnorm, mlx_array hidden, mlx_array codes,
                           mlx_array semantic, mlx_array shared_input) {
  if (!mix_ready) {
    const char *in_names[] = {"embedding", "enorm", "hnorm", "hidden", "codes", "semantic",
                              "shared"};
    const char *out_names[] = {"out"};
    char header[64];
    snprintf(header, sizeof(header), "constant float EPS = %.9g;\n", c->t_eps);
    mlx_vector_string ins = mlx_vector_string_new_data(in_names, 7);
    mlx_vector_string outs = mlx_vector_string_new_data(out_names, 1);
    mix_kernel = mlx_fast_metal_kernel_new("drama3_tower_mix", ins, outs, MIX_SOURCE, header,
                                           true, false);
    mlx_vector_string_free(ins);
    mlx_vector_string_free(outs);
    mix_eps = c->t_eps;
    mix_ready = true;
  }
  if (mix_eps != c->t_eps) die("tower mix kernel built for another eps");
  int L = m_dim(hidden, 0), D = c->t_dim;
  bool has_shared = !is_null(shared_input);
  mlx_dtype dtype = m_dtype(hidden);
  mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){L, 2 * D}, 2, dtype);
  int tg = 32 * (((D + 3) / 4 + 31) / 32); /* MLX's rms_single_row threadgroup */
  mlx_fast_metal_kernel_config_set_grid(cfg, tg * L, 1, 1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "V", c->code_vocab);
  mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "SHARED", has_shared);
  mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dtype);
  mlx_array inputs[7] = {embedding, enorm, hnorm, hidden, codes, semantic,
                         has_shared ? shared_input : hidden /* unused */};
  mlx_vector_array in_vec = mlx_vector_array_new_data(inputs, 7);
  mlx_vector_array out_vec = mlx_vector_array_new();
  if (mlx_fast_metal_kernel_apply(&out_vec, mix_kernel, in_vec, cfg, g_stream))
    die("tower mix kernel failed");
  mlx_array r = mlx_array_new();
  mlx_vector_array_get(&r, out_vec, 0);
  mlx_vector_array_free(in_vec);
  mlx_vector_array_free(out_vec);
  mlx_fast_metal_kernel_config_free(cfg);
  return track(r);
}

/* Mix the code embedding into hidden [L, dim] and project for attention:
 * mixed [L, dim] and the projection [L, 2 * q_width + 2 * kv_width], whose rows
 * are [Q | Gate | K | V]. */
static mlx_array tower_project(const Config *c, const PreW *w, mlx_array hidden,
                               mlx_array codes, mlx_array semantic, mlx_array shared_input,
                               mlx_array *mixed) {
  mlx_array cat;
  if (getenv("D3_NO_FUSED_MIX")) {
    mlx_array clamped = m_clip(codes, m_int(0), m_int(c->code_vocab - 1));
    mlx_array e = m_take(w->embedding, clamped, 0);
    e = m_where(m_expand_dims(semantic, -1), e, m_scalar(0.0f, m_dtype(e)));
    if (!is_null(shared_input)) e = m_add(e, shared_input);
    cat = m_concat2(zc_norm(e, w->enorm, c->t_eps), zc_norm(hidden, w->hnorm, c->t_eps), -1);
  } else {
    cat = tower_mix(c, w->embedding, w->enorm, w->hnorm, hidden, codes, semantic,
                    shared_input);
  }
  *mixed = m_linear(cat, w->eh_proj);
  return m_linear(zc_norm(*mixed, w->attn_norm, c->t_eps), w->wqkv);
}

/* tower_project, then q / k / v [1, heads, L, hd] (RoPE at offset) and the gate
 * [L, q_width]. */
static void tower_pre(const Config *c, const PreW *w, mlx_array hidden, mlx_array codes,
                      mlx_array semantic, mlx_array shared_input, mlx_array offset,
                      mlx_array *q, mlx_array *k, mlx_array *v, mlx_array *gate,
                      mlx_array *mixed) {
  int L = m_dim(hidden, 0), hd = c->t_head_dim;
  int q_width = c->t_heads * hd, kv_width = c->t_kv_heads * hd;
  mlx_array parts[4];
  m_split(tower_project(c, w, hidden, codes, semantic, shared_input, mixed),
          (int[]){q_width, 2 * q_width, 2 * q_width + kv_width}, 3, -1, parts);
  mlx_array qn = zc_norm(m_reshape(parts[0], SH(L, c->t_heads, hd)), w->q_norm, c->t_eps);
  mlx_array kn = zc_norm(m_reshape(parts[2], SH(L, c->t_kv_heads, hd)), w->k_norm, c->t_eps);
  /* [1, heads, L, hd]; interleaved partial RoPE over absolute row positions */
  *q = m_rope_dyn(m_expand_dims(m_transpose(qn, SH(1, 0, 2)), 0), c->t_rotary_dim, true,
                  c->t_rope_base, offset);
  *k = m_rope_dyn(m_expand_dims(m_transpose(kn, SH(1, 0, 2)), 0), c->t_rotary_dim, true,
                  c->t_rope_base, offset);
  *v = m_expand_dims(m_transpose(m_reshape(parts[3], SH(L, c->t_kv_heads, hd)), SH(1, 0, 2)), 0);
  *gate = parts[1];
}

/* Attention output, feed-forward and final norm: the layer's output [L, dim]. */
static mlx_array tower_post_gated(const Config *c, const PostW *w, mlx_array mixed,
                                  mlx_array gated) {
  mlx_array attn = m_linear(gated, w->wo);
  mlx_array x, h;
  if (getenv("D3_NO_FUSED_ADDNORM")) {
    x = m_add(mixed, attn);
    h = zc_norm(x, w->ffn_norm, c->t_eps);
  } else {
    m_add_rms_norm_f32(mixed, attn, w->ffn_norm, c->t_eps, &x, &h);
  }
  mlx_array ff = m_linear(m_linear_silu_pair(h, w->w1, w->w3), w->w2);
  if (getenv("D3_NO_FUSED_ADDNORM")) return zc_norm(m_add(x, ff), w->final_norm, c->t_eps);
  mlx_array sum, normed;
  m_add_rms_norm_f32(x, ff, w->final_norm, c->t_eps, &sum, &normed);
  return normed;
}

/* Gated attention output from attended [1, heads, L, hd] and gate [L, q_width],
 * then tower_post_gated. */
static mlx_array tower_post(const Config *c, const PostW *w, mlx_array mixed,
                            mlx_array attended, mlx_array gate) {
  int L = m_dim(mixed, 0), q_width = c->t_heads * c->t_head_dim;
  mlx_array out = m_reshape(m_transpose(m_squeeze(attended, 0), SH(1, 0, 2)), SH(L, q_width));
  return tower_post_gated(c, w, mixed, m_mul(out, m_sigmoid(gate)));
}

/* ---- compiled single-row pieces ---- */

/* in: hidden, codes, semantic, shared_input, offset, PreW...  out: q, k, v, gate, mixed */
static void pre_fn(const mlx_array *in, mlx_array *out, const void *ctx) {
  tower_pre(ctx, (const PreW *)(in + 5), in[0], in[1], in[2], in[3], in[4], &out[0], &out[1],
            &out[2], &out[3], &out[4]);
}

/* in: mixed, attended, gate, PostW...  out: layer output */
static void post_fn(const mlx_array *in, mlx_array *out, const void *ctx) {
  out[0] = tower_post(ctx, (const PostW *)(in + 3), in[0], in[1], in[2]);
}

/* in: mixed, attended, gate, PostW..., head  out: layer output, head logits */
static void post_head_fn(const mlx_array *in, mlx_array *out, const void *ctx) {
  out[0] = tower_post(ctx, (const PostW *)(in + 3), in[0], in[1], in[2]);
  out[1] = m_linear(out[0], in[3 + POST_W_COUNT]);
}

static AttnShape attn_shape(const Config *c) {
  return (AttnShape){.heads = c->t_heads,
                     .kv_heads = c->t_kv_heads,
                     .head_dim = c->t_head_dim,
                     .rotary_dim = c->t_rotary_dim,
                     .traditional = true,
                     .rope_base = c->t_rope_base,
                     .eps = c->t_eps,
                     .q_stride = c->t_head_dim,
                     .gate_base = c->t_heads * c->t_head_dim};
}

/* A whole single-row layer with the fused attention kernel.
 * in: hidden, codes, semantic, shared_input, pos, k_cache, v_cache, variant tag,
 *     PreW..., PostW... [, head]
 * out: layer output, k_new, v_new [, head logits] */
static void layer_common(const mlx_array *in, mlx_array *out, const Config *c, bool head) {
  const PreW *pw = (const PreW *)(in + 8);
  const PostW *qw = (const PostW *)(in + 8 + PRE_W_COUNT);
  AttnShape shape = attn_shape(c);
  mlx_array mixed, gated;
  mlx_array proj = tower_project(c, pw, in[0], in[1], in[2], in[3], &mixed);
  /* wqkv rows are [Q | Gate | K | V] */
  int q_width = c->t_heads * c->t_head_dim, kv_width = c->t_kv_heads * c->t_head_dim;
  mlx_array row = m_reshape(proj, SH(1, 1, -1));
  attn_decode_fused(&shape, ATTN_TAG_BLOCKS(in[7]), m_slice_axis(row, -1, 0, 2 * q_width),
                    m_slice_axis(row, -1, 2 * q_width, 2 * q_width + kv_width),
                    m_slice_axis(row, -1, 2 * q_width + kv_width, 2 * q_width + 2 * kv_width),
                    pw->q_norm, pw->k_norm, in[5], in[6], in[4], &gated, &out[1], &out[2]);
  out[0] = tower_post_gated(c, qw, mixed, m_reshape(gated, SH(1, c->t_heads * c->t_head_dim)));
  if (head) out[3] = m_linear(out[0], in[8 + PRE_W_COUNT + POST_W_COUNT]);
}

static void layer_fn(const mlx_array *in, mlx_array *out, const void *ctx) {
  layer_common(in, out, ctx, false);
}

static void layer_head_fn(const mlx_array *in, mlx_array *out, const void *ctx) {
  layer_common(in, out, ctx, true);
}

static void ensure_compiled(Model *m) {
  if (m->tower_compiled.pre.fn) return;
  m->tower_compiled.layer = (Compiled){.fn = layer_fn, .ctx = &m->cfg, .n_out = 3};
  m->tower_compiled.layer_head = (Compiled){.fn = layer_head_fn, .ctx = &m->cfg, .n_out = 4};
  m->tower_compiled.pre = (Compiled){.fn = pre_fn, .ctx = &m->cfg, .n_out = 5};
  m->tower_compiled.post = (Compiled){.fn = post_fn, .ctx = &m->cfg, .n_out = 1};
  m->tower_compiled.post_head = (Compiled){.fn = post_head_fn, .ctx = &m->cfg, .n_out = 2};
}

/* Writes rows pos..pos+L-1 into the layer's cache and attends over it. */
static mlx_array attend_cache(Model *m, TowerLayer *t, mlx_array q, mlx_array k, mlx_array v,
                              int pos) {
  const Config *c = &m->cfg;
  int L = m_dim(q, 2), hd = c->t_head_dim, end = pos + L;
  if (end > m_dim(t->k_cache, 2)) die("tower cache capacity exceeded");
  assign(&t->k_cache, m_slice_update(t->k_cache, k, (int[]){0, 0, pos, 0},
                                     (int[]){1, c->t_kv_heads, end, hd}, 4));
  assign(&t->v_cache, m_slice_update(t->v_cache, v, (int[]){0, 0, pos, 0},
                                     (int[]){1, c->t_kv_heads, end, hd}, 4));
  return m_sdpa(q, m_slice_axis(t->k_cache, 2, 0, end), m_slice_axis(t->v_cache, 2, 0, end),
                1.0f / sqrtf((float)hd), L > 1 ? "causal" : "", null_array());
}

/* TowerLayer: mix a code embedding into the hidden state, then one block.
 * hidden [L, dim], codes [L] int32, semantic [L] bool. With head, also returns
 * the head's logits for the next codebook in *head_logits. */
static mlx_array tower_layer(Model *m, TowerLayer *t, mlx_array hidden, mlx_array codes,
                             mlx_array semantic, int pos, mlx_array shared_input,
                             mlx_array head, mlx_array *head_logits) {
  const Config *c = &m->cfg;
  PreW pw = pre_weights(t);
  PostW qw = post_weights(t);
  mlx_array offset = m_int(pos);
  if (m_dim(hidden, 0) > 1) {
    mlx_array q, k, v, gate, mixed;
    tower_pre(c, &pw, hidden, codes, semantic, shared_input, offset, &q, &k, &v, &gate,
              &mixed);
    mlx_array out = tower_post(c, &qw, mixed, attend_cache(m, t, q, k, v, pos), gate);
    if (!is_null(head)) *head_logits = m_linear(out, head);
    return out;
  }
  ensure_compiled(m);
  if (is_null(shared_input)) shared_input = m_zeros(SH(1, c->t_dim), MLX_BFLOAT16);
  AttnShape shape = attn_shape(c);
  int blocks = attn_route(&shape, pos + 1);
  if (blocks >= 0) {
    mlx_array in[9 + PRE_W_COUNT + POST_W_COUNT] = {
        hidden, codes, semantic, shared_input, offset, t->k_cache, t->v_cache,
        attn_variant_tag(blocks)};
    memcpy(in + 8, &pw, sizeof(pw));
    memcpy(in + 8 + PRE_W_COUNT, &qw, sizeof(qw));
    mlx_array out[4];
    if (is_null(head)) {
      compiled_call(&m->tower_compiled.layer, in, 8 + PRE_W_COUNT + POST_W_COUNT, out);
    } else {
      in[8 + PRE_W_COUNT + POST_W_COUNT] = head;
      compiled_call(&m->tower_compiled.layer_head, in, 9 + PRE_W_COUNT + POST_W_COUNT, out);
      *head_logits = out[3];
    }
    cache_write_row(&t->k_cache, out[1], pos);
    cache_write_row(&t->v_cache, out[2], pos);
    return out[0];
  }
  mlx_array in[5 + PRE_W_COUNT] = {hidden, codes, semantic, shared_input, offset};
  memcpy(in + 5, &pw, sizeof(pw));
  mlx_array pre[5];
  compiled_call(&m->tower_compiled.pre, in, sizeof(in) / sizeof(in[0]), pre);
  mlx_array attended = attend_cache(m, t, pre[0], pre[1], pre[2], pos);
  mlx_array in2[3 + POST_W_COUNT + 1] = {pre[4], attended, pre[3]};
  memcpy(in2 + 3, &qw, sizeof(qw));
  if (is_null(head)) {
    mlx_array out;
    compiled_call(&m->tower_compiled.post, in2, 3 + POST_W_COUNT, &out);
    return out;
  }
  in2[3 + POST_W_COUNT] = head;
  mlx_array out[2];
  compiled_call(&m->tower_compiled.post_head, in2, 4 + POST_W_COUNT, out);
  *head_logits = out[1];
  return out[0];
}

static mlx_array is_semantic(Model *m, mlx_array tokens) {
  return m_logical_and(m_greater_equal(tokens, m_int(m->cfg.semantic_start)),
                       m_less_equal(tokens, m_int(m->cfg.semantic_end)));
}

/* Summed embeddings of codes first..end-1 of each frame row, [L, dim] bf16. */
static mlx_array embedding_sum(Model *m, mlx_array frames, int first, int end) {
  const Config *c = &m->cfg;
  int L = m_dim(frames, 0);
  if (first == end) return m_zeros(SH(L, c->t_dim), MLX_BFLOAT16);
  mlx_array acc = null_array();
  for (int j = first; j < end; j++) {
    mlx_array codes = m_reshape(m_slice_axis(frames, 1, 1 + j, 2 + j), SH(L));
    codes = m_add(codes, m_int(j * c->code_vocab));
    mlx_array e = m_astype(m_take(m->codebook_embeddings, codes, 0), MLX_FLOAT32);
    acc = is_null(acc) ? e : m_add(acc, e);
  }
  return m_astype(acc, MLX_BFLOAT16);
}

static mlx_array run_shared(Model *m, mlx_array hidden, mlx_array c0, mlx_array semantic,
                            int pos, mlx_array frames) {
  const Config *c = &m->cfg;
  int L = m_dim(frames, 0);
  mlx_array unseen = embedding_sum(m, frames, 1 + c->vq_backbone_codebooks, c->n_codebooks);
  mlx_array frame_semantic =
      is_semantic(m, m_reshape(m_slice_axis(frames, 1, 0, 1), SH(L)));
  unseen = m_where(m_expand_dims(frame_semantic, -1), unseen, m_scalar(0.0f, MLX_BFLOAT16));
  for (int i = 0; i < c->n_shared; i++)
    hidden = tower_layer(m, &m->shared[i], hidden, c0, semantic, pos,
                         i == 0 ? unseen : null_array(), null_array(), NULL);
  return hidden;
}

static mlx_array column(mlx_array frames, int col) {
  return m_reshape(m_slice_axis(frames, 1, col, col + 1), SH(m_dim(frames, 0)));
}

void tower_prefill(Model *m, mlx_array hidden, int pos, mlx_array frames,
                   mlx_array next_frames) {
  const Config *c = &m->cfg;
  mlx_array semantic = is_semantic(m, column(next_frames, 0));
  mlx_array h = run_shared(m, hidden, column(next_frames, 1), semantic, pos, frames);
  for (int j = 0; j < c->n_codebooks - 1; j++)
    h = tower_layer(m, &m->residual[j], h, column(next_frames, j + 1), semantic, pos,
                    null_array(), null_array(), NULL);
}

/* Candidate index -> token id: 0 is eos, i >= 1 is semantic_start + i - 1. */
static mlx_array candidate_token(Model *m, mlx_array idx) {
  return m_where(m_equal(idx, m_int(0)), m_int(m->cfg.eos_id),
                 m_add(idx, m_int(m->cfg.semantic_start - 1)));
}

mlx_array tower_decode_step(Model *m, mlx_array logits, mlx_array hidden, int pos,
                            mlx_array frame, mlx_array history, const SamplingParams *p,
                            mlx_array *backbone_input) {
  const Config *c = &m->cfg;
  /* every uniform draw of the step, in the order the samplers consume them, made
   * on the CPU up front so no random-number kernels sit between the GPU steps */
  int V = m_dim(logits, -1);
  mlx_array noise[MAX_CODEBOOKS + 2];
  int n_noise = 0;
  noise[n_noise++] = sample_noise(1, V, p->top_k);
  noise[n_noise++] = sample_noise(1, V, RAS_TOP_K);
  for (int j = 0; j < c->n_codebooks - 1; j++)
    noise[n_noise++] = sample_noise(1, m_dim(m->heads[j], 0), p->top_k);
  m_eval_many(noise, n_noise);
  mlx_array first = candidate_token(
      m, m_astype(sample_with_noise(logits, noise[0], p->temperature, p->top_p), MLX_INT32));
  mlx_array retry = candidate_token(
      m, m_astype(sample_with_noise(logits, noise[1], RAS_TEMPERATURE, RAS_TOP_P), MLX_INT32));
  /* Only audio tokens are resampled; a stop must not be redrawn. */
  mlx_array repeated =
      m_any(m_equal(history, m_reshape(first, SH(1))));
  repeated = m_logical_and(repeated, m_reshape(is_semantic(m, first), SH(-1)));
  mlx_array main_tok = m_where(repeated, retry, first); /* [1] */
  mlx_array c0 = m_clip(m_sub(main_tok, m_int(c->semantic_start)), m_int(0),
                        m_int(c->code_vocab - 1));
  mlx_array semantic = is_semantic(m, main_tok);
  mlx_array h = run_shared(m, hidden, c0, semantic, pos, frame);
  mlx_array codes[MAX_CODEBOOKS + 1];
  codes[0] = main_tok;
  codes[1] = c0;
  for (int j = 0; j < c->n_codebooks - 1; j++) {
    mlx_array head_logits;
    h = tower_layer(m, &m->residual[j], h, codes[j + 1], semantic, pos, null_array(),
                    m->heads[j], &head_logits);
    codes[j + 2] = m_astype(
        sample_with_noise(head_logits, noise[2 + j], p->temperature, p->top_p), MLX_INT32);
    if (backbone_input && j + 2 == 1 + c->vq_backbone_codebooks)
      *backbone_input = m_stack(codes, j + 3, -1);
  }
  if (backbone_input && c->vq_backbone_codebooks == 0) *backbone_input = m_stack(codes, 2, -1);
  return m_stack(codes, c->n_codebooks + 1, -1); /* [1, 1 + N] */
}

mlx_array tower_forced_logits(Model *m, mlx_array hidden, int pos, mlx_array frame,
                              mlx_array next_frame) {
  const Config *c = &m->cfg;
  mlx_array semantic = is_semantic(m, column(next_frame, 0));
  mlx_array h = run_shared(m, hidden, column(next_frame, 1), semantic, pos, frame);
  mlx_array logits[MAX_CODEBOOKS];
  for (int j = 0; j < c->n_codebooks - 1; j++)
    h = tower_layer(m, &m->residual[j], h, column(next_frame, j + 1), semantic, pos,
                    null_array(), m->heads[j], &logits[j]);
  return m_concat(logits, c->n_codebooks - 1, 0);
}
