/* Qwen3.5-MoE hybrid backbone, following fish_speech fish_omni backbone.py /
 * layers.py / moe.py, including where the reference computes in fp32. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"

/* ---- shared helpers ---- */

/* ZeroCenteredRMSNorm: fp32 rmsnorm times fp32 (1 + w), back to x's dtype. */
static mlx_array zc_norm(mlx_array x, mlx_array w1, float eps) {
  return m_rms_norm_f32(x, w1, eps, m_dtype(x));
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

/* ---- fused single-row gated delta net ----
 * One kernel for a decode step of the delta net, after the fused input
 * projection: causal conv taps + SiLU, q/k L2 norm, beta / decay gates, the
 * recurrence, and the GatedRMSNorm with SiLU(z). It performs the same sequence of
 * fp32 / bf16 roundings as gdn_mixer (the multi-row path), using MLX's own
 * sigmoid / logaddexp / log1p formulas. One threadgroup per value head. */

static const char *GDN_STEP_HEADER =
    "inline float d3_sigmoid(float x) {\n"
    "  float y = 1 / (1 + metal::precise::exp(metal::abs(x)));\n"
    "  return (x < 0) ? y : 1 - y;\n"
    "}\n"
    "/* MLX's log1p and LogAddExp, with the precise exp / log of MLX's prebuilt\n"
    "   kernels (a JIT-compiled metal::exp is the fast one) */\n"
    "inline float d3_log1p(float x) {\n"
    "  float xp1 = 1.0f + x;\n"
    "  if (xp1 == FLT_MAX) return FLT_MAX;\n"
    "  if (xp1 == 1.0f) return x;\n"
    "  return x * (metal::precise::log(xp1) / (xp1 - 1.0f));\n"
    "}\n"
    "inline float d3_softplus(float x) {\n"
    "  if (metal::isnan(x)) return NAN;\n"
    "  float mx = metal::max(x, 0.0f), mn = metal::min(x, 0.0f);\n"
    "  if (mn == -INFINITY || mx == INFINITY) return mx;\n"
    "  return mx + d3_log1p(metal::precise::exp(mn - mx));\n"
    "}\n";

static const char *GDN_STEP_SOURCE =
    "  constexpr int KEY_DIM = NK * DK;\n"
    "  constexpr int VALUE_DIM = NV * DV;\n"
    "  constexpr int CONV_DIM = 2 * KEY_DIM + VALUE_DIM;\n"
    "  constexpr int HIST = KC - 1;\n"
    "  uint hv = threadgroup_position_in_grid.x;\n"
    "  uint tid = thread_position_in_threadgroup.x;\n"
    "  uint lane = thread_index_in_simdgroup;\n"
    "  uint sg = simdgroup_index_in_threadgroup;\n"
    "  uint hk = hv / (NV / NK);\n"
    "  threadgroup float qs[DK], ks[DK], vs[DV], ys[DV], red[40], gb[2];\n"
    "  /* this simdgroup's recurrent state rows, loaded up front so the reads\n"
    "   * overlap the conv and gate phases (32 simdgroups, DV / 32 rows each) */\n"
    "  constexpr int NP = DK / 32;\n"
    "  constexpr int RPS = DV / 32;\n"
    "  float st[RPS][NP];\n"
    "  for (int j = 0; j < RPS; j++) {\n"
    "    auto si = state_in + ((size_t)hv * DV + sg + 32 * j) * DK;\n"
    "    for (int i = 0; i < NP; i++) st[j][i] = si[NP * lane + i];\n"
    "  }\n"
    "  if (tid < (uint)(2 * DK + DV)) {\n"
    "    int which = tid < DK ? 0 : (tid < 2 * DK ? 1 : 2);\n"
    "    int j = which == 0 ? (int)tid : (which == 1 ? (int)tid - DK : (int)tid - 2 * DK);\n"
    "    int ch = which == 0 ? (int)hk * DK + j\n"
    "           : (which == 1 ? KEY_DIM + (int)hk * DK + j : 2 * KEY_DIM + (int)hv * DV + j);\n"
    "    float acc = 0.0f;\n"
    "    for (int t = 0; t < KC; t++) {\n"
    "      float p = t < HIST ? float(conv_state[t * CONV_DIM + ch]) : float(qkv[ch]);\n"
    "      /* each tap rounded on its own, as the op path's multiply then add */\n"
    "      float term = fma(p, float(conv_w[ch * KC + t]), 0.0f);\n"
    "      acc = t == 0 ? term : acc + term;\n"
    "    }\n"
    "    float a = float(static_cast<bfloat16_t>(acc));\n"
    "    float m = float(static_cast<bfloat16_t>(a * d3_sigmoid(a)));\n"
    "    if (which == 0) qs[j] = m; else if (which == 1) ks[j] = m; else vs[j] = m;\n"
    "    if (which == 2 || (hv % (NV / NK)) == 0) {\n"
    "      for (int t = 0; t < HIST - 1; t++)\n"
    "        conv_out[t * CONV_DIM + ch] = conv_state[(t + 1) * CONV_DIM + ch];\n"
    "      conv_out[(HIST - 1) * CONV_DIM + ch] = qkv[ch];\n"
    "    }\n"
    "  } else if (tid == (uint)(2 * DK + DV)) {\n"
    "    float b = float(b_in[hv]);\n"
    "    float a = float(a_in[hv]);\n"
    "    float decay = -metal::precise::exp(float(a_log[hv])) * d3_softplus(a + float(dt_bias[hv]));\n"
    "    gb[0] = metal::precise::exp(decay);\n"
    "    gb[1] = d3_sigmoid(b);\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  /* l2norm_f32's sum of squares, as MLX reduces a row of 128: rounded\n"
    "   * squares, 4 contiguous per lane summed in order, then simd_sum */\n"
    "  if (sg < 2) {\n"
    "    threadgroup float *src = sg == 0 ? qs : ks;\n"
    "    float acc = 0.0f;\n"
    "    for (int i = 4 * lane; i < 4 * (int)lane + 4; i++) {\n"
    "      float sq = fma(src[i], src[i], 0.0f);\n"
    "      acc = sq + acc;\n"
    "    }\n"
    "    acc = simd_sum(acc);\n"
    "    if (lane == 0) red[sg] = acc;\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  float rq = metal::precise::rsqrt(red[0] + 1e-6f);\n"
    "  float rk = metal::precise::rsqrt(red[1] + 1e-6f);\n"
    "  float g = gb[0], beta = gb[1];\n"

    "  for (int j = 0; j < RPS; j++) {\n"
    "    int r = sg + 32 * j;\n"
    "    auto so = state_out + ((size_t)hv * DV + r) * DK;\n"
    "    float kv = 0.0f;\n"
    "    for (int i = 0; i < NP; i++) {\n"
    "      int d = NP * lane + i;\n"
    "      st[j][i] = st[j][i] * g;\n"
    "      kv += st[j][i] * (ks[d] * rk);\n"
    "    }\n"
    "    kv = simd_sum(kv);\n"
    "    float delta = (vs[r] - kv) * beta;\n"
    "    float out = 0.0f;\n"
    "    for (int i = 0; i < NP; i++) {\n"
    "      int d = NP * lane + i;\n"
    "      st[j][i] = st[j][i] + (ks[d] * rk) * delta;\n"
    "      out += st[j][i] * ((qs[d] * rq) * SCALE);\n"
    "      so[d] = st[j][i];\n"
    "    }\n"
    "    out = simd_sum(out);\n"
    "    if (lane == 0) ys[r] = out;\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  /* GatedRMSNorm's rms_norm, MLX's rms_single_row for a row of DV: DV / 4\n"
    "   * threads, 4 contiguous elements each */\n"
    "  if (sg == 0) {\n"
    "    float acc = 0.0f;\n"
    "    for (int i = 4 * lane; i < 4 * (int)lane + 4; i++) {\n"
    "      float v = float(static_cast<bfloat16_t>(ys[i]));\n"
    "      acc += v * v;\n"
    "    }\n"
    "    acc = simd_sum(acc);\n"
    "    if (lane == 0) red[2] = acc;\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (tid < (uint)DV) {\n"
    "    float cf = float(static_cast<bfloat16_t>(ys[tid]));\n"
    "    float inv = metal::precise::rsqrt(red[2] / float(DV) + EPS);\n"
    "    float n = float(static_cast<bfloat16_t>(cf * inv));\n"
    "    float nw = float(static_cast<bfloat16_t>(n * float(gnorm[tid])));\n"
    "    float zf = float(z[hv * DV + tid]);\n"
    "    gated[hv * DV + tid] = static_cast<bfloat16_t>(nw * (zf * d3_sigmoid(zf)));\n"
    "  }\n";

static mlx_fast_metal_kernel gdn_step_kernel;
static bool gdn_step_ready;

/* Single-row delta net from the projections qkv, z, b, a [1, 1, *]: the gated core
 * [1, 1, value_dim] (input of out_proj) and the new conv / recurrent state. */
static void gdn_step(const Config *c, mlx_array qkv, mlx_array z, mlx_array b, mlx_array a,
                     mlx_array conv, mlx_array conv_w, mlx_array a_log, mlx_array dt_bias,
                     mlx_array rec, mlx_array gdn_norm, mlx_array *gated, mlx_array *conv_out,
                     mlx_array *rec_out) {
  if (!gdn_step_ready) {
    char header[2048];
    snprintf(header, sizeof(header), "constant float EPS = %.9g;\nconstant float SCALE = %.9g;\n%s",
             c->rms_eps, 1.0 / sqrt((double)c->lin_key_dim), GDN_STEP_HEADER);
    const char *in_names[] = {"qkv", "z", "b_in", "a_in", "conv_state", "conv_w", "a_log",
                              "dt_bias", "state_in", "gnorm"};
    const char *out_names[] = {"gated", "conv_out", "state_out"};
    mlx_vector_string ins = mlx_vector_string_new_data(in_names, 10);
    mlx_vector_string outs = mlx_vector_string_new_data(out_names, 3);
    gdn_step_kernel = mlx_fast_metal_kernel_new("drama3_gdn_step", ins, outs, GDN_STEP_SOURCE,
                                                header, true, false);
    mlx_vector_string_free(ins);
    mlx_vector_string_free(outs);
    gdn_step_ready = true;
  }
  int nk = c->lin_n_key, nv = c->lin_n_value, dk = c->lin_key_dim, dv = c->lin_value_dim;
  if (dk != 128 || dv != 128) die("gated delta step kernel: head dims of 128 expected");
  int conv_dim = 2 * nk * dk + nv * dv;
  mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){1, 1, nv * dv}, 3, MLX_BFLOAT16);
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){1, c->conv_kernel - 1, conv_dim}, 3,
                                              MLX_BFLOAT16);
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){1, nv, dv, dk}, 4, MLX_FLOAT32);
  mlx_fast_metal_kernel_config_set_grid(cfg, 1024 * nv, 1, 1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NK", nk);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NV", nv);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "DK", dk);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "DV", dv);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KC", c->conv_kernel);
  mlx_array inputs[] = {qkv, z, b, a, conv, conv_w, a_log, dt_bias, rec, gdn_norm};
  mlx_vector_array in_vec = mlx_vector_array_new_data(inputs, 10);
  mlx_vector_array out_vec = mlx_vector_array_new();
  if (mlx_fast_metal_kernel_apply(&out_vec, gdn_step_kernel, in_vec, cfg, g_stream))
    die("gated delta step kernel failed");
  mlx_array outs[3];
  for (int i = 0; i < 3; i++) {
    outs[i] = mlx_array_new();
    mlx_vector_array_get(&outs[i], out_vec, i);
    track(outs[i]);
  }
  *gated = outs[0];
  *conv_out = outs[1];
  *rec_out = outs[2];
  mlx_vector_array_free(in_vec);
  mlx_vector_array_free(out_vec);
  mlx_fast_metal_kernel_config_free(cfg);
}

/* ---- fused single-row MoE routing ----
 * Exactly the op path of moe_parts for one row: softmax of the fp32 router logits
 * with MLX's softmax_single_row (4 contiguous logits per thread, fast exp, times
 * the reciprocal sum), the top K of a stable ascending argsort (so ties at the
 * K-th place go to the higher id) in that ascending order, normalized by their
 * sum taken in that order and rounded through bf16. */

static const char *ROUTER_SOURCE =
    "  constexpr int NR = 4;\n"
    "  constexpr int NT = E / NR;\n"
    "  constexpr int NSG = (NT + 31) / 32;\n"
    "  uint tid = thread_position_in_threadgroup.x;\n"
    "  uint lane = thread_index_in_simdgroup;\n"
    "  uint sg = simdgroup_index_in_threadgroup;\n"
    "  threadgroup float local_max[32];\n"
    "  threadgroup float local_norm[32];\n"
    "  threadgroup float cval[NSG * K];\n"
    "  threadgroup int cidx[NSG * K];\n"
    "  float ld[NR];\n"
    "  for (int i = 0; i < NR; i++) ld[i] = float(proj[tid * NR + i]);\n"
    "  if (sg == 0) { local_max[lane] = -INFINITY; local_norm[lane] = 0; }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  float maxval = -FLT_MAX;\n"
    "  for (int i = 0; i < NR; i++) maxval = (maxval < ld[i]) ? ld[i] : maxval;\n"
    "  maxval = simd_max(maxval);\n"
    "  if (lane == 0) local_max[sg] = maxval;\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (sg == 0) {\n"
    "    maxval = simd_max(local_max[lane]);\n"
    "    if (lane == 0) local_max[0] = maxval;\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  maxval = local_max[0];\n"
    "  float normalizer = 0;\n"
    "  for (int i = 0; i < NR; i++) {\n"
    "    float e = fast::exp(ld[i] - maxval);\n"
    "    ld[i] = e;\n"
    "    normalizer += e;\n"
    "  }\n"
    "  normalizer = simd_sum(normalizer);\n"
    "  if (lane == 0) local_norm[sg] = normalizer;\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (sg == 0) {\n"
    "    normalizer = simd_sum(local_norm[lane]);\n"
    "    if (lane == 0) local_norm[0] = normalizer;\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  normalizer = 1 / local_norm[0];\n"
    "  float p[NR];\n"
    "  for (int i = 0; i < NR; i++) p[i] = ld[i] * normalizer;\n"
    "  /* each simdgroup's top K, ranked by probability then by the higher id\n"
    "   * (a stable ascending sort puts equal values in id order) */\n"
    "  bool taken[NR] = {false, false, false, false};\n"
    "  for (int r = 0; r < K; r++) {\n"
    "    float bv = -1.0f;\n"
    "    int bi = -1, bj = -1;\n"
    "    for (int i = 0; i < NR; i++) {\n"
    "      int idx = (int)tid * NR + i;\n"
    "      if (!taken[i] && (p[i] > bv || (p[i] == bv && idx > bi))) { bv = p[i]; bi = idx; bj = i; }\n"
    "    }\n"
    "    float best = simd_max(bv);\n"
    "    int win = simd_max(bv == best ? bi : -1);\n"
    "    if (bj >= 0 && bi == win) taken[bj] = true;\n"
    "    if (lane == 0) { cval[sg * K + r] = best; cidx[sg * K + r] = win; }\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (sg == 0) {\n"
    "    constexpr int NC = NSG * K;\n"
    "    constexpr int PER = (NC + 31) / 32;\n"
    "    float v[PER];\n"
    "    int id[PER];\n"
    "    for (int j = 0; j < PER; j++) {\n"
    "      uint c = lane + 32 * j;\n"
    "      v[j] = c < (uint)NC ? cval[c] : -1.0f;\n"
    "      id[j] = c < (uint)NC ? cidx[c] : -1;\n"
    "    }\n"
    "    float fv[K];\n"
    "    int fi[K];\n"
    "    for (int r = 0; r < K; r++) {\n"
    "      float lb = -2.0f;\n"
    "      int li = -1, lj = 0;\n"
    "      for (int j = 0; j < PER; j++)\n"
    "        if (v[j] > lb || (v[j] == lb && id[j] > li)) { lb = v[j]; li = id[j]; lj = j; }\n"
    "      float best = simd_max(lb);\n"
    "      int win = simd_max(lb == best ? li : -1);\n"
    "      if (lb == best && li == win) v[lj] = -1.5f;\n"
    "      fv[r] = best;\n"
    "      fi[r] = win;\n"
    "    }\n"
    "    if (lane == 0) {\n"
    "      /* ascending order, as the argsort leaves them; the sum in that order */\n"
    "      float sum = 0.0f;\n"
    "      for (int r = K - 1; r >= 0; r--) sum = fv[r] + sum;\n"
    "      for (int r = 0; r < K; r++) {\n"
    "        float s = fv[K - 1 - r];\n"
    "        float w = NORM ? s / sum : s;\n"
    "        weights[r] = float(static_cast<bfloat16_t>(w));\n"
    "        inds[r] = (uint)fi[K - 1 - r];\n"
    "      }\n"
    "    }\n"
    "  }\n";

static mlx_fast_metal_kernel router_kernel;
static bool router_ready;

/* inds [1, K] uint32 and weights [1, K] fp32 from the router logits [1, E]. */
static void route_step(const Config *c, mlx_array proj, mlx_array *inds, mlx_array *weights) {
  if (!router_ready) {
    const char *in_names[] = {"proj"};
    const char *out_names[] = {"inds", "weights"};
    mlx_vector_string ins = mlx_vector_string_new_data(in_names, 1);
    mlx_vector_string outs = mlx_vector_string_new_data(out_names, 2);
    router_kernel = mlx_fast_metal_kernel_new("drama3_moe_route", ins, outs, ROUTER_SOURCE, "",
                                              true, false);
    mlx_vector_string_free(ins);
    mlx_vector_string_free(outs);
    router_ready = true;
  }
  if (c->n_experts % 128 || c->n_experts > 4096 || c->top_k > 32)
    die("router kernel needs 128 | E <= 4096 and K <= 32");
  mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){1, c->top_k}, 2, MLX_UINT32);
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){1, c->top_k}, 2, MLX_FLOAT32);
  mlx_fast_metal_kernel_config_set_grid(cfg, c->n_experts / 4, 1, 1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, c->n_experts / 4, 1, 1);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "E", c->n_experts);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", c->top_k);
  mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "NORM", c->norm_topk);
  mlx_vector_array in_vec = mlx_vector_array_new_data(&proj, 1);
  mlx_vector_array out_vec = mlx_vector_array_new();
  if (mlx_fast_metal_kernel_apply(&out_vec, router_kernel, in_vec, cfg, g_stream))
    die("router kernel failed");
  mlx_array a = mlx_array_new(), b = mlx_array_new();
  mlx_vector_array_get(&a, out_vec, 0);
  mlx_vector_array_get(&b, out_vec, 1);
  *inds = track(a);
  *weights = track(b);
  mlx_vector_array_free(in_vec);
  mlx_vector_array_free(out_vec);
  mlx_fast_metal_kernel_config_free(cfg);
}

/* ---- fused single-row MoE output ----
 * routed = bf16(sum over k of e_k * w_k) in fp32, adding k in order as MLX's
 * strided sum does; moe = routed + shared; x' = x + moe (bf16 adds, as m_add);
 * then the next zero-centered norm of x' with m_rms_norm_f32's reduction. One
 * threadgroup. */

static const char *COMBINE_SOURCE =
    "  uint tid = thread_position_in_threadgroup.x;\n"
    "  uint tg = threads_per_threadgroup.x;\n"
    "  /* MLX's rms_single_row for the norm: 4 contiguous elements per thread */\n"
    "  float acc = 0.0f;\n"
    "  for (uint i = 4 * tid; i < 4 * tid + 4 && i < D; i++) {\n"
    "    float r = 0.0f;\n"
    "    for (uint k = 0; k < K; k++) {\n"
    "      float p = float(e[k * D + i]) * w[k];\n"
    "      r = r + p;\n"
    "    }\n"
    "    T moe = static_cast<T>(float(static_cast<T>(r)) + float(shared[i]));\n"
    "    T s = static_cast<T>(float(x[i]) + float(moe));\n"
    "    x_out[i] = s;\n"
    "    float v = float(s);\n"
    "    acc += v * v;\n"
    "  }\n"
    "  threadgroup float partial[32];\n"
    "  acc = simd_sum(acc);\n"
    "  uint lane = tid % 32, sg = tid / 32;\n"
    "  if (lane == 0) partial[sg] = acc;\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (sg == 0) {\n"
    "    float v = lane < (tg + 31) / 32 ? partial[lane] : 0.0f;\n"
    "    v = simd_sum(v);\n"
    "    if (lane == 0) partial[0] = v;\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  float inv = metal::precise::rsqrt(partial[0] / float(D) + EPS);\n"
    "  for (uint i = 4 * tid; i < 4 * tid + 4 && i < D; i++) {\n"
    "    float n = float(x_out[i]) * inv;\n"
    "    n = n * nw[i];\n"
    "    h_out[i] = static_cast<T>(n);\n"
    "  }\n";

static mlx_fast_metal_kernel combine_kernel;
static float combine_eps;
static bool combine_ready;

/* expert_out [1, K, 1, H] bf16, weights [1, K] fp32, shared [1, H] and the
 * residual x [1, 1, H]: x' = x + moe and h = norm(x') with norm_w, both [1, 1, H]. */
static void moe_combine(const Config *c, mlx_array expert_out, mlx_array weights,
                        mlx_array shared, mlx_array x, mlx_array norm_w, mlx_array *x_out,
                        mlx_array *h_out) {
  if (!combine_ready) {
    const char *in_names[] = {"e", "w", "shared", "x", "nw"};
    const char *out_names[] = {"x_out", "h_out"};
    char header[64];
    snprintf(header, sizeof(header), "constant float EPS = %.9g;\n", c->rms_eps);
    mlx_vector_string ins = mlx_vector_string_new_data(in_names, 5);
    mlx_vector_string outs = mlx_vector_string_new_data(out_names, 2);
    combine_kernel = mlx_fast_metal_kernel_new("drama3_moe_combine", ins, outs, COMBINE_SOURCE,
                                               header, true, false);
    mlx_vector_string_free(ins);
    mlx_vector_string_free(outs);
    combine_eps = c->rms_eps;
    combine_ready = true;
  }
  if (combine_eps != c->rms_eps) die("combine kernel built for another eps");
  int H = c->hidden;
  mlx_dtype dtype = m_dtype(x);
  mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){1, 1, H}, 3, dtype);
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){1, 1, H}, 3, dtype);
  int tg = 32 * (((H + 3) / 4 + 31) / 32); /* MLX's rms_single_row threadgroup */
  mlx_fast_metal_kernel_config_set_grid(cfg, tg, 1, 1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", H);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", c->top_k);
  mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dtype);
  mlx_array inputs[5] = {expert_out, weights, shared, x, norm_w};
  mlx_vector_array in_vec = mlx_vector_array_new_data(inputs, 5);
  mlx_vector_array out_vec = mlx_vector_array_new();
  if (mlx_fast_metal_kernel_apply(&out_vec, combine_kernel, in_vec, cfg, g_stream))
    die("combine kernel failed");
  mlx_array a = mlx_array_new(), b = mlx_array_new();
  mlx_vector_array_get(&a, out_vec, 0);
  mlx_vector_array_get(&b, out_vec, 1);
  *x_out = track(a);
  *h_out = track(b);
  mlx_vector_array_free(in_vec);
  mlx_vector_array_free(out_vec);
  mlx_fast_metal_kernel_config_free(cfg);
}

/* ---- layer math: pure functions of their inputs ----
 * The same functions run eagerly for prefill and inside compiled closures for
 * single-row decode, so both paths share one definition of the math. */

typedef struct {
  mlx_array in_qkv, in_z, in_b, in_a, conv_w, dt_bias, a_log, gdn_norm, out_proj;
} GdnW;

typedef struct {
  mlx_array q_proj, k_proj, v_proj, q_norm, k_norm, o_proj;
} AttnW;

typedef struct {
  mlx_array post_norm, router, sh_gate, sh_up, sh_expert_gate, sh_down;
  mlx_array gu_w, gu_s, gu_b, down_w, down_s, down_b;
  mlx_array lhs_up, lhs_down; /* single-row gather indices */
} MoeW;

#define GDN_W_COUNT 9
#define ATTN_W_COUNT 6
#define MOE_W_COUNT 14

static GdnW gdn_weights(const BackboneLayer *L) {
  return (GdnW){L->in_qkv, L->in_z,     L->in_b,     L->in_a,    L->conv_w,
                L->dt_bias, L->a_log, L->gdn_norm, L->out_proj};
}

static AttnW attn_weights(const BackboneLayer *L) {
  return (AttnW){L->q_proj, L->k_proj, L->v_proj, L->q_norm, L->k_norm, L->o_proj};
}

static MoeW moe_weights(const Model *m, const BackboneLayer *L) {
  return (MoeW){L->post_norm, L->router, L->sh_gate, L->sh_up,     L->sh_expert_gate,
                L->sh_down,   L->gu_w,   L->gu_s,    L->gu_b,      L->down_w,
                L->down_s,    L->down_b, m->moe_lhs_up, m->moe_lhs_down};
}

/* GatedDeltaNet on the normed input h [1, T, H] (bf16): the mixer output and the
 * updated conv / recurrent state. */
static void gdn_mixer(const Config *c, const GdnW *w, mlx_array h, mlx_array conv,
                      mlx_array rec, mlx_array *out, mlx_array *conv_out, mlx_array *rec_out) {
  int T = m_dim(h, 1);
  int key_dim = c->lin_n_key * c->lin_key_dim;
  int value_dim = c->lin_n_value * c->lin_value_dim;
  int nv = c->lin_n_value;
  int hist = c->conv_kernel - 1;

  if (T == 1 && !getenv("D3_NO_FUSED_GDN")) {
    mlx_array gated;
    gdn_step(c, m_linear(h, w->in_qkv), m_linear(h, w->in_z), m_linear(h, w->in_b),
             m_linear(h, w->in_a), conv, w->conv_w, w->a_log, w->dt_bias, rec, w->gdn_norm,
             &gated, conv_out, rec_out);
    *out = m_linear(gated, w->out_proj);
    return;
  }

  mlx_array proj[4] = {m_linear(h, w->in_qkv), m_linear(h, w->in_z), m_linear(h, w->in_b),
                       m_linear(h, w->in_a)};
  mlx_array padded = m_concat2(conv, proj[0], 1);
  *conv_out = m_slice_axis(padded, 1, T, T + hist);

  /* depthwise causal conv: out[t] = sum_j w[:, j] * padded[t + j], fp32 accumulation */
  mlx_array pf = m_astype(padded, MLX_FLOAT32);
  mlx_array wf = m_astype(w->conv_w, MLX_FLOAT32); /* [C, K] */
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
  mlx_array v = m_reshape(parts[2], SH(1, T, nv, c->lin_value_dim));
  mlx_array z = m_reshape(proj[1], SH(1, T, nv, c->lin_value_dim));
  mlx_array beta = m_sigmoid(m_astype(proj[2], MLX_FLOAT32));
  mlx_array a = m_astype(proj[3], MLX_FLOAT32);
  mlx_array decay = m_mul(m_neg(m_exp(m_astype(w->a_log, MLX_FLOAT32))),
                          m_softplus(m_add(a, m_astype(w->dt_bias, MLX_FLOAT32))));
  mlx_array g = m_exp(decay);

  mlx_array qf = m_mul(l2norm_f32(m_astype(q, MLX_FLOAT32)),
                       m_float(1.0f / sqrtf((float)c->lin_key_dim)));
  mlx_array kf = l2norm_f32(m_astype(k, MLX_FLOAT32));
  mlx_array vf = m_astype(v, MLX_FLOAT32);
  mlx_array y;
  gated_delta(qf, kf, vf, g, beta, rec, &y, rec_out);

  /* GatedRMSNorm: weight * rmsnorm(core).to(bf16), then * silu(z) in fp32 */
  mlx_array core = m_astype(y, MLX_BFLOAT16);
  mlx_array normed =
      m_astype(m_rms_norm(m_astype(core, MLX_FLOAT32), null_array(), c->rms_eps), MLX_BFLOAT16);
  normed = m_mul(w->gdn_norm, normed);
  mlx_array zf = m_astype(z, MLX_FLOAT32);
  mlx_array gated = m_astype(m_mul(m_astype(normed, MLX_FLOAT32), m_mul(zf, m_sigmoid(zf))),
                             MLX_BFLOAT16);
  *out = m_linear(m_reshape(gated, SH(1, T, value_dim)), w->out_proj);
}

/* Gated attention, before the KV cache: q / k ([1, heads, T, hd], RoPE applied at
 * position offset), v and the output gate [1, T, heads * hd]. */
static void attn_project(const Config *c, const AttnW *w, mlx_array h, mlx_array offset,
                         mlx_array *q, mlx_array *k, mlx_array *v, mlx_array *gate) {
  int T = m_dim(h, 1), hd = c->head_dim;
  mlx_array proj[3] = {m_linear(h, w->q_proj), m_linear(h, w->k_proj), m_linear(h, w->v_proj)};
  mlx_array qparts[2];
  m_split(m_reshape(proj[0], SH(1, T, c->n_heads, 2 * hd)), (int[]){hd}, 1, -1, qparts);
  *gate = m_reshape(qparts[1], SH(1, T, c->n_heads * hd));
  mlx_array qn = zc_norm(qparts[0], w->q_norm, c->rms_eps);
  mlx_array kn = zc_norm(m_reshape(proj[1], SH(1, T, c->n_kv_heads, hd)), w->k_norm, c->rms_eps);
  *q = m_rope_dyn(m_transpose(qn, SH(0, 2, 1, 3)), c->rotary_dim, false, c->rope_theta, offset);
  *k = m_rope_dyn(m_transpose(kn, SH(0, 2, 1, 3)), c->rotary_dim, false, c->rope_theta, offset);
  *v = m_transpose(m_reshape(proj[2], SH(1, T, c->n_kv_heads, hd)), SH(0, 2, 1, 3));
}

static mlx_array attn_output(const Config *c, const AttnW *w, mlx_array attended,
                             mlx_array gate) {
  int T = m_dim(attended, 2);
  mlx_array out =
      m_reshape(m_transpose(attended, SH(0, 2, 1, 3)), SH(1, T, c->n_heads * c->head_dim));
  return m_linear(m_mul(out, m_sigmoid(gate)), w->o_proj);
}

/* ---- fused single-row expert up projection ----
 * gate_up rows of the K routed experts (8-bit affine) times x, with
 * h = silu(gate) * up as the epilogue. Each lane does exactly the arithmetic of
 * MLX's qmv_fast (8 consecutive values per 256-value block: scale * sum(x * w) +
 * bias * sum(x), blocks in order, then simd_sum), so gate and up are the values
 * gather_qmm produces; the epilogue rounds like m_silu_f32 and m_mul. A simdgroup
 * takes RPS gate rows and the matching up rows per pass over x. */

static const char *MOE_UP_SOURCE =
    "  constexpr int BLOCK = 256;\n"
    "  constexpr int ROW_GROUPS = H / GS;\n"
    "  uint lane = thread_index_in_simdgroup;\n"
    "  uint sg = simdgroup_index_in_threadgroup;\n"
    "  uint slot = threadgroup_position_in_grid.y;\n"
    "  uint e = inds[slot];\n"
    "  int row0 = (threadgroup_position_in_grid.x * NSG + sg) * RPS;\n"
    "  const device uint8_t* wb = (const device uint8_t*)w + (size_t)e * 2 * I * H + lane * 8;\n"
    "  const device T* sb = scales + (size_t)e * 2 * I * ROW_GROUPS + lane / 8;\n"
    "  const device T* bb = biases + (size_t)e * 2 * I * ROW_GROUPS + lane / 8;\n"
    "  const device T* xp = x + lane * 8;\n"
    "  float res[2 * RPS] = {0};\n"
    "  for (int blk = 0; blk < H / BLOCK; blk++) {\n"
    "    float xt[8];\n"
    "    float sum = 0;\n"
    "    for (int i = 0; i < 8; i++) {\n"
    "      sum += xp[i];\n"
    "      xt[i] = xp[i];\n"
    "    }\n"
    "    for (int r = 0; r < 2 * RPS; r++) {\n"
    "      int row = r < RPS ? row0 + r : I + row0 + r - RPS;\n"
    "      const device uint8_t* wl = wb + (size_t)row * H + blk * BLOCK;\n"
    "      float sc = sb[(size_t)row * ROW_GROUPS + blk * (BLOCK / GS)];\n"
    "      float bi = bb[(size_t)row * ROW_GROUPS + blk * (BLOCK / GS)];\n"
    "      float accum = 0;\n"
    "      for (int i = 0; i < 8; i++) accum += xt[i] * wl[i];\n"
    "      res[r] += sc * accum + sum * bi;\n"
    "    }\n"
    "    xp += BLOCK;\n"
    "  }\n"
    "  for (int r = 0; r < 2 * RPS; r++) res[r] = simd_sum(res[r]);\n"
    "  if (lane == 0) {\n"
    "    for (int r = 0; r < RPS; r++) {\n"
    "      float g = static_cast<float>(static_cast<T>(res[r]));\n"
    "      auto y = 1 / (1 + metal::precise::exp(metal::abs(g)));\n"
    "      float sig = (g < 0) ? y : 1 - y;\n"
    "      T silu = static_cast<T>(g * sig);\n"
    "      h[slot * I + row0 + r] = silu * static_cast<T>(res[RPS + r]);\n"
    "    }\n"
    "  }\n";

static mlx_fast_metal_kernel moe_up_kernel;
static bool moe_up_ready;

#define MOE_UP_RPS 4
#define MOE_UP_NSG 4

/* x [1, H] normed input, inds [1, K] experts: h [1, K, 1, inter]. */
static mlx_array moe_up_fused(const Config *c, const MoeW *w, mlx_array x, mlx_array inds) {
  if (!moe_up_ready) {
    const char *in_names[] = {"x", "w", "scales", "biases", "inds"};
    const char *out_names[] = {"h"};
    mlx_vector_string ins = mlx_vector_string_new_data(in_names, 5);
    mlx_vector_string outs = mlx_vector_string_new_data(out_names, 1);
    moe_up_kernel =
        mlx_fast_metal_kernel_new("drama3_moe_up", ins, outs, MOE_UP_SOURCE, "", true, false);
    mlx_vector_string_free(ins);
    mlx_vector_string_free(outs);
    moe_up_ready = true;
  }
  int K = c->top_k, inter = c->moe_inter, H = c->hidden;
  int rows_per_tg = MOE_UP_RPS * MOE_UP_NSG;
  mlx_dtype dtype = m_dtype(x);
  mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){1, K, 1, inter}, 4, dtype);
  mlx_fast_metal_kernel_config_set_grid(cfg, 32 * MOE_UP_NSG * (inter / rows_per_tg), K, 1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * MOE_UP_NSG, 1, 1);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", H);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "I", inter);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", c->expert_group);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RPS", MOE_UP_RPS);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSG", MOE_UP_NSG);
  mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dtype);
  mlx_array inputs[5] = {x, w->gu_w, w->gu_s, w->gu_b, inds};
  mlx_vector_array in_vec = mlx_vector_array_new_data(inputs, 5);
  mlx_vector_array out_vec = mlx_vector_array_new();
  if (mlx_fast_metal_kernel_apply(&out_vec, moe_up_kernel, in_vec, cfg, g_stream))
    die("expert up kernel failed");
  mlx_array r = mlx_array_new();
  mlx_vector_array_get(&r, out_vec, 0);
  mlx_vector_array_free(in_vec);
  mlx_vector_array_free(out_vec);
  mlx_fast_metal_kernel_config_free(cfg);
  return track(r);
}

static bool moe_up_supported(const Config *c) {
  static int off = -1;
  if (off < 0) off = getenv("D3_NO_FUSED_MOE_UP") != NULL;
  return !off && c->expert_bits == 8 && c->hidden % 256 == 0 && c->expert_group % 8 == 0 &&
         256 % c->expert_group == 0 && c->moe_inter % (MOE_UP_RPS * MOE_UP_NSG) == 0;
}

/* Sparse MoE on the normed input x [1, T, H], before the routed experts are
 * combined: expert outputs, their fp32 routing weights [T, K] and the gated
 * shared expert [T, H]. */
static void moe_parts(const Config *c, const MoeW *w, mlx_array x, mlx_array *expert_out,
                      mlx_array *routing, mlx_array *shared_term) {
  int T = m_dim(x, 1), H = c->hidden, K = c->top_k, E = c->n_experts;
  mlx_array flat = m_reshape(x, SH(T, H));
  mlx_array router_logits = m_linear(flat, w->router); /* [T, E] */

  mlx_array inds, weights; /* [T, K]: expert ids, fp32 weights rounded through bf16 */
  if (T == 1 && !getenv("D3_NO_FUSED_ROUTER")) {
    route_step(c, router_logits, &inds, &weights);
  } else {
    mlx_array probs = m_softmax(m_astype(router_logits, MLX_FLOAT32), -1);
    inds = m_slice_axis(m_argpartition(probs, E - K, -1), -1, E - K, E);
    mlx_array scores = m_take_along_axis(probs, inds, -1);
    if (c->norm_topk) scores = m_div(scores, m_sum(scores, -1, true));
    weights = m_astype(m_astype(scores, MLX_BFLOAT16), MLX_FLOAT32);
  }

  int gs = c->expert_group, bits = c->expert_bits;
  int inter = c->moe_inter;
  if (T * K >= 64) {
    mlx_array flat_idx = m_reshape(inds, SH(T * K));
    mlx_array order = m_argsort(flat_idx, 0);
    mlx_array inv = m_argsort(order, 0);
    mlx_array rows = m_floor_divide(order, m_astype(m_int(K), m_dtype(order)));
    mlx_array xs = m_take(m_reshape(flat, SH(T, 1, H)), rows, 0); /* [T*K, 1, H] */
    mlx_array idx = m_take(flat_idx, order, 0);
    mlx_array gu = m_gather_qmm(xs, w->gu_w, w->gu_s, w->gu_b, idx, gs, bits, true);
    mlx_array gparts[2];
    m_split(gu, (int[]){inter}, 1, -1, gparts);
    mlx_array h = m_mul(m_silu_f32(gparts[0]), gparts[1]);
    mlx_array d = m_gather_qmm(h, w->down_w, w->down_s, w->down_b, idx, gs, bits, true);
    *expert_out = m_reshape(m_take(d, inv, 0), SH(T, K, H));
  } else {
    mlx_array h;
    if (T == 1 && moe_up_supported(c)) {
      h = moe_up_fused(c, w, flat, inds);
    } else {
      mlx_array xs = m_reshape(flat, SH(T, 1, 1, H));
      mlx_array gu = T == 1 ? m_gather_qmm_lhs(xs, w->gu_w, w->gu_s, w->gu_b, w->lhs_up, inds, gs,
                                               bits)
                            : m_gather_qmm(xs, w->gu_w, w->gu_s, w->gu_b, inds, gs, bits, false);
      mlx_array gparts[2];
      m_split(gu, (int[]){inter}, 1, -1, gparts);
      h = m_mul(m_silu_f32(gparts[0]), gparts[1]);
    }
    mlx_array d = T == 1 ? m_gather_qmm_lhs(h, w->down_w, w->down_s, w->down_b, w->lhs_down, inds,
                                            gs, bits)
                         : m_gather_qmm(h, w->down_w, w->down_s, w->down_b, inds, gs, bits, false);
    *expert_out = d; /* [T, K, 1, H] */
  }
  *routing = weights;
  mlx_array shared = m_linear(
      m_mul(m_silu_f32(m_linear(flat, w->sh_gate)), m_linear(flat, w->sh_up)), w->sh_down);
  *shared_term = m_mul(m_sigmoid(m_linear(flat, w->sh_expert_gate)), shared);
}

/* Routed experts plus the gated shared expert, [1, T, H]. */
static mlx_array moe_block(const Config *c, const MoeW *w, mlx_array x) {
  int T = m_dim(x, 1), H = c->hidden, K = c->top_k;
  mlx_array expert_out, weights, shared_term;
  moe_parts(c, w, x, &expert_out, &weights, &shared_term);
  mlx_array routed = m_astype(m_sum(m_mul(m_astype(m_reshape(expert_out, SH(T, K, H)),
                                                   MLX_FLOAT32),
                                          m_expand_dims(weights, -1)),
                                    1, false),
                              MLX_BFLOAT16);
  return m_reshape(m_add(routed, shared_term), SH(1, T, H));
}

static mlx_array moe_residual(const Config *c, const MoeW *w, mlx_array x) {
  return m_add(x, moe_block(c, w, zc_norm(x, w->post_norm, c->rms_eps)));
}

/* One row's layer tail: x2 = x + mix, x' = x2 + moe(post_norm(x2)), and
 * next_norm(x') for the next layer (or the final norm) in *h_out. */
static void layer_tail(const Config *c, const MoeW *w, mlx_array x, mlx_array mix,
                       mlx_array next_norm, mlx_array *x_out, mlx_array *h_out) {
  mlx_array x2, h2;
  if (getenv("D3_NO_FUSED_ADDNORM")) {
    x2 = m_add(x, mix);
    h2 = zc_norm(x2, w->post_norm, c->rms_eps);
  } else {
    m_add_rms_norm_f32(x, mix, w->post_norm, c->rms_eps, &x2, &h2);
  }
  if (getenv("D3_NO_FUSED_COMBINE")) {
    *x_out = m_add(x2, moe_block(c, w, h2));
    *h_out = zc_norm(*x_out, next_norm, c->rms_eps);
    return;
  }
  mlx_array expert_out, weights, shared_term;
  moe_parts(c, w, h2, &expert_out, &weights, &shared_term);
  moe_combine(c, expert_out, weights, shared_term, x2, next_norm, x_out, h_out);
}

/* ---- compiled single-row decode ----
 * A row carries the residual x and h = input_norm(x): each layer's fused tail
 * produces the next layer's normed input. */

/* in: x, h, conv, rec, next_norm, GdnW..., MoeW...  out: x', h', conv', rec' */
static void gdn_layer_fn(const mlx_array *in, mlx_array *out, const void *ctx) {
  const Config *c = ctx;
  const GdnW *gw = (const GdnW *)(in + 5);
  const MoeW *mw = (const MoeW *)(in + 5 + GDN_W_COUNT);
  mlx_array mix;
  gdn_mixer(c, gw, in[1], in[2], in[3], &mix, &out[2], &out[3]);
  layer_tail(c, mw, in[0], mix, in[4], &out[0], &out[1]);
}

/* in: h, offset, AttnW...  out: q, k, v, gate */
static void attn_pre_fn(const mlx_array *in, mlx_array *out, const void *ctx) {
  attn_project(ctx, (const AttnW *)(in + 2), in[0], in[1], &out[0], &out[1], &out[2],
               &out[3]);
}

/* in: x, attended, gate, next_norm, AttnW..., MoeW...  out: x', h' */
static void attn_post_fn(const mlx_array *in, mlx_array *out, const void *ctx) {
  const Config *c = ctx;
  const AttnW *aw = (const AttnW *)(in + 4);
  const MoeW *mw = (const MoeW *)(in + 4 + ATTN_W_COUNT);
  layer_tail(c, mw, in[0], attn_output(c, aw, in[1], in[2]), in[3], &out[0], &out[1]);
}

static AttnShape attn_shape(const Config *c) {
  return (AttnShape){.heads = c->n_heads,
                     .kv_heads = c->n_kv_heads,
                     .head_dim = c->head_dim,
                     .rotary_dim = c->rotary_dim,
                     .traditional = false,
                     .rope_base = c->rope_theta,
                     .eps = c->rms_eps,
                     .q_stride = 2 * c->head_dim,
                     .gate_base = c->head_dim};
}

/* The whole attention layer with the fused attention kernel.
 * in: x, h, pos, k_cache, v_cache, next_norm, variant tag, AttnW..., MoeW...
 * out: x', h', k_new, v_new */
static void attn_layer_fn(const mlx_array *in, mlx_array *out, const void *ctx) {
  const Config *c = ctx;
  const AttnW *aw = (const AttnW *)(in + 7);
  const MoeW *mw = (const MoeW *)(in + 7 + ATTN_W_COUNT);
  AttnShape shape = attn_shape(c);
  mlx_array gated;
  attn_decode_fused(&shape, ATTN_TAG_BLOCKS(in[6]), m_linear(in[1], aw->q_proj),
                    m_linear(in[1], aw->k_proj), m_linear(in[1], aw->v_proj), aw->q_norm,
                    aw->k_norm, in[3], in[4], in[2], &gated, &out[2], &out[3]);
  layer_tail(c, mw, in[0], m_linear(gated, aw->o_proj), in[5], &out[0], &out[1]);
}

static void ensure_compiled(Model *m) {
  if (m->backbone_compiled.gdn.fn) return;
  m->backbone_compiled.attn = (Compiled){.fn = attn_layer_fn, .ctx = &m->cfg, .n_out = 4};
  m->backbone_compiled.gdn = (Compiled){.fn = gdn_layer_fn, .ctx = &m->cfg, .n_out = 4};
  m->backbone_compiled.attn_pre = (Compiled){.fn = attn_pre_fn, .ctx = &m->cfg, .n_out = 4};
  m->backbone_compiled.attn_post = (Compiled){.fn = attn_post_fn, .ctx = &m->cfg, .n_out = 2};
}

/* Attention over the cache after writing rows pos..pos+T-1. */
static mlx_array attend_cache(Model *m, int li, mlx_array q, mlx_array k, mlx_array v, int pos) {
  const Config *c = &m->cfg;
  LayerCache *C = &m->cache[li];
  int T = m_dim(q, 2), hd = c->head_dim, end = pos + T;
  if (end > m_dim(C->k, 2)) die("backbone cache capacity exceeded");
  assign(&C->k, m_slice_update(C->k, k, (int[]){0, 0, pos, 0},
                               (int[]){1, c->n_kv_heads, end, hd}, 4));
  assign(&C->v, m_slice_update(C->v, v, (int[]){0, 0, pos, 0},
                               (int[]){1, c->n_kv_heads, end, hd}, 4));
  return m_sdpa(q, m_slice_axis(C->k, 2, 0, end), m_slice_axis(C->v, 2, 0, end),
                1.0f / sqrtf((float)hd), T > 1 ? "causal" : "", null_array());
}

/* One row through layer li: x [1, 1, H] and h = input_norm(x); returns x' and
 * next_norm(x') in *h_out. */
static mlx_array layer_step(Model *m, int li, mlx_array x, mlx_array h, mlx_array next_norm,
                            int pos, mlx_array *h_out) {
  BackboneLayer *L = &m->layers[li];
  LayerCache *C = &m->cache[li];
  MoeW mw = moe_weights(m, L);
  ensure_compiled(m);
  if (m->cfg.is_linear[li]) {
    GdnW gw = gdn_weights(L);
    mlx_array in[5 + GDN_W_COUNT + MOE_W_COUNT] = {x, h, C->conv, C->rec, next_norm};
    memcpy(in + 5, &gw, sizeof(gw));
    memcpy(in + 5 + GDN_W_COUNT, &mw, sizeof(mw));
    mlx_array out[4];
    compiled_call(&m->backbone_compiled.gdn, in, sizeof(in) / sizeof(in[0]), out);
    assign(&C->conv, out[2]);
    assign(&C->rec, out[3]);
    *h_out = out[1];
    return out[0];
  }
  AttnW aw = attn_weights(L);
  AttnShape shape = attn_shape(&m->cfg);
  int blocks = attn_route(&shape, pos + 1);
  if (blocks >= 0) {
    mlx_array in[7 + ATTN_W_COUNT + MOE_W_COUNT] = {
        x, h, m_int(pos), C->k, C->v, next_norm, attn_variant_tag(blocks)};
    memcpy(in + 7, &aw, sizeof(aw));
    memcpy(in + 7 + ATTN_W_COUNT, &mw, sizeof(mw));
    mlx_array out[4];
    compiled_call(&m->backbone_compiled.attn, in, sizeof(in) / sizeof(in[0]), out);
    cache_write_row(&C->k, out[2], pos);
    cache_write_row(&C->v, out[3], pos);
    *h_out = out[1];
    return out[0];
  }
  mlx_array in[2 + ATTN_W_COUNT] = {h, m_int(pos)};
  memcpy(in + 2, &aw, sizeof(aw));
  mlx_array qkvg[4];
  compiled_call(&m->backbone_compiled.attn_pre, in, sizeof(in) / sizeof(in[0]), qkvg);
  mlx_array attended = attend_cache(m, li, qkvg[0], qkvg[1], qkvg[2], pos);
  mlx_array in2[4 + ATTN_W_COUNT + MOE_W_COUNT] = {x, attended, qkvg[3], next_norm};
  memcpy(in2 + 4, &aw, sizeof(aw));
  memcpy(in2 + 4 + ATTN_W_COUNT, &mw, sizeof(mw));
  mlx_array out[2];
  compiled_call(&m->backbone_compiled.attn_post, in2, sizeof(in2) / sizeof(in2[0]), out);
  *h_out = out[1];
  return out[0];
}

mlx_array backbone_layer_forward(Model *m, int li, mlx_array x, int pos) {
  const Config *c = &m->cfg;
  BackboneLayer *L = &m->layers[li];
  LayerCache *C = &m->cache[li];
  mlx_array h = zc_norm(x, L->input_norm, c->rms_eps);
  if (m_dim(x, 1) == 1) {
    mlx_array unused;
    return layer_step(m, li, x, h, L->input_norm, pos, &unused);
  }
  MoeW mw = moe_weights(m, L);
  if (c->is_linear[li]) {
    GdnW gw = gdn_weights(L);
    mlx_array mix, conv, rec;
    gdn_mixer(c, &gw, h, C->conv, C->rec, &mix, &conv, &rec);
    assign(&C->conv, conv);
    assign(&C->rec, rec);
    return moe_residual(c, &mw, m_add(x, mix));
  }
  AttnW aw = attn_weights(L);
  mlx_array q, k, v, gate;
  attn_project(c, &aw, h, m_int(pos), &q, &k, &v, &gate);
  mlx_array attended = attend_cache(m, li, q, k, v, pos);
  return moe_residual(c, &mw, m_add(x, attn_output(c, &aw, attended, gate)));
}

/* ---- public ---- */

mlx_array backbone_mixer_forward(Model *m, int li, mlx_array x, int pos) {
  const Config *c = &m->cfg;
  BackboneLayer *L = &m->layers[li];
  LayerCache *C = &m->cache[li];
  if (c->is_linear[li]) {
    GdnW gw = gdn_weights(L);
    mlx_array mix, conv, rec;
    gdn_mixer(c, &gw, x, C->conv, C->rec, &mix, &conv, &rec);
    assign(&C->conv, conv);
    assign(&C->rec, rec);
    return mix;
  }
  AttnW aw = attn_weights(L);
  mlx_array q, k, v, gate;
  attn_project(c, &aw, x, m_int(pos), &q, &k, &v, &gate);
  return attn_output(c, &aw, attend_cache(m, li, q, k, v, pos), gate);
}

mlx_array backbone_moe_forward(Model *m, int layer, mlx_array x) {
  MoeW mw = moe_weights(m, &m->layers[layer]);
  return moe_block(&m->cfg, &mw, x);
}

/* ---- fused input embedding ----
 * One threadgroup per frame row: the text embedding, for semantic rows mixed with
 * the first codebooks' embeddings (fp32 sum -> bf16, bf16 add, fp32 / sqrt(1 +
 * count) -> bf16, as the op path rounds), and the first layer's input norm of the
 * result (m_rms_norm_f32's reduction). */

static const char *EMBED_SOURCE =
    "  uint row = threadgroup_position_in_grid.x;\n"
    "  uint tid = thread_position_in_threadgroup.x;\n"
    "  uint tg = threads_per_threadgroup.x;\n"
    "  auto fr = frames + (size_t)row * W;\n"
    "  int tok = fr[0];\n"
    "  bool semantic = tok >= SEM_START && tok <= SEM_END;\n"
    "  auto text = embed + (size_t)tok * D;\n"
    "  auto xr = x + (size_t)row * D;\n"
    "  auto hr = h + (size_t)row * D;\n"
    "  /* MLX's rms_single_row for the norm: 4 contiguous elements per thread */\n"
    "  float acc = 0.0f;\n"
    "  for (uint i = 4 * tid; i < 4 * tid + 4 && i < D; i++) {\n"
    "    T t = text[i];\n"
    "    T v = t;\n"
    "    if (semantic && COUNT > 0) {\n"
    "      float e = 0.0f;\n"
    "      for (int j = 1; j <= COUNT; j++) {\n"
    "        float c = float(codebooks[(size_t)(fr[1 + j] + j * CODE_VOCAB) * D + i]);\n"
    "        e = j == 1 ? c : e + c;\n"
    "      }\n"
    "      T s = static_cast<T>(float(t) + float(static_cast<T>(e)));\n"
    "      v = static_cast<T>(float(s) / MIX_DIV);\n"
    "    }\n"
    "    xr[i] = v;\n"
    "    float f = float(v);\n"
    "    acc += f * f;\n"
    "  }\n"
    "  threadgroup float partial[32];\n"
    "  acc = simd_sum(acc);\n"
    "  uint lane = tid % 32, sg = tid / 32;\n"
    "  if (lane == 0) partial[sg] = acc;\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (sg == 0) {\n"
    "    float v = lane < (tg + 31) / 32 ? partial[lane] : 0.0f;\n"
    "    v = simd_sum(v);\n"
    "    if (lane == 0) partial[0] = v;\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  float inv = metal::precise::rsqrt(partial[0] / float(D) + EPS);\n"
    "  for (uint i = 4 * tid; i < 4 * tid + 4 && i < D; i++) {\n"
    "    float n = float(xr[i]) * inv;\n"
    "    n = n * nw[i];\n"
    "    hr[i] = static_cast<T>(n);\n"
    "  }\n";

static mlx_fast_metal_kernel embed_kernel;
static bool embed_ready;

/* frames [T, W] int32 -> x [T, H] (backbone_embed_frames) and its first-layer
 * input norm h [T, H]. */
static void embed_rows(Model *m, mlx_array frames, mlx_array *x, mlx_array *h) {
  const Config *c = &m->cfg;
  int count = c->vq_backbone_codebooks;
  if (!embed_ready) {
    const char *in_names[] = {"frames", "embed", "codebooks", "nw"};
    const char *out_names[] = {"x", "h"};
    char header[96];
    snprintf(header, sizeof(header), "constant float EPS = %.9g;\nconstant float MIX_DIV = %.9g;\n",
             c->rms_eps, sqrtf((float)(1 + count)));
    mlx_vector_string ins = mlx_vector_string_new_data(in_names, 4);
    mlx_vector_string outs = mlx_vector_string_new_data(out_names, 2);
    embed_kernel = mlx_fast_metal_kernel_new("drama3_embed_rows", ins, outs, EMBED_SOURCE, header,
                                             true, false);
    mlx_vector_string_free(ins);
    mlx_vector_string_free(outs);
    embed_ready = true;
  }
  int T = m_dim(frames, 0), W = m_dim(frames, 1), H = c->hidden;
  mlx_dtype dtype = m_dtype(m->embed_tokens);
  mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){T, H}, 2, dtype);
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){T, H}, 2, dtype);
  int tg = 32 * (((H + 3) / 4 + 31) / 32); /* MLX's rms_single_row threadgroup */
  mlx_fast_metal_kernel_config_set_grid(cfg, tg * T, 1, 1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", H);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "W", W);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "COUNT", count);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "CODE_VOCAB", c->code_vocab);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SEM_START", c->semantic_start);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SEM_END", c->semantic_end);
  mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dtype);
  mlx_array inputs[4] = {m_astype(frames, MLX_INT32), m->embed_tokens, m->codebook_embeddings,
                         m->layers[0].input_norm};
  mlx_vector_array in_vec = mlx_vector_array_new_data(inputs, 4);
  mlx_vector_array out_vec = mlx_vector_array_new();
  if (mlx_fast_metal_kernel_apply(&out_vec, embed_kernel, in_vec, cfg, g_stream))
    die("embedding kernel failed");
  mlx_array a = mlx_array_new(), b = mlx_array_new();
  mlx_vector_array_get(&a, out_vec, 0);
  mlx_vector_array_get(&b, out_vec, 1);
  *x = track(a);
  *h = track(b);
  mlx_vector_array_free(in_vec);
  mlx_vector_array_free(out_vec);
  mlx_fast_metal_kernel_config_free(cfg);
}

static bool fused_embed(void) {
  static int off = -1;
  if (off < 0) off = getenv("D3_NO_FUSED_EMBED") != NULL;
  return !off;
}

mlx_array backbone_embed_frames(Model *m, mlx_array frames) {
  const Config *c = &m->cfg;
  if (fused_embed()) {
    mlx_array x, h;
    embed_rows(m, frames, &x, &h);
    return x;
  }
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

/* One row through every layer from x [1, 1, H] and its first-layer norm h. */
static mlx_array forward_row(Model *m, mlx_array x, mlx_array h, int pos) {
  const Config *c = &m->cfg;
  for (int i = 0; i < c->n_layers; i++)
    x = layer_step(m, i, x, h, i + 1 < c->n_layers ? m->layers[i + 1].input_norm : m->final_norm,
                   pos, &h);
  return h;
}

mlx_array backbone_forward(Model *m, mlx_array embeds, int pos) {
  const Config *c = &m->cfg;
  mlx_array x = embeds;
  if (m_ndim(x) == 2) x = m_expand_dims(x, 0);
  if (m_dim(x, 1) == 1) return forward_row(m, x, zc_norm(x, m->layers[0].input_norm, c->rms_eps), pos);
  for (int i = 0; i < c->n_layers; i++) x = backbone_layer_forward(m, i, x, pos);
  return zc_norm(x, m->final_norm, c->rms_eps);
}

mlx_array backbone_forward_frames(Model *m, mlx_array frames, int pos) {
  if (m_dim(frames, 0) == 1 && fused_embed()) {
    mlx_array x, h;
    embed_rows(m, frames, &x, &h);
    return forward_row(m, m_expand_dims(x, 0), m_expand_dims(h, 0), pos);
  }
  return backbone_forward(m, backbone_embed_frames(m, frames), pos);
}

mlx_array backbone_audio_logits(Model *m, mlx_array hidden) {
  return m_linear(hidden, m->lm_head_audio);
}
