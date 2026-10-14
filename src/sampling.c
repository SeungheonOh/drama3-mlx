/* Top-k / top-p sampling with the semantics of tower.py sample_top_k_top_p:
 * the token whose cumulative probability crosses top_p is dropped (except the
 * highest-ranked one), and the draw is argmax(p / Exp(1) noise).
 *
 * For top_k <= 64 one Metal kernel does the whole draw per row: temperature,
 * the full-row log-sum-exp, the top k (per-simdgroup selection, then a merge of
 * the sorted lists; ties go to the lower id, as the op graph's stable sort
 * does), the cumulative top-p cut, the masked softmax and the noisy argmax, with
 * MLX's exp / log variants where the op graph used them. Larger top_k uses the op
 * graph. Both consume the same uniform draws, so a seed gives the same ids. */
#include <math.h>
#include <stdlib.h>

#include "model.h"

#define MIN_TEMPERATURE 1e-5f
#define KERNEL_MAX_K 64

/* in: logits [N, V], uniform [N, k] in [0, 1), temperature, top_p  out: ids [N].
 * k (the top_k actually kept) comes from the noise shape, so each k gets its own
 * compiled trace. */
static void sample_fn(const mlx_array *in, mlx_array *out, const void *ctx) {
  mlx_array logits = in[0], u = in[1];
  int V = m_dim(logits, -1), N = m_dim(logits, 0), k = m_dim(u, -1);
  mlx_array x = m_div(m_astype(logits, MLX_FLOAT32), in[2]);
  mlx_array ids, values;
  if (k < V) {
    mlx_array part = m_slice_axis(m_argpartition(m_neg(x), k - 1, -1), -1, 0, k);
    mlx_array part_vals = m_take_along_axis(x, part, -1);
    mlx_array order = m_argsort(m_neg(part_vals), -1);
    ids = m_take_along_axis(part, order, -1);
    values = m_take_along_axis(part_vals, order, -1);
  } else {
    ids = m_argsort(m_neg(x), -1);
    values = m_take_along_axis(x, ids, -1);
  }
  mlx_array lse = m_max(x, -1, true);
  lse = m_add(lse, m_log(m_sum(m_exp(m_sub(x, lse)), -1, true)));
  mlx_array cumulative = m_cumsum(m_exp(m_sub(values, lse)), -1);
  mlx_array ranks = m_arange(0, k, 1, MLX_INT32);
  mlx_array dropped = m_greater(cumulative, in[3]);
  dropped = m_logical_and(dropped, m_greater(ranks, m_int(0)));
  mlx_array masked = m_where(dropped, m_float(-INFINITY), values);
  mlx_array probs = m_softmax(masked, -1);
  /* Exp(1) noise: -log(1 - u), with 1 - u in (0, 1] */
  mlx_array noise = m_neg(m_log(m_sub(m_float(1.0f), u)));
  mlx_array choice = m_argmax(m_div(probs, noise), -1, true);
  out[0] = m_reshape(m_take_along_axis(ids, choice, -1), SH(N));
}

static Compiled sampler = {.fn = sample_fn, .ctx = NULL, .n_out = 1};

static const char *SAMPLE_SOURCE =
    "  constexpr int TG = 1024;\n"
    "  constexpr int PER = (V + TG - 1) / TG;\n"
    "  uint row = threadgroup_position_in_grid.x;\n"
    "  uint tid = thread_position_in_threadgroup.x;\n"
    "  uint lane = thread_index_in_simdgroup;\n"
    "  uint sg = simdgroup_index_in_threadgroup;\n"
    "  threadgroup float red[32];\n"
    "  threadgroup float lv[32 * (K < 32 * PER ? K : 32 * PER)];\n"
    "  threadgroup uint li[32 * (K < 32 * PER ? K : 32 * PER)];\n"
    "  threadgroup float sv[K];\n"
    "  threadgroup uint si[K];\n"
    "  float temperature = params[0], top_p = params[1];\n"
    "  float vals[PER];\n"
    "  float lmax = -INFINITY;\n"
    "  for (int i = 0; i < PER; i++) {\n"
    "    uint idx = tid + i * TG;\n"
    "    float v = idx < (uint)V ? float(logits[row * V + idx]) / temperature : -INFINITY;\n"
    "    vals[i] = v;\n"
    "    lmax = metal::max(lmax, v);\n"
    "  }\n"
    "  float m = simd_max(lmax);\n"
    "  if (lane == 0) red[sg] = m;\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (sg == 0) { float t = simd_max(red[lane]); if (lane == 0) red[0] = t; }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  float gmax = red[0];\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  float ls = 0.0f;\n"
    "  for (int i = 0; i < PER; i++)\n"
    "    if (tid + i * TG < (uint)V) ls += metal::precise::exp(vals[i] - gmax);\n"
    "  ls = simd_sum(ls);\n"
    "  if (lane == 0) red[sg] = ls;\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (sg == 0) { float t = simd_sum(red[lane]); if (lane == 0) red[0] = t; }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  float lse = gmax + metal::precise::log(red[0]);\n"
    "  /* top K by value, ties to the lower id (a stable sort's order): each\n"
    "   * simdgroup ranks its own top KS, then simdgroup 0 merges the 32 sorted\n"
    "   * lists */\n"
    "  constexpr int KS = K < 32 * PER ? K : 32 * PER;\n"
    "  bool taken[PER];\n"
    "  for (int i = 0; i < PER; i++) taken[i] = false;\n"
    "  for (int r = 0; r < KS; r++) {\n"
    "    float bv = -INFINITY;\n"
    "    uint bi = 0xFFFFFFFFu;\n"
    "    int bj = -1;\n"
    "    for (int i = 0; i < PER; i++) {\n"
    "      uint idx = tid + i * TG;\n"
    "      if (idx < (uint)V && !taken[i] && (bj < 0 || vals[i] > bv)) { bv = vals[i]; bi = idx; bj = i; }\n"
    "    }\n"
    "    float best = simd_max(bv);\n"
    "    uint win = simd_min(bj >= 0 && bv == best ? bi : 0xFFFFFFFFu);\n"
    "    if (bj >= 0 && bi == win) taken[bj] = true;\n"
    "    if (lane == 0) { lv[sg * KS + r] = best; li[sg * KS + r] = win; }\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (sg == 0) {\n"
    "    uint head = 0;\n"
    "    for (int r = 0; r < K; r++) {\n"
    "      bool has = head < (uint)KS;\n"
    "      float hv = has ? lv[lane * KS + head] : -INFINITY;\n"
    "      uint hi = has ? li[lane * KS + head] : 0xFFFFFFFFu;\n"
    "      float best = simd_max(hv);\n"
    "      uint win = simd_min(hv == best ? hi : 0xFFFFFFFFu);\n"
    "      if (has && hi == win) head++;\n"
    "      if (lane == 0) { sv[r] = best; si[r] = win; }\n"
    "    }\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (tid == 0) {\n"
    "    float cum = 0.0f, top = sv[0], sum = 0.0f;\n"
    "    bool drop[K];\n"
    "    for (int r = 0; r < K; r++) {\n"
    "      cum += metal::precise::exp(sv[r] - lse);\n"
    "      drop[r] = r > 0 && cum > top_p;\n"
    "      if (!drop[r]) sum += fast::exp(sv[r] - top);\n"
    "    }\n"
    "    float inv = 1 / sum;\n"
    "    float best = -1.0f; uint choice = 0;\n"
    "    for (int r = 0; r < K; r++) {\n"
    "      float p = drop[r] ? 0.0f : fast::exp(sv[r] - top) * inv;\n"
    "      float noise = -metal::precise::log(1.0f - u[row * K + r]);\n"
    "      float score = p / noise;\n"
    "      if (score > best) { best = score; choice = (uint)r; }\n"
    "    }\n"
    "    ids[row] = (int)si[choice];\n"
    "  }\n";

static mlx_fast_metal_kernel sample_kernel;
static bool sample_kernel_ready;

static mlx_array sample_kernel_call(mlx_array logits, mlx_array u, float temperature,
                                    float top_p) {
  if (!sample_kernel_ready) {
    const char *in_names[] = {"logits", "u", "params"};
    const char *out_names[] = {"ids"};
    mlx_vector_string ins = mlx_vector_string_new_data(in_names, 3);
    mlx_vector_string outs = mlx_vector_string_new_data(out_names, 1);
    sample_kernel = mlx_fast_metal_kernel_new("drama3_sample", ins, outs, SAMPLE_SOURCE, "",
                                              true, false);
    mlx_vector_string_free(ins);
    mlx_vector_string_free(outs);
    sample_kernel_ready = true;
  }
  int N = m_dim(logits, 0), V = m_dim(logits, -1), k = m_dim(u, -1);
  float params[2] = {temperature, top_p};
  mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){N}, 1, MLX_INT32);
  mlx_fast_metal_kernel_config_set_grid(cfg, 1024 * N, 1, 1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "V", V);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", k);
  mlx_array inputs[3] = {logits, u, m_from_data(params, SH(2), MLX_FLOAT32)};
  mlx_vector_array in_vec = mlx_vector_array_new_data(inputs, 3);
  mlx_vector_array out_vec = mlx_vector_array_new();
  if (mlx_fast_metal_kernel_apply(&out_vec, sample_kernel, in_vec, cfg, g_stream))
    die("sampling kernel failed");
  mlx_array r = mlx_array_new();
  mlx_vector_array_get(&r, out_vec, 0);
  mlx_vector_array_free(in_vec);
  mlx_vector_array_free(out_vec);
  mlx_fast_metal_kernel_config_free(cfg);
  return track(r);
}

mlx_array sample_noise(int rows, int V, int top_k) {
  static mlx_stream cpu;
  static bool cpu_ready;
  if (!cpu_ready) {
    cpu = mlx_default_cpu_stream_new();
    cpu_ready = true;
  }
  int k = top_k < V ? top_k : V;
  mlx_stream saved = g_stream;
  g_stream = cpu;
  mlx_array u = m_uniform(SH(rows, k), MLX_FLOAT32);
  g_stream = saved;
  return u;
}

mlx_array sample_with_noise(mlx_array logits, mlx_array u, float temperature, float top_p) {
  if (temperature < MIN_TEMPERATURE) temperature = MIN_TEMPERATURE;
  int V = m_dim(logits, -1), k = m_dim(u, -1);
  if (k <= KERNEL_MAX_K && k < V && V <= 1024 * 64 && !getenv("D3_NO_FUSED_SAMPLER"))
    return sample_kernel_call(logits, u, temperature, top_p);
  mlx_array in[4] = {logits, u, m_float(temperature), m_float(top_p)};
  mlx_array out;
  compiled_call(&sampler, in, 4, &out);
  return out;
}

mlx_array sample_top_k_top_p(mlx_array logits, float temperature, int top_k, float top_p) {
  int V = m_dim(logits, -1), N = m_dim(logits, 0);
  int k = top_k < V ? top_k : V;
  return sample_with_noise(logits, m_uniform(SH(N, k), MLX_FLOAT32), temperature, top_p);
}
