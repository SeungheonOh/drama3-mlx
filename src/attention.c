/* Fused single-row gated attention for decode.
 *
 * Takes the attention projection of one row to the gated attention output: the
 * zero-centered q / k norms (m_rms_norm_f32's reduction), RoPE (the formula of
 * MLX's rope kernel), attention over the cached rows plus the new one, and the
 * output gate out * sigmoid(gate) in bf16; it also returns the new row's k / v
 * for the cache update. Attention follows MLX's own kernels (sdpa_vector.h,
 * adapted from MLX, MIT license) and MLX's choice between them: the single-pass
 * sdpa_vector, or the two-pass kernel with the same block count. The roundings
 * match the op path, so the result is the same as norm -> rope -> cache update
 * -> sdpa -> gate run as separate MLX ops.
 *
 * Single pass: one kernel. Two passes: prologue (norms, RoPE, the new row),
 * pass 1 (per block partials), pass 2 (combine and gate). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"

/* q / k norms and RoPE of head h's query and its kv head's key into threadgroup
 * memory (qr, kr; v row in vv). Needs at least 3 * HD threads. */
static const char *PROLOGUE =
    "  constexpr int GQA = NQ / NKV;\n"
    "  uint kvh = h / GQA;\n"
    "  int p = pos;\n"
    "  threadgroup bfloat16_t qn[HD], kn[HD], qr[HD], kr[HD], vv[HD];\n"
    "  threadgroup float part[32], inv_s[2];\n"
    "  {\n"
    "    const device T* qrow = qg + Q_STRIDE * h;\n"
    "    const device T* krow = kin + kvh * HD;\n"
    "    const device T* vrow = vin + kvh * HD;\n"
    "    /* MLX's rms_single_row: a row of HD on HD / 4 threads, 4 contiguous\n"
    "     * elements each; q on the first NT threads, k on the next NT */\n"
    "    constexpr int NT = 32 * ((HD / 4 + 31) / 32);\n"
    "    if (tid < 2 * NT) {\n"
    "      const device T* row = tid < NT ? qrow : krow;\n"
    "      uint t = tid % NT;\n"
    "      float acc = 0.0f;\n"
    "      for (uint i = 4 * t; i < 4 * t + 4 && i < (uint)HD; i++) {\n"
    "        float v = float(row[i]);\n"
    "        acc += v * v;\n"
    "      }\n"
    "      acc = simd_sum(acc);\n"
    "      if (lane == 0) part[sg] = acc;\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (sg < 2) {\n"
    "      /* simdgroup 0 for q, 1 for k: MLX's sum over the row's partials */\n"
    "      constexpr int NS = NT / 32;\n"
    "      float v = lane < (uint)NS ? part[sg * NS + lane] : 0.0f;\n"
    "      v = simd_sum(v);\n"
    "      if (lane == 0) inv_s[sg] = metal::precise::rsqrt(v / float(HD) + EPS);\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (tid < 2 * HD) {\n"
    "      bool isk = tid >= (uint)HD;\n"
    "      uint i = tid % HD;\n"
    "      float n = float((isk ? krow : qrow)[i]) * inv_s[isk ? 1 : 0];\n"
    "      n = n * (isk ? k_norm : q_norm)[i];\n"
    "      if (isk) kn[i] = static_cast<bfloat16_t>(n); else qn[i] = static_cast<bfloat16_t>(n);\n"
    "    } else if (tid < 3 * HD) {\n"
    "      vv[tid - 2 * HD] = vrow[tid - 2 * HD];\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    /* RoPE over the first ROT dims at position p; the rest pass through */\n"
    "    if (tid < 2 * HD) {\n"
    "      bool isk = tid >= (uint)HD;\n"
    "      uint i = tid % HD;\n"
    "      threadgroup bfloat16_t* src = isk ? kn : qn;\n"
    "      bfloat16_t r = src[i];\n"
    "      if (i < (uint)ROT) {\n"
    "        uint pair, i1, i2;\n"
    "        bool first;\n"
    "        if (TRAD) { pair = i / 2; i1 = 2 * pair; i2 = i1 + 1; first = (i % 2) == 0; }\n"
    "        else { pair = i % (ROT / 2); i1 = pair; i2 = pair + ROT / 2; first = i < (uint)(ROT / 2); }\n"
    "        float d = static_cast<float>(pair) / static_cast<float>(ROT / 2);\n"
    "        float inv_freq = metal::precise::exp2(-d * LOG2_BASE);\n"
    "        float L = 1.0f * static_cast<float>(p);\n"
    "        float theta = L * inv_freq;\n"
    "        float costheta = metal::fast::cos(theta);\n"
    "        float sintheta = metal::fast::sin(theta);\n"
    "        float x1 = static_cast<float>(src[i1]);\n"
    "        float x2 = static_cast<float>(src[i2]);\n"
    "        float rx = first ? x1 * costheta - x2 * sintheta : x1 * sintheta + x2 * costheta;\n"
    "        r = static_cast<bfloat16_t>(rx);\n"
    "      }\n"
    "      if (isk) kr[i] = r; else qr[i] = r;\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (h % GQA == 0) {\n"
    "      if (tid < (uint)HD) k_new[kvh * HD + tid] = kr[tid];\n"
    "      else if (tid < 2 * HD) v_new[kvh * HD + tid - HD] = vv[tid - HD];\n"
    "    }\n"
    "  }\n";

/* out * sigmoid(gate) with MLX's bf16 sigmoid, for the QK outputs o[] of
 * simdgroup sg (elements sg * QK + i of head h). */
static const char *GATE =
    "  if (lane == 0) {\n"
    "    for (int i = 0; i < QK; i++) {\n"
    "      uint e = sg * QK + i;\n"
    "      T g = qg[GATE_BASE + Q_STRIDE * h + e];\n"
    "      auto y = 1 / (1 + metal::precise::exp(metal::abs(g)));\n"
    "      T sig = (g < 0) ? y : 1 - y;\n"
    "      out[h * HD + e] = static_cast<T>(o[i]) * sig;\n"
    "    }\n"
    "  }\n";

/* sdpa_vector over rows 0..p (row p is the new one, in kr / vv), then the gate.
 * One threadgroup of 1024 per query head. */
static const char *SINGLE_BODY =
    "  constexpr int BN = 32;\n"
    "  constexpr int BD = 32;\n"
    "  constexpr int QK = HD / BD;\n"
    "  int N = p + 1;\n"
    "  threadgroup float outputs[BN * BD], max_scores[BN], sum_exp_scores[BN];\n"
    "  float q[QK], k[QK], o[QK];\n"
    "  for (int j = 0; j < QK; j++) q[j] = static_cast<float>(SCALE) * qr[lane * QK + j];\n"
    "  for (int j = 0; j < QK; j++) o[j] = 0;\n"
    "  float max_score = Limits<float>::finite_min;\n"
    "  float sum_exp_score = 0;\n"
    "  const device T* keys = k_cache + (size_t)kvh * CAP * HD + lane * QK;\n"
    "  const device T* values = v_cache + (size_t)kvh * CAP * HD + lane * QK;\n"
    "  for (int i = sg; i < N; i += BN) {\n"
    "    bool fresh = i == p;\n"
    "    for (int j = 0; j < QK; j++)\n"
    "      k[j] = fresh ? static_cast<float>(kr[lane * QK + j]) : static_cast<float>(keys[(size_t)i * HD + j]);\n"
    "    float score = 0;\n"
    "    for (int j = 0; j < QK; j++) score += q[j] * k[j];\n"
    "    score = simd_sum(score);\n"
    "    float new_max = max(max_score, score);\n"
    "    float factor = fast::exp(max_score - new_max);\n"
    "    float exp_score = fast::exp(score - new_max);\n"
    "    max_score = new_max;\n"
    "    sum_exp_score = sum_exp_score * factor + exp_score;\n"
    "    for (int j = 0; j < QK; j++) {\n"
    "      float v = fresh ? static_cast<float>(vv[lane * QK + j]) : static_cast<float>(values[(size_t)i * HD + j]);\n"
    "      o[j] = o[j] * factor + exp_score * v;\n"
    "    }\n"
    "  }\n"
    "  if (lane == 0) {\n"
    "    max_scores[sg] = max_score;\n"
    "    sum_exp_scores[sg] = sum_exp_score;\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  max_score = max_scores[lane];\n"
    "  float new_max = simd_max(max_score);\n"
    "  float factor = fast::exp(max_score - new_max);\n"
    "  sum_exp_score = simd_sum(sum_exp_scores[lane] * factor);\n"
    "  for (int i = 0; i < QK; i++) {\n"
    "    outputs[lane * BD + sg] = o[i];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    o[i] = simd_sum(outputs[sg * BD + lane] * factor);\n"
    "    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  }\n";

static const char *HEAD_IDS =
    "  uint h = threadgroup_position_in_grid.x;\n"
    "  uint tid = thread_position_in_threadgroup.x;\n"
    "  uint lane = thread_index_in_simdgroup;\n"
    "  uint sg = simdgroup_index_in_threadgroup;\n";

/* Two passes, prologue: the roped query of every head and the new row. */
static const char *PRE_TAIL =
    "  if (tid < (uint)HD) q_roped[h * HD + tid] = qr[tid];\n";

/* Two passes, pass 1 (sdpa_vector_2pass_1): threadgroup (32, GQA) per kv head
 * and block; simdgroup g takes query head GQA * kvh + g over rows block, block +
 * BLOCKS, ... */
static const char *PASS1_SOURCE =
    "  constexpr int BD = 32;\n"
    "  constexpr int QK = HD / BD;\n"
    "  constexpr int GQA = NQ / NKV;\n"
    "  uint kvh = threadgroup_position_in_grid.x;\n"
    "  uint block = threadgroup_position_in_grid.z;\n"
    "  uint g = thread_position_in_threadgroup.y;\n"
    "  uint lane = thread_position_in_threadgroup.x;\n"
    "  int p = pos;\n"
    "  int N = p + 1;\n"
    "  uint h = GQA * kvh + g;\n"
    "  float q[QK], o[QK];\n"
    "  for (int i = 0; i < QK; i++) q[i] = static_cast<float>(SCALE) * q_roped[h * HD + lane * QK + i];\n"
    "  for (int i = 0; i < QK; i++) o[i] = 0;\n"
    "  float max_score = Limits<float>::finite_min;\n"
    "  float sum_exp_score = 0;\n"
    "  const device T* keys = k_cache + (size_t)kvh * CAP * HD + lane * QK;\n"
    "  const device T* values = v_cache + (size_t)kvh * CAP * HD + lane * QK;\n"
    "  const device T* fresh_k = k_new + kvh * HD + lane * QK;\n"
    "  const device T* fresh_v = v_new + kvh * HD + lane * QK;\n"
    "  for (int i = block; i < N; i += BLOCKS) {\n"
    "    const device T* kp = i == p ? fresh_k : keys + (size_t)i * HD;\n"
    "    const device T* vp = i == p ? fresh_v : values + (size_t)i * HD;\n"
    "    float score = 0;\n"
    "    for (int j = 0; j < QK; j++) score += q[j] * kp[j];\n"
    "    score = simd_sum(score);\n"
    "    float new_max = max(max_score, score);\n"
    "    float factor = fast::exp(max_score - new_max);\n"
    "    float exp_score = fast::exp(score - new_max);\n"
    "    max_score = new_max;\n"
    "    sum_exp_score = sum_exp_score * factor + exp_score;\n"
    "    for (int j = 0; j < QK; j++) o[j] = o[j] * factor + exp_score * vp[j];\n"
    "  }\n"
    "  if (lane == 0) {\n"
    "    sums[h * BLOCKS + block] = sum_exp_score;\n"
    "    maxs[h * BLOCKS + block] = max_score;\n"
    "  }\n"
    "  for (int i = 0; i < QK; i++)\n"
    "    partials[((size_t)h * BLOCKS + block) * HD + lane * QK + i] = static_cast<T>(o[i]);\n";

/* Two passes, pass 2 (sdpa_vector_2pass_2) and the gate: one threadgroup of
 * 1024 per query head. */
static const char *PASS2_BODY =
    "  constexpr int BN = 32;\n"
    "  constexpr int BD = 32;\n"
    "  constexpr int QK = HD / BD;\n"
    "  float o[QK] = {0};\n"
    "  threadgroup float outputs[BN * BD];\n"
    "  const device T* part = partials + (size_t)h * BLOCKS * HD + sg * HD + lane * QK;\n"
    "  const device float* sm = sums + h * BLOCKS;\n"
    "  const device float* mx = maxs + h * BLOCKS;\n"
    "  float sum_exp_score = 0.0;\n"
    "  float max_score = Limits<float>::finite_min;\n"
    "  for (int b = 0; b < BLOCKS / BN; ++b) max_score = max(max_score, mx[lane + BN * b]);\n"
    "  max_score = simd_max(max_score);\n"
    "  for (int b = 0; b < BLOCKS / BN; ++b) {\n"
    "    float factor = fast::exp(mx[lane + BN * b] - max_score);\n"
    "    sum_exp_score += factor * sm[lane + BN * b];\n"
    "  }\n"
    "  sum_exp_score = simd_sum(sum_exp_score);\n"
    "  for (int b = 0; b < BLOCKS / BN; ++b) {\n"
    "    float factor = fast::exp(mx[sg] - max_score);\n"
    "    for (int i = 0; i < QK; i++) o[i] += factor * static_cast<float>(part[i]);\n"
    "    mx += BN;\n"
    "    part += BN * HD;\n"
    "  }\n"
    "  for (int i = 0; i < QK; i++) {\n"
    "    outputs[lane * BD + sg] = o[i];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    o[i] = simd_sum(outputs[sg * BD + lane]);\n"
    "    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  }\n";

enum { SINGLE, PRE, PASS1, PASS2, N_VARIANTS };

typedef struct {
  int variant;
  float eps, log2_base;
  mlx_fast_metal_kernel kernel;
} AttnKernel;

static AttnKernel kernels[16];
static int n_kernels;

static mlx_fast_metal_kernel attn_kernel(int variant, float eps, float log2_base, float scale) {
  for (int i = 0; i < n_kernels; i++)
    if (kernels[i].variant == variant && kernels[i].eps == eps &&
        kernels[i].log2_base == log2_base)
      return kernels[i].kernel;
  if (n_kernels == 16) die("too many fused attention variants");
  static const char *names[N_VARIANTS] = {"single", "pre", "pass1", "pass2"};
  static const char *single_in[] = {"qg", "kin", "vin", "q_norm", "k_norm", "k_cache", "v_cache",
                                    "pos"};
  static const char *single_out[] = {"out", "k_new", "v_new"};
  static const char *pre_in[] = {"qg", "kin", "vin", "q_norm", "k_norm", "pos"};
  static const char *pre_out[] = {"q_roped", "k_new", "v_new"};
  static const char *pass1_in[] = {"q_roped", "k_new", "v_new", "k_cache", "v_cache", "pos"};
  static const char *pass1_out[] = {"partials", "sums", "maxs"};
  static const char *pass2_in[] = {"partials", "sums", "maxs", "qg"};
  static const char *pass2_out[] = {"out"};
  const char **in_names = single_in, **out_names = single_out;
  int n_in = 8, n_out = 3;
  char source[16384];
  switch (variant) {
  case SINGLE:
    snprintf(source, sizeof(source), "%s%s%s%s", HEAD_IDS, PROLOGUE, SINGLE_BODY, GATE);
    break;
  case PRE:
    in_names = pre_in;
    out_names = pre_out;
    n_in = 6;
    snprintf(source, sizeof(source), "%s%s%s", HEAD_IDS, PROLOGUE, PRE_TAIL);
    break;
  case PASS1:
    in_names = pass1_in;
    out_names = pass1_out;
    n_in = 6;
    snprintf(source, sizeof(source), "%s", PASS1_SOURCE);
    break;
  default:
    in_names = pass2_in;
    out_names = pass2_out;
    n_in = 4;
    n_out = 1;
    snprintf(source, sizeof(source), "%s%s%s", HEAD_IDS, PASS2_BODY, GATE);
    break;
  }
  char header[160], name[64];
  snprintf(header, sizeof(header),
           "constant float EPS = %.9g;\nconstant float LOG2_BASE = %.9g;\n"
           "constant float SCALE = %.9g;\n",
           eps, log2_base, scale);
  snprintf(name, sizeof(name), "drama3_attn_%s_%d", names[variant], n_kernels);
  mlx_vector_string ins = mlx_vector_string_new_data(in_names, n_in);
  mlx_vector_string outs = mlx_vector_string_new_data(out_names, n_out);
  AttnKernel *k = &kernels[n_kernels++];
  k->variant = variant;
  k->eps = eps;
  k->log2_base = log2_base;
  k->kernel = mlx_fast_metal_kernel_new(name, ins, outs, source, header, true, false);
  mlx_vector_string_free(ins);
  mlx_vector_string_free(outs);
  return k->kernel;
}

static void add_shape_args(mlx_fast_metal_kernel_config cfg, const AttnShape *s, int cap,
                           int blocks, mlx_dtype dtype) {
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HD", s->head_dim);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NQ", s->heads);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NKV", s->kv_heads);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROT", s->rotary_dim);
  mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "TRAD", s->traditional);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Q_STRIDE", s->q_stride);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GATE_BASE", s->gate_base);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "CAP", cap);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BLOCKS", blocks);
  mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dtype);
}

/* Applies a kernel; outputs are tracked. */
static void run_kernel(mlx_fast_metal_kernel kernel, mlx_fast_metal_kernel_config cfg,
                       const mlx_array *inputs, int n_in, mlx_array *outputs, int n_out) {
  mlx_vector_array in_vec = mlx_vector_array_new_data(inputs, n_in);
  mlx_vector_array out_vec = mlx_vector_array_new();
  if (mlx_fast_metal_kernel_apply(&out_vec, kernel, in_vec, cfg, g_stream))
    die("fused attention kernel failed");
  for (int i = 0; i < n_out; i++) {
    outputs[i] = mlx_array_new();
    mlx_vector_array_get(&outputs[i], out_vec, i);
    track(outputs[i]);
  }
  mlx_vector_array_free(in_vec);
  mlx_vector_array_free(out_vec);
  mlx_fast_metal_kernel_config_free(cfg);
}

/* The last letter of the GPU architecture name: MLX's device class. */
static char device_class(void) {
  static char cls;
  if (cls) return cls;
  cls = '?';
  mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
  mlx_device_info info = mlx_device_info_new();
  const char *arch = NULL;
  if (!mlx_device_info_get(&info, gpu) &&
      !mlx_device_info_get_string(&arch, info, "architecture") && arch && *arch)
    cls = arch[strlen(arch) - 1];
  mlx_device_info_free(info);
  mlx_device_free(gpu);
  return cls;
}

int attn_route(const AttnShape *s, int rows) {
  static int disabled = -1;
  if (disabled < 0) disabled = getenv("D3_NO_FUSED_ATTN") != NULL;
  if (disabled) return -1;
  /* MLX's routing for a single query (scaled_dot_product_attention.cpp, 0.32) */
  int gqa = s->heads / s->kv_heads, n_simds = gqa;
  char devc = device_class();
  bool two_pass = ((devc == 'd' || devc == 's') && rows >= 1024) ||
                  (s->kv_heads < s->heads && rows >= 4096);
  if (!two_pass) return 0;
  /* MLX's grouped-query variant of pass 1 is not reproduced */
  if (rows >= 8192 && ((gqa == 8 && (s->head_dim == 64 || s->head_dim == 128)) ||
                       ((gqa == 12 || gqa == 16) && s->head_dim == 128)))
    return -1;
  int blocks;
  if (devc == 's') {
    blocks = 64;
    if (rows > 1024 && n_simds > 4)
      blocks = rows <= 8192 ? 128 : rows <= 32768 ? 256 : rows <= 65536 ? 512 : 1024;
  } else if (devc == 'd') {
    blocks = 128;
    if (n_simds <= 2 && rows > 8192) blocks = 256;
    else if (n_simds >= 6 && rows >= 16384) blocks = rows < 65536 ? 512 : 1024;
  } else {
    blocks = n_simds >= 4 ? 64 : 32;
  }
  const char *env = getenv("MLX_SDPA_BLOCKS");
  if (env && atoi(env) > 0) blocks = (atoi(env) + 31) / 32 * 32;
  return blocks;
}

void attn_decode_fused(const AttnShape *s, int blocks, mlx_array qg, mlx_array k, mlx_array v,
                       mlx_array q_norm, mlx_array k_norm, mlx_array k_cache, mlx_array v_cache,
                       mlx_array pos, mlx_array *out, mlx_array *k_new, mlx_array *v_new) {
  int hd = s->head_dim, nq = s->heads, nkv = s->kv_heads, cap = m_dim(k_cache, 2);
  if (hd % 32 || nq % nkv || 3 * hd > 1024 || s->rotary_dim > hd || blocks % 32)
    die("fused attention: unsupported shape");
  float scale = 1.0f / sqrtf((float)hd), log2_base = log2f(s->rope_base);
  mlx_dtype dtype = m_dtype(qg);
  int out_shape[] = {1, 1, nq * hd}, row_shape[] = {1, nkv, 1, hd};
  if (blocks == 0) {
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_output_arg(cfg, out_shape, 3, dtype);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, row_shape, 4, dtype);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, row_shape, 4, dtype);
    mlx_fast_metal_kernel_config_set_grid(cfg, 1024 * nq, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1);
    add_shape_args(cfg, s, cap, 0, dtype);
    mlx_array inputs[] = {qg, k, v, q_norm, k_norm, k_cache, v_cache, pos}, outputs[3];
    run_kernel(attn_kernel(SINGLE, s->eps, log2_base, scale), cfg, inputs, 8, outputs, 3);
    *out = outputs[0];
    *k_new = outputs[1];
    *v_new = outputs[2];
    return;
  }
  /* prologue */
  mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){nq, hd}, 2, dtype);
  mlx_fast_metal_kernel_config_add_output_arg(cfg, row_shape, 4, dtype);
  mlx_fast_metal_kernel_config_add_output_arg(cfg, row_shape, 4, dtype);
  mlx_fast_metal_kernel_config_set_grid(cfg, 1024 * nq, 1, 1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1);
  add_shape_args(cfg, s, cap, blocks, dtype);
  mlx_array pre_in[] = {qg, k, v, q_norm, k_norm, pos}, pre[3];
  run_kernel(attn_kernel(PRE, s->eps, log2_base, scale), cfg, pre_in, 6, pre, 3);
  *k_new = pre[1];
  *v_new = pre[2];
  /* pass 1 */
  cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){nq, blocks, hd}, 3, dtype);
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){nq, blocks}, 2, MLX_FLOAT32);
  mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){nq, blocks}, 2, MLX_FLOAT32);
  mlx_fast_metal_kernel_config_set_grid(cfg, 32 * nkv, nq / nkv, blocks);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, nq / nkv, 1);
  add_shape_args(cfg, s, cap, blocks, dtype);
  mlx_array p1_in[] = {pre[0], pre[1], pre[2], k_cache, v_cache, pos}, p1[3];
  run_kernel(attn_kernel(PASS1, s->eps, log2_base, scale), cfg, p1_in, 6, p1, 3);
  /* pass 2 and the gate */
  cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, out_shape, 3, dtype);
  mlx_fast_metal_kernel_config_set_grid(cfg, 1024 * nq, 1, 1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1);
  add_shape_args(cfg, s, cap, blocks, dtype);
  mlx_array p2_in[] = {p1[0], p1[1], p1[2], qg};
  run_kernel(attn_kernel(PASS2, s->eps, log2_base, scale), cfg, p2_in, 4, out, 1);
}

mlx_array attn_variant_tag(int blocks) {
  /* one array per variant: its shape keys the compiled trace */
  static struct {
    int blocks;
    mlx_array tag;
  } tags[8];
  static int n_tags;
  for (int i = 0; i < n_tags; i++)
    if (tags[i].blocks == blocks) return tags[i].tag;
  if (n_tags == 8) die("too many attention variants");
  mlx_array tag = mlx_array_new();
  if (mlx_zeros(&tag, (int[]){blocks + 1}, 1, MLX_INT32, g_stream)) die("attention tag");
  m_eval(tag);
  tags[n_tags].blocks = blocks;
  tags[n_tags].tag = tag;
  return tags[n_tags++].tag;
}

void cache_write_row(mlx_array *cache, mlx_array row, int pos) {
  int kv = m_dim(*cache, 1), hd = m_dim(*cache, 3);
  if (pos >= m_dim(*cache, 2)) die("cache capacity exceeded");
  assign(cache, m_slice_update(*cache, row, (int[]){0, 0, pos, 0}, (int[]){1, kv, pos + 1, hd}, 4));
}
