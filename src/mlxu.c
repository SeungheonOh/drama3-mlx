#include "mlxu.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

mlx_stream g_stream;

static mlx_array *arena_items;
static size_t arena_count, arena_capacity;

static void error_handler(const char *msg, void *data) {
  (void)data;
  fprintf(stderr, "mlx error: %s\n", msg);
  abort();
}

void mlxu_init(void) {
  static bool initialized;
  if (initialized) return;
  initialized = true;
  mlx_set_error_handler(error_handler, NULL, NULL);
  g_stream = mlx_default_gpu_stream_new();
}

void die(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  fprintf(stderr, "error: ");
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
  va_end(args);
  exit(1);
}

void log_info(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
  va_end(args);
}

double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- arena ---- */

size_t arena_mark(void) { return arena_count; }

void arena_release(size_t mark) {
  while (arena_count > mark) mlx_array_free(arena_items[--arena_count]);
}

mlx_array track(mlx_array a) {
  if (arena_count == arena_capacity) {
    arena_capacity = arena_capacity ? arena_capacity * 2 : 4096;
    arena_items = realloc(arena_items, arena_capacity * sizeof(mlx_array));
    if (!arena_items) die("out of memory");
  }
  arena_items[arena_count++] = a;
  return a;
}

mlx_array own(mlx_array a) {
  mlx_array r = mlx_array_new();
  mlx_array_set(&r, a);
  return r;
}

void assign(mlx_array *dst, mlx_array src) {
  if (dst->ctx == NULL) *dst = mlx_array_new();
  mlx_array_set(dst, src);
}

void release(mlx_array *a) {
  if (a->ctx) mlx_array_free(*a);
  a->ctx = NULL;
}

#define NEW(expr)                    \
  do {                               \
    mlx_array r_ = mlx_array_new();  \
    mlx_array *res = &r_;            \
    if (expr) die("mlx op failed: %s", #expr); \
    return track(r_);                \
  } while (0)

/* ---- creation ---- */

mlx_array m_float(float v) { return track(mlx_array_new_float(v)); }

mlx_array m_scalar(float v, mlx_dtype dtype) {
  return m_astype(m_float(v), dtype);
}

mlx_array m_int(int v) { return track(mlx_array_new_int(v)); }

mlx_array m_bool(bool v) { return track(mlx_array_new_bool(v)); }

mlx_array m_zeros(const int *shape, size_t n, mlx_dtype dtype) {
  NEW(mlx_zeros(res, shape, n, dtype, g_stream));
}

mlx_array m_full(const int *shape, size_t n, float v, mlx_dtype dtype) {
  mlx_array value = m_float(v);
  NEW(mlx_full(res, shape, n, value, dtype, g_stream));
}

mlx_array m_from_data(const void *data, const int *shape, size_t n, mlx_dtype dtype) {
  return track(mlx_array_new_data(data, shape, (int)n, dtype));
}

mlx_array m_from_int32(const int32_t *data, int count) {
  return m_from_data(data, (int[]){count}, 1, MLX_INT32);
}

mlx_array m_arange(double start, double stop, double step, mlx_dtype dtype) {
  NEW(mlx_arange(res, start, stop, step, dtype, g_stream));
}

/* ---- elementwise ---- */

#define BINARY(name, fn) \
  mlx_array name(mlx_array a, mlx_array b) { NEW(fn(res, a, b, g_stream)); }
#define UNARY(name, fn) \
  mlx_array name(mlx_array a) { NEW(fn(res, a, g_stream)); }

BINARY(m_add, mlx_add)
BINARY(m_sub, mlx_subtract)
BINARY(m_mul, mlx_multiply)
BINARY(m_div, mlx_divide)
BINARY(m_floor_divide, mlx_floor_divide)
BINARY(m_remainder, mlx_remainder)
BINARY(m_maximum, mlx_maximum)
BINARY(m_minimum, mlx_minimum)
BINARY(m_power, mlx_power)
BINARY(m_logaddexp, mlx_logaddexp)
BINARY(m_equal, mlx_equal)
BINARY(m_not_equal, mlx_not_equal)
BINARY(m_less, mlx_less)
BINARY(m_less_equal, mlx_less_equal)
BINARY(m_greater, mlx_greater)
BINARY(m_greater_equal, mlx_greater_equal)
BINARY(m_logical_and, mlx_logical_and)
BINARY(m_logical_or, mlx_logical_or)
UNARY(m_logical_not, mlx_logical_not)
UNARY(m_neg, mlx_negative)
UNARY(m_exp, mlx_exp)
UNARY(m_log, mlx_log)
UNARY(m_sqrt, mlx_sqrt)
UNARY(m_rsqrt, mlx_rsqrt)
UNARY(m_sin, mlx_sin)
UNARY(m_cos, mlx_cos)
UNARY(m_tanh, mlx_tanh)
UNARY(m_erf, mlx_erf)
UNARY(m_sigmoid, mlx_sigmoid)
UNARY(m_square, mlx_square)
UNARY(m_abs, mlx_abs)
UNARY(m_reciprocal, mlx_reciprocal)
UNARY(m_isfinite, mlx_isfinite)

mlx_array m_where(mlx_array cond, mlx_array x, mlx_array y) {
  NEW(mlx_where(res, cond, x, y, g_stream));
}

mlx_array m_clip(mlx_array a, mlx_array lo, mlx_array hi) {
  NEW(mlx_clip(res, a, lo, hi, g_stream));
}

mlx_array m_astype(mlx_array a, mlx_dtype dtype) {
  NEW(mlx_astype(res, a, dtype, g_stream));
}

mlx_array m_silu(mlx_array a) { return m_mul(a, m_sigmoid(a)); }

mlx_array m_silu_f32(mlx_array a) {
  mlx_dtype dtype = mlx_array_dtype(a);
  mlx_array x = m_astype(a, MLX_FLOAT32);
  return m_astype(m_mul(x, m_sigmoid(x)), dtype);
}

mlx_array m_softplus(mlx_array a) { return m_logaddexp(a, m_scalar(0.0f, mlx_array_dtype(a))); }

mlx_array m_gelu(mlx_array a) {
  /* exact GELU computed in fp32, rounded back like torch's reduced-precision path */
  mlx_dtype dtype = mlx_array_dtype(a);
  mlx_array x = m_astype(a, MLX_FLOAT32);
  mlx_array y = m_mul(m_mul(m_float(0.5f), x),
                      m_add(m_float(1.0f), m_erf(m_mul(x, m_float(0.70710678118654752440f)))));
  return m_astype(y, dtype);
}

/* ---- shape ---- */

mlx_array m_reshape(mlx_array a, const int *shape, size_t n) {
  NEW(mlx_reshape(res, a, shape, n, g_stream));
}

mlx_array m_transpose(mlx_array a, const int *axes, size_t n) {
  NEW(mlx_transpose_axes(res, a, axes, n, g_stream));
}

mlx_array m_swapaxes(mlx_array a, int a1, int a2) {
  NEW(mlx_swapaxes(res, a, a1, a2, g_stream));
}

mlx_array m_expand_dims(mlx_array a, int axis) {
  NEW(mlx_expand_dims(res, a, axis, g_stream));
}

mlx_array m_squeeze(mlx_array a, int axis) {
  NEW(mlx_squeeze_axis(res, a, axis, g_stream));
}

mlx_array m_flatten(mlx_array a, int start, int end) {
  NEW(mlx_flatten(res, a, start, end, g_stream));
}

mlx_array m_broadcast_to(mlx_array a, const int *shape, size_t n) {
  NEW(mlx_broadcast_to(res, a, shape, n, g_stream));
}

mlx_array m_contiguous(mlx_array a) { NEW(mlx_contiguous(res, a, false, g_stream)); }

mlx_array m_slice_axis(mlx_array a, int axis, int start, int stop) {
  int ndim = (int)mlx_array_ndim(a);
  if (axis < 0) axis += ndim;
  int lo[16], hi[16];
  const int *shape = mlx_array_shape(a);
  for (int i = 0; i < ndim; i++) {
    lo[i] = 0;
    hi[i] = shape[i];
  }
  int len = shape[axis];
  if (start < 0) start += len;
  if (stop < 0) stop += len;
  if (stop > len) stop = len;
  lo[axis] = start;
  hi[axis] = stop;
  return m_slice(a, lo, hi, ndim);
}

mlx_array m_slice(mlx_array a, const int *start, const int *stop, size_t n) {
  int strides[16];
  for (size_t i = 0; i < n; i++) strides[i] = 1;
  NEW(mlx_slice(res, a, start, n, stop, n, strides, n, g_stream));
}

mlx_array m_slice_strided(mlx_array a, const int *start, const int *stop,
                          const int *strides, size_t n) {
  NEW(mlx_slice(res, a, start, n, stop, n, strides, n, g_stream));
}

mlx_array m_slice_update(mlx_array src, mlx_array update, const int *start,
                         const int *stop, size_t n) {
  int strides[16];
  for (size_t i = 0; i < n; i++) strides[i] = 1;
  NEW(mlx_slice_update(res, src, update, start, n, stop, n, strides, n, g_stream));
}

mlx_array m_concat(const mlx_array *arrays, size_t count, int axis) {
  mlx_vector_array vec = mlx_vector_array_new_data(arrays, count);
  mlx_array r = mlx_array_new();
  if (mlx_concatenate_axis(&r, vec, axis, g_stream)) die("concatenate failed");
  mlx_vector_array_free(vec);
  return track(r);
}

mlx_array m_concat2(mlx_array a, mlx_array b, int axis) {
  mlx_array parts[2] = {a, b};
  return m_concat(parts, 2, axis);
}

mlx_array m_stack(const mlx_array *arrays, size_t count, int axis) {
  mlx_vector_array vec = mlx_vector_array_new_data(arrays, count);
  mlx_array r = mlx_array_new();
  if (mlx_stack_axis(&r, vec, axis, g_stream)) die("stack failed");
  mlx_vector_array_free(vec);
  return track(r);
}

void m_split(mlx_array a, const int *indices, size_t count, int axis, mlx_array *out) {
  mlx_vector_array parts = mlx_vector_array_new();
  if (mlx_split_sections(&parts, a, indices, count, axis, g_stream)) die("split failed");
  for (size_t i = 0; i <= count; i++) {
    mlx_array part = mlx_array_new();
    mlx_vector_array_get(&part, parts, i);
    out[i] = track(part);
  }
  mlx_vector_array_free(parts);
}

mlx_array m_take(mlx_array a, mlx_array indices, int axis) {
  NEW(mlx_take_axis(res, a, indices, axis, g_stream));
}

mlx_array m_take_along_axis(mlx_array a, mlx_array indices, int axis) {
  NEW(mlx_take_along_axis(res, a, indices, axis, g_stream));
}

mlx_array m_repeat(mlx_array a, int repeats, int axis) {
  NEW(mlx_repeat_axis(res, a, repeats, axis, g_stream));
}

mlx_array m_pad(mlx_array a, int axis, int low, int high, float value) {
  int ndim = (int)mlx_array_ndim(a);
  if (axis < 0) axis += ndim;
  mlx_array pad_value = m_scalar(value, mlx_array_dtype(a));
  NEW(mlx_pad(res, a, (int[]){axis}, 1, (int[]){low}, 1, (int[]){high}, 1, pad_value,
              "constant", g_stream));
}

/* ---- reductions ---- */

mlx_array m_sum(mlx_array a, int axis, bool keepdims) {
  NEW(mlx_sum_axis(res, a, axis, keepdims, g_stream));
}

mlx_array m_mean(mlx_array a, int axis, bool keepdims) {
  NEW(mlx_mean_axis(res, a, axis, keepdims, g_stream));
}

mlx_array m_max(mlx_array a, int axis, bool keepdims) {
  NEW(mlx_max_axis(res, a, axis, keepdims, g_stream));
}

mlx_array m_argmax(mlx_array a, int axis, bool keepdims) {
  NEW(mlx_argmax_axis(res, a, axis, keepdims, g_stream));
}

mlx_array m_all(mlx_array a) { NEW(mlx_all(res, a, false, g_stream)); }

mlx_array m_any(mlx_array a) { NEW(mlx_any(res, a, false, g_stream)); }

mlx_array m_cumsum(mlx_array a, int axis) {
  mlx_optional_dtype none = {0};
  NEW(mlx_cumsum_axis(res, a, axis, false, true, none, g_stream));
}

mlx_array m_softmax(mlx_array a, int axis) {
  NEW(mlx_softmax_axis(res, a, axis, true, g_stream));
}

mlx_array m_sort(mlx_array a, int axis) { NEW(mlx_sort_axis(res, a, axis, g_stream)); }

mlx_array m_argsort(mlx_array a, int axis) {
  NEW(mlx_argsort_axis(res, a, axis, g_stream));
}

mlx_array m_argpartition(mlx_array a, int kth, int axis) {
  NEW(mlx_argpartition_axis(res, a, kth, axis, g_stream));
}

/* ---- linear algebra ---- */

mlx_array m_matmul(mlx_array a, mlx_array b) { NEW(mlx_matmul(res, a, b, g_stream)); }

/* ---- single-row matrix-vector product ----
 * x with one row of K values times w [N, K]^T. The per-row arithmetic is MLX's
 * gemv (each lane takes 4 consecutive values per 128-wide block, blocks in order,
 * then a shuffle-down tree), so results are bit-identical to m_matmul; but each
 * simdgroup takes a single row and loads up to 16 blocks of it before
 * accumulating, so the loads are in flight together instead of one dependent
 * step per block. */

static const char *GEMV_SOURCE =
    "  uint lane = thread_index_in_simdgroup;\n"
    "  uint sg = simdgroup_index_in_threadgroup;\n"
    "  int row = threadgroup_position_in_grid.x * NSG + sg;\n"
    "  if (row >= N) return;\n"
    "  constexpr int ITERS = K / 128;\n"
    "  constexpr int CHUNK = ITERS < 16 ? ITERS : 16;\n"
    "  const device T* wp = w + (size_t)row * K + lane * 4;\n"
    "  const device T* xp = x + lane * 4;\n"
    "  float result = 0;\n"
    "  for (int c0 = 0; c0 < ITERS; c0 += CHUNK) {\n"
    "    vec<T, 4> wv[CHUNK];\n"
    "    vec<T, 4> xv[CHUNK];\n"
    "    for (int i = 0; i < CHUNK; i++) {\n"
    "      wv[i] = *(const device vec<T, 4>*)(wp + (c0 + i) * 128);\n"
    "      xv[i] = *(const device vec<T, 4>*)(xp + (c0 + i) * 128);\n"
    "    }\n"
    "    for (int i = 0; i < CHUNK; i++) {\n"
    "      float v[4] = {static_cast<float>(xv[i][0]), static_cast<float>(xv[i][1]),\n"
    "                    static_cast<float>(xv[i][2]), static_cast<float>(xv[i][3])};\n"
    "      T in[4] = {wv[i][0], wv[i][1], wv[i][2], wv[i][3]};\n"
    "      for (int t = 0; t < 4; t++) result += in[t] * v[t];\n"
    "    }\n"
    "  }\n"
    "  for (ushort sn = 16; sn >= 1; sn >>= 1) result += simd_shuffle_down(result, sn);\n"
    "  if (lane == 0) y[row] = static_cast<T>(result);\n";

/* The gated pair: row r of the gate and of the up matrix in one simdgroup, written
 * as silu(gate) * up with m_silu_f32's and m_mul's roundings. */
static const char *GEMV_PAIR_SOURCE =
    "  uint lane = thread_index_in_simdgroup;\n"
    "  uint sg = simdgroup_index_in_threadgroup;\n"
    "  int row = threadgroup_position_in_grid.x * NSG + sg;\n"
    "  if (row >= PAIR) return;\n"
    "  constexpr int ITERS = K / 128;\n"
    "  constexpr int CHUNK = ITERS < 8 ? ITERS : 8;\n"
    "  const device T* gp = wg + (size_t)row * K + lane * 4;\n"
    "  const device T* up = wu + (size_t)row * K + lane * 4;\n"
    "  const device T* xp = x + lane * 4;\n"
    "  float rg = 0, ru = 0;\n"
    "  for (int c0 = 0; c0 < ITERS; c0 += CHUNK) {\n"
    "    vec<T, 4> gv[CHUNK], uv[CHUNK], xv[CHUNK];\n"
    "    for (int i = 0; i < CHUNK; i++) {\n"
    "      gv[i] = *(const device vec<T, 4>*)(gp + (c0 + i) * 128);\n"
    "      uv[i] = *(const device vec<T, 4>*)(up + (c0 + i) * 128);\n"
    "      xv[i] = *(const device vec<T, 4>*)(xp + (c0 + i) * 128);\n"
    "    }\n"
    "    for (int i = 0; i < CHUNK; i++) {\n"
    "      float v[4] = {static_cast<float>(xv[i][0]), static_cast<float>(xv[i][1]),\n"
    "                    static_cast<float>(xv[i][2]), static_cast<float>(xv[i][3])};\n"
    "      T g4[4] = {gv[i][0], gv[i][1], gv[i][2], gv[i][3]};\n"
    "      T u4[4] = {uv[i][0], uv[i][1], uv[i][2], uv[i][3]};\n"
    "      for (int t = 0; t < 4; t++) rg += g4[t] * v[t];\n"
    "      for (int t = 0; t < 4; t++) ru += u4[t] * v[t];\n"
    "    }\n"
    "  }\n"
    "  for (ushort sn = 16; sn >= 1; sn >>= 1) {\n"
    "    rg += simd_shuffle_down(rg, sn);\n"
    "    ru += simd_shuffle_down(ru, sn);\n"
    "  }\n"
    "  if (lane == 0) {\n"
    "    float g = static_cast<float>(static_cast<T>(rg));\n"
    "    auto s = 1 / (1 + metal::precise::exp(metal::abs(g)));\n"
    "    float sig = (g < 0) ? s : 1 - s;\n"
    "    T silu = static_cast<T>(g * sig);\n"
    "    y[row] = silu * static_cast<T>(ru);\n"
    "  }\n";

#define GEMV_NSG 4

static bool fast_gemv(mlx_array x, mlx_array w) {
  static int off = -1;
  if (off < 0) off = getenv("D3_NO_FAST_GEMV") != NULL;
  if (off || mlx_array_ndim(w) != 2 || mlx_array_dtype(w) != MLX_BFLOAT16 ||
      mlx_array_dtype(x) != MLX_BFLOAT16)
    return false;
  const int *ws = mlx_array_shape(w);
  int N = ws[0], K = ws[1];
  /* MLX's gemv uses these per-row semantics for K > 64, K < 16 N, out >= 4 */
  return mlx_array_size(x) == (size_t)K && K % 128 == 0 && K < 16 * N && N >= 4;
}

static mlx_array gemv_row(mlx_array x, mlx_array w) {
  static mlx_fast_metal_kernel kernel;
  static bool ready;
  if (!ready) {
    const char *in_names[] = {"x", "w"};
    const char *out_names[] = {"y"};
    mlx_vector_string ins = mlx_vector_string_new_data(in_names, 2);
    mlx_vector_string outs = mlx_vector_string_new_data(out_names, 1);
    kernel = mlx_fast_metal_kernel_new("drama3_gemv_row", ins, outs, GEMV_SOURCE, "", true, false);
    mlx_vector_string_free(ins);
    mlx_vector_string_free(outs);
    ready = true;
  }
  const int *ws = mlx_array_shape(w);
  int N = ws[0], K = ws[1];
  int ndim = (int)mlx_array_ndim(x);
  int shape[8];
  memcpy(shape, mlx_array_shape(x), ndim * sizeof(int));
  shape[ndim - 1] = N;
  mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, shape, ndim, MLX_BFLOAT16);
  mlx_fast_metal_kernel_config_set_grid(cfg, 32 * GEMV_NSG * ((N + GEMV_NSG - 1) / GEMV_NSG), 1, 1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * GEMV_NSG, 1, 1);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", N);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", K);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSG", GEMV_NSG);
  mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", MLX_BFLOAT16);
  mlx_array inputs[2] = {x, w};
  mlx_vector_array in_vec = mlx_vector_array_new_data(inputs, 2);
  mlx_vector_array out_vec = mlx_vector_array_new();
  if (mlx_fast_metal_kernel_apply(&out_vec, kernel, in_vec, cfg, g_stream)) die("gemv kernel failed");
  mlx_array r = mlx_array_new();
  mlx_vector_array_get(&r, out_vec, 0);
  mlx_vector_array_free(in_vec);
  mlx_vector_array_free(out_vec);
  mlx_fast_metal_kernel_config_free(cfg);
  return track(r);
}

mlx_array m_linear(mlx_array x, mlx_array w) {
  if (fast_gemv(x, w)) return gemv_row(x, w);
  return m_matmul(x, m_swapaxes(w, -1, -2));
}

mlx_array m_linear_silu_pair(mlx_array x, mlx_array wg, mlx_array wu) {
  const int *ws = mlx_array_shape(wg);
  int pair = ws[0], K = ws[1];
  if (!fast_gemv(x, wg) || !fast_gemv(x, wu) || mlx_array_shape(wu)[0] != pair ||
      getenv("D3_NO_FUSED_PAIR"))
    return m_mul(m_silu_f32(m_linear(x, wg)), m_linear(x, wu));
  static mlx_fast_metal_kernel kernel;
  static bool ready;
  if (!ready) {
    const char *in_names[] = {"x", "wg", "wu"};
    const char *out_names[] = {"y"};
    mlx_vector_string ins = mlx_vector_string_new_data(in_names, 3);
    mlx_vector_string outs = mlx_vector_string_new_data(out_names, 1);
    kernel = mlx_fast_metal_kernel_new("drama3_gemv_silu_pair", ins, outs, GEMV_PAIR_SOURCE, "",
                                       true, false);
    mlx_vector_string_free(ins);
    mlx_vector_string_free(outs);
    ready = true;
  }
  int ndim = (int)mlx_array_ndim(x);
  int shape[8];
  memcpy(shape, mlx_array_shape(x), ndim * sizeof(int));
  shape[ndim - 1] = pair;
  mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, shape, ndim, MLX_BFLOAT16);
  mlx_fast_metal_kernel_config_set_grid(cfg, 32 * GEMV_NSG * ((pair + GEMV_NSG - 1) / GEMV_NSG), 1,
                                        1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * GEMV_NSG, 1, 1);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PAIR", pair);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", K);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSG", GEMV_NSG);
  mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", MLX_BFLOAT16);
  mlx_array inputs[3] = {x, wg, wu};
  mlx_vector_array in_vec = mlx_vector_array_new_data(inputs, 3);
  mlx_vector_array out_vec = mlx_vector_array_new();
  if (mlx_fast_metal_kernel_apply(&out_vec, kernel, in_vec, cfg, g_stream))
    die("gemv pair kernel failed");
  mlx_array r = mlx_array_new();
  mlx_vector_array_get(&r, out_vec, 0);
  mlx_vector_array_free(in_vec);
  mlx_vector_array_free(out_vec);
  mlx_fast_metal_kernel_config_free(cfg);
  return track(r);
}

mlx_array m_gather_qmm(mlx_array x, mlx_array w, mlx_array scales, mlx_array biases,
                       mlx_array rhs_indices, int group_size, int bits, bool sorted) {
  mlx_optional_int gs = {group_size, true}, nb = {bits, true};
  mlx_array none = {0};
  NEW(mlx_gather_qmm(res, x, w, scales, biases, none, rhs_indices, true, gs, nb, "affine",
                     sorted, g_stream));
}

mlx_array m_gather_qmm_lhs(mlx_array x, mlx_array w, mlx_array scales, mlx_array biases,
                           mlx_array lhs_indices, mlx_array rhs_indices, int group_size,
                           int bits) {
  mlx_optional_int gs = {group_size, true}, nb = {bits, true};
  NEW(mlx_gather_qmm(res, x, w, scales, biases, lhs_indices, rhs_indices, true, gs, nb, "affine",
                     false, g_stream));
}

/* ---- fast ---- */

mlx_array m_rms_norm(mlx_array x, mlx_array weight, float eps) {
  NEW(mlx_fast_rms_norm(res, x, weight, eps, g_stream));
}

static const char *RMS_SOURCE =
    "  uint row = threadgroup_position_in_grid.x;\n"
    "  uint tid = thread_position_in_threadgroup.x;\n"
    "  uint tg = threads_per_threadgroup.x;\n"
    "  auto xr = x + (size_t)row * D;\n"
    "  auto outr = out + (size_t)row * D;\n"
    "  /* MLX's rms_single_row: each thread sums 4 contiguous squares */\n"
    "  float acc = 0.0f;\n"
    "  for (uint i = 4 * tid; i < 4 * tid + 4 && i < D; i++) { float v = float(xr[i]); acc += v * v; }\n"
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
    "  for (uint i = tid; i < D; i += tg) {\n"
    "    float n = float(xr[i]) * inv;\n"
    "    if (HAS_W) n = n * w[i];\n"
    "    outr[i] = static_cast<OutT>(n);\n"
    "  }\n";

mlx_array m_rms_norm_f32(mlx_array x, mlx_array weight_f32, float eps, mlx_dtype out_dtype) {
  /* eps is baked into the source: one kernel per distinct eps */
  static struct {
    float eps;
    mlx_fast_metal_kernel kernel;
  } kernels[4];
  static int n_kernels;
  int which = -1;
  for (int i = 0; i < n_kernels; i++)
    if (kernels[i].eps == eps) which = i;
  if (which < 0) {
    if (n_kernels == 4) die("too many rms norm eps values");
    char header[64], name[64];
    snprintf(header, sizeof(header), "constant float EPS = %.9g;\n", eps);
    snprintf(name, sizeof(name), "drama3_rms_norm_f32_%d", n_kernels);
    const char *in_names[] = {"x", "w"};
    const char *out_names[] = {"out"};
    mlx_vector_string ins = mlx_vector_string_new_data(in_names, 2);
    mlx_vector_string outs = mlx_vector_string_new_data(out_names, 1);
    kernels[n_kernels].eps = eps;
    kernels[n_kernels].kernel =
        mlx_fast_metal_kernel_new(name, ins, outs, RMS_SOURCE, header, true, false);
    mlx_vector_string_free(ins);
    mlx_vector_string_free(outs);
    which = n_kernels++;
  }
  int ndim = (int)mlx_array_ndim(x);
  const int *shape = mlx_array_shape(x);
  int D = shape[ndim - 1];
  int rows = (int)(mlx_array_size(x) / (size_t)D);
  if (D > 4096) die("rms norm kernel: rows of more than 4096");
  int tg = 32 * (((D + 3) / 4 + 31) / 32); /* MLX's single-row threadgroup */
  bool has_w = !is_null(weight_f32);
  mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, shape, ndim, out_dtype);
  mlx_fast_metal_kernel_config_set_grid(cfg, rows * tg, 1, 1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D);
  mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "HAS_W", has_w);
  mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OutT", out_dtype);
  mlx_array inputs[2] = {x, has_w ? weight_f32 : x}; /* w unused without HAS_W */
  mlx_vector_array in_vec = mlx_vector_array_new_data(inputs, 2);
  mlx_vector_array out_vec = mlx_vector_array_new();
  if (mlx_fast_metal_kernel_apply(&out_vec, kernels[which].kernel, in_vec, cfg, g_stream))
    die("rms norm kernel failed");
  mlx_array r = mlx_array_new();
  mlx_vector_array_get(&r, out_vec, 0);
  mlx_vector_array_free(in_vec);
  mlx_vector_array_free(out_vec);
  mlx_fast_metal_kernel_config_free(cfg);
  return track(r);
}

static const char *ADD_RMS_SOURCE =
    "  uint row = threadgroup_position_in_grid.x;\n"
    "  uint tid = thread_position_in_threadgroup.x;\n"
    "  uint tg = threads_per_threadgroup.x;\n"
    "  auto ar = a + (size_t)row * D;\n"
    "  auto br = b + (size_t)row * D;\n"
    "  auto sr = sum + (size_t)row * D;\n"
    "  auto outr = out + (size_t)row * D;\n"
    "  /* the sum, then MLX's rms_single_row: 4 contiguous squares per thread */\n"
    "  float acc = 0.0f;\n"
    "  for (uint i = 4 * tid; i < 4 * tid + 4 && i < D; i++) {\n"
    "    T s = static_cast<T>(float(ar[i]) + float(br[i]));\n"
    "    sr[i] = s;\n"
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
    "    float n = float(sr[i]) * inv;\n"
    "    outr[i] = static_cast<T>(n * w[i]);\n"
    "  }\n";

void m_add_rms_norm_f32(mlx_array a, mlx_array b, mlx_array weight_f32, float eps,
                        mlx_array *sum, mlx_array *normed) {
  static struct {
    float eps;
    mlx_fast_metal_kernel kernel;
  } kernels[4];
  static int n_kernels;
  int which = -1;
  for (int i = 0; i < n_kernels; i++)
    if (kernels[i].eps == eps) which = i;
  if (which < 0) {
    if (n_kernels == 4) die("too many add-rms eps values");
    char header[64], name[64];
    snprintf(header, sizeof(header), "constant float EPS = %.9g;\n", eps);
    snprintf(name, sizeof(name), "drama3_add_rms_norm_%d", n_kernels);
    const char *in_names[] = {"a", "b", "w"};
    const char *out_names[] = {"sum", "out"};
    mlx_vector_string ins = mlx_vector_string_new_data(in_names, 3);
    mlx_vector_string outs = mlx_vector_string_new_data(out_names, 2);
    kernels[n_kernels].eps = eps;
    kernels[n_kernels].kernel =
        mlx_fast_metal_kernel_new(name, ins, outs, ADD_RMS_SOURCE, header, true, false);
    mlx_vector_string_free(ins);
    mlx_vector_string_free(outs);
    which = n_kernels++;
  }
  int ndim = (int)mlx_array_ndim(a);
  const int *shape = mlx_array_shape(a);
  int D = shape[ndim - 1];
  int rows = (int)(mlx_array_size(a) / (size_t)D);
  if (D > 4096) die("rms norm kernel: rows of more than 4096");
  int tg = 32 * (((D + 3) / 4 + 31) / 32); /* MLX's single-row threadgroup */
  mlx_dtype dtype = mlx_array_dtype(a);
  mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg, shape, ndim, dtype);
  mlx_fast_metal_kernel_config_add_output_arg(cfg, shape, ndim, dtype);
  mlx_fast_metal_kernel_config_set_grid(cfg, rows * tg, 1, 1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1);
  mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D);
  mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dtype);
  mlx_array inputs[3] = {a, b, weight_f32};
  mlx_vector_array in_vec = mlx_vector_array_new_data(inputs, 3);
  mlx_vector_array out_vec = mlx_vector_array_new();
  if (mlx_fast_metal_kernel_apply(&out_vec, kernels[which].kernel, in_vec, cfg, g_stream))
    die("add rms norm kernel failed");
  mlx_array s0 = mlx_array_new(), s1 = mlx_array_new();
  mlx_vector_array_get(&s0, out_vec, 0);
  mlx_vector_array_get(&s1, out_vec, 1);
  *sum = track(s0);
  *normed = track(s1);
  mlx_vector_array_free(in_vec);
  mlx_vector_array_free(out_vec);
  mlx_fast_metal_kernel_config_free(cfg);
}

mlx_array m_layer_norm(mlx_array x, mlx_array weight, mlx_array bias, float eps) {
  NEW(mlx_fast_layer_norm(res, x, weight, bias, eps, g_stream));
}

mlx_array m_rope(mlx_array x, int dims, bool traditional, float base, int offset) {
  mlx_optional_float b = {base, true};
  mlx_array none = {0};
  NEW(mlx_fast_rope(res, x, dims, traditional, b, 1.0f, offset, none, g_stream));
}

mlx_array m_rope_dyn(mlx_array x, int dims, bool traditional, float base, mlx_array offset) {
  mlx_optional_float b = {base, true};
  mlx_array none = {0};
  NEW(mlx_fast_rope_dynamic(res, x, dims, traditional, b, 1.0f, offset, none, g_stream));
}

mlx_array m_sdpa(mlx_array q, mlx_array k, mlx_array v, float scale,
                 const char *mask_mode, mlx_array mask) {
  mlx_array none = {0};
  NEW(mlx_fast_scaled_dot_product_attention(res, q, k, v, scale, mask_mode, mask, none,
                                            false, g_stream));
}

/* ---- convolution ---- */

mlx_array m_conv1d(mlx_array x, mlx_array w, int stride, int padding, int dilation,
                   int groups) {
  NEW(mlx_conv1d(res, x, w, stride, padding, dilation, groups, g_stream));
}

mlx_array m_conv_transpose1d(mlx_array x, mlx_array w, int stride, int padding) {
  NEW(mlx_conv_transpose1d(res, x, w, stride, padding, 1, 0, 1, g_stream));
}

/* ---- random ---- */

void m_seed(uint64_t seed) { mlx_random_seed(seed); }

mlx_array m_uniform(const int *shape, size_t n, mlx_dtype dtype) {
  mlx_array lo = m_float(0.0f), hi = m_float(1.0f), none = {0};
  NEW(mlx_random_uniform(res, lo, hi, shape, n, dtype, none, g_stream));
}

/* ---- compilation ---- */

static int compiled_trampoline(mlx_vector_array *res, const mlx_vector_array in, void *payload) {
  Compiled *c = payload;
  size_t mark = arena_mark();
  size_t n_in = mlx_vector_array_size(in);
  mlx_array inputs[64], outputs[16];
  if (n_in > 64 || c->n_out > 16) die("compiled function arity too large");
  for (size_t i = 0; i < n_in; i++) {
    inputs[i] = mlx_array_new();
    mlx_vector_array_get(&inputs[i], in, i);
    track(inputs[i]);
  }
  c->fn(inputs, outputs, c->ctx);
  mlx_vector_array_set_data(res, outputs, c->n_out);
  arena_release(mark);
  return 0;
}

static bool compile_disabled(void) {
  static int state = -1;
  if (state < 0) {
    const char *v = getenv("D3_NO_COMPILE");
    state = v && *v && strcmp(v, "0") != 0;
  }
  return state;
}

void compiled_call(Compiled *c, const mlx_array *in, size_t n_in, mlx_array *out) {
  if (compile_disabled()) {
    c->fn(in, out, c->ctx);
    return;
  }
  if (!c->ready) {
    mlx_closure raw = mlx_closure_new_func_payload(compiled_trampoline, c, NULL);
    c->closure = mlx_closure_new();
    if (mlx_compile(&c->closure, raw, false)) die("mlx_compile failed");
    mlx_closure_free(raw);
    c->ready = true;
  }
  mlx_vector_array args = mlx_vector_array_new_data(in, n_in);
  mlx_vector_array results = mlx_vector_array_new();
  if (mlx_closure_apply(&results, c->closure, args)) die("compiled call failed");
  for (size_t i = 0; i < c->n_out; i++) {
    mlx_array a = mlx_array_new();
    mlx_vector_array_get(&a, results, i);
    out[i] = track(a);
  }
  mlx_vector_array_free(args);
  mlx_vector_array_free(results);
}

/* ---- evaluation & host access ---- */

void m_eval(mlx_array a) {
  if (mlx_array_eval(a)) die("eval failed");
}

void m_eval_many(const mlx_array *arrays, size_t count) {
  mlx_vector_array vec = mlx_vector_array_new_data(arrays, count);
  if (mlx_eval(vec)) die("eval failed");
  mlx_vector_array_free(vec);
}

void m_async_eval_many(const mlx_array *arrays, size_t count) {
  mlx_vector_array vec = mlx_vector_array_new_data(arrays, count);
  if (mlx_async_eval(vec)) die("async eval failed");
  mlx_vector_array_free(vec);
}

int m_ndim(mlx_array a) { return (int)mlx_array_ndim(a); }

int m_dim(mlx_array a, int axis) {
  if (axis < 0) axis += m_ndim(a);
  return mlx_array_dim(a, axis);
}

size_t m_size(mlx_array a) { return mlx_array_size(a); }

mlx_dtype m_dtype(mlx_array a) { return mlx_array_dtype(a); }

int32_t *m_to_int32(mlx_array a, size_t *count) {
  size_t mark = arena_mark();
  mlx_array c = m_contiguous(m_astype(a, MLX_INT32));
  m_eval(c);
  size_t n = mlx_array_size(c);
  int32_t *out = malloc((n ? n : 1) * sizeof(int32_t));
  memcpy(out, mlx_array_data_int32(c), n * sizeof(int32_t));
  arena_release(mark);
  if (count) *count = n;
  return out;
}

float *m_to_float(mlx_array a, size_t *count) {
  size_t mark = arena_mark();
  mlx_array c = m_contiguous(m_astype(a, MLX_FLOAT32));
  m_eval(c);
  size_t n = mlx_array_size(c);
  float *out = malloc((n ? n : 1) * sizeof(float));
  memcpy(out, mlx_array_data_float32(c), n * sizeof(float));
  arena_release(mark);
  if (count) *count = n;
  return out;
}

int m_item_int(mlx_array a) {
  int32_t *v = m_to_int32(a, NULL);
  int r = v[0];
  free(v);
  return r;
}

float m_item_float(mlx_array a) {
  float *v = m_to_float(a, NULL);
  float r = v[0];
  free(v);
  return r;
}

void m_print_shape(const char *label, mlx_array a) {
  fprintf(stderr, "%s: [", label);
  for (int i = 0; i < m_ndim(a); i++) fprintf(stderr, i ? ", %d" : "%d", m_dim(a, i));
  fprintf(stderr, "] dtype=%d\n", (int)m_dtype(a));
}
