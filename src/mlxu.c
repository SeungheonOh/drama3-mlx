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

mlx_array m_linear(mlx_array x, mlx_array w) { return m_matmul(x, m_swapaxes(w, -1, -2)); }

mlx_array m_gather_qmm(mlx_array x, mlx_array w, mlx_array scales, mlx_array biases,
                       mlx_array rhs_indices, int group_size, int bits, bool sorted) {
  mlx_optional_int gs = {group_size, true}, nb = {bits, true};
  mlx_array none = {0};
  NEW(mlx_gather_qmm(res, x, w, scales, biases, none, rhs_indices, true, gs, nb, "affine",
                     sorted, g_stream));
}

/* ---- fast ---- */

mlx_array m_rms_norm(mlx_array x, mlx_array weight, float eps) {
  NEW(mlx_fast_rms_norm(res, x, weight, eps, g_stream));
}

mlx_array m_layer_norm(mlx_array x, mlx_array weight, mlx_array bias, float eps) {
  NEW(mlx_fast_layer_norm(res, x, weight, bias, eps, g_stream));
}

mlx_array m_rope(mlx_array x, int dims, bool traditional, float base, int offset) {
  mlx_optional_float b = {base, true};
  mlx_array none = {0};
  NEW(mlx_fast_rope(res, x, dims, traditional, b, 1.0f, offset, none, g_stream));
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
