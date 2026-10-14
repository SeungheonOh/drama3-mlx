/* Thin helpers over mlx-c.
 *
 * Every m_* op returns a temporary registered in the current arena. Code that
 * builds a graph marks the arena on entry and releases it on exit; arrays that
 * must outlive the scope are copied into owned handles with own() / assign().
 * MLX arrays are reference counted, so releasing a temporary handle never
 * invalidates a graph that still uses it.
 */
#ifndef DRAMA3_MLXU_H
#define DRAMA3_MLXU_H

#include <mlx/c/mlx.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

extern mlx_stream g_stream; /* default GPU stream */

void mlxu_init(void);
_Noreturn void die(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void log_info(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
double now_seconds(void);

/* ---- arena ---- */
size_t arena_mark(void);
void arena_release(size_t mark);
mlx_array track(mlx_array a);
/* A new owning handle to the same array, outside the arena. */
mlx_array own(mlx_array a);
/* *dst (owned) := src; frees the previous handle in *dst. */
void assign(mlx_array *dst, mlx_array src);
void release(mlx_array *a);
static inline bool is_null(mlx_array a) { return a.ctx == NULL; }
static inline mlx_array null_array(void) { mlx_array a = {0}; return a; }

#define SH(...) (const int[]){__VA_ARGS__}, (sizeof((int[]){__VA_ARGS__}) / sizeof(int))

/* ---- creation ---- */
mlx_array m_float(float v);
mlx_array m_scalar(float v, mlx_dtype dtype);
mlx_array m_int(int v);
mlx_array m_bool(bool v);
mlx_array m_zeros(const int *shape, size_t n, mlx_dtype dtype);
mlx_array m_full(const int *shape, size_t n, float v, mlx_dtype dtype);
mlx_array m_from_data(const void *data, const int *shape, size_t n, mlx_dtype dtype);
mlx_array m_from_int32(const int32_t *data, int count);
mlx_array m_arange(double start, double stop, double step, mlx_dtype dtype);

/* ---- elementwise ---- */
mlx_array m_add(mlx_array a, mlx_array b);
mlx_array m_sub(mlx_array a, mlx_array b);
mlx_array m_mul(mlx_array a, mlx_array b);
mlx_array m_div(mlx_array a, mlx_array b);
mlx_array m_floor_divide(mlx_array a, mlx_array b);
mlx_array m_remainder(mlx_array a, mlx_array b);
mlx_array m_maximum(mlx_array a, mlx_array b);
mlx_array m_minimum(mlx_array a, mlx_array b);
mlx_array m_power(mlx_array a, mlx_array b);
mlx_array m_logaddexp(mlx_array a, mlx_array b);
mlx_array m_equal(mlx_array a, mlx_array b);
mlx_array m_not_equal(mlx_array a, mlx_array b);
mlx_array m_less(mlx_array a, mlx_array b);
mlx_array m_less_equal(mlx_array a, mlx_array b);
mlx_array m_greater(mlx_array a, mlx_array b);
mlx_array m_greater_equal(mlx_array a, mlx_array b);
mlx_array m_logical_and(mlx_array a, mlx_array b);
mlx_array m_logical_or(mlx_array a, mlx_array b);
mlx_array m_logical_not(mlx_array a);
mlx_array m_where(mlx_array cond, mlx_array x, mlx_array y);
mlx_array m_neg(mlx_array a);
mlx_array m_exp(mlx_array a);
mlx_array m_log(mlx_array a);
mlx_array m_sqrt(mlx_array a);
mlx_array m_rsqrt(mlx_array a);
mlx_array m_sin(mlx_array a);
mlx_array m_cos(mlx_array a);
mlx_array m_tanh(mlx_array a);
mlx_array m_erf(mlx_array a);
mlx_array m_sigmoid(mlx_array a);
mlx_array m_square(mlx_array a);
mlx_array m_abs(mlx_array a);
mlx_array m_reciprocal(mlx_array a);
mlx_array m_isfinite(mlx_array a);
mlx_array m_clip(mlx_array a, mlx_array lo, mlx_array hi);
mlx_array m_astype(mlx_array a, mlx_dtype dtype);
/* x * sigmoid(x), computed in the input dtype. */
mlx_array m_silu(mlx_array a);
/* torch.nn.functional.silu on a reduced-precision input: fp32 math, rounded back. */
mlx_array m_silu_f32(mlx_array a);
mlx_array m_softplus(mlx_array a);
mlx_array m_gelu(mlx_array a); /* exact (erf) GELU */

/* ---- shape ---- */
mlx_array m_reshape(mlx_array a, const int *shape, size_t n);
mlx_array m_transpose(mlx_array a, const int *axes, size_t n);
mlx_array m_swapaxes(mlx_array a, int a1, int a2);
mlx_array m_expand_dims(mlx_array a, int axis);
mlx_array m_squeeze(mlx_array a, int axis);
mlx_array m_flatten(mlx_array a, int start, int end);
mlx_array m_broadcast_to(mlx_array a, const int *shape, size_t n);
mlx_array m_contiguous(mlx_array a);
/* a[start:stop] along one axis (negative indices count from the end). */
mlx_array m_slice_axis(mlx_array a, int axis, int start, int stop);
mlx_array m_slice(mlx_array a, const int *start, const int *stop, size_t n);
mlx_array m_slice_strided(mlx_array a, const int *start, const int *stop,
                          const int *strides, size_t n);
mlx_array m_slice_update(mlx_array src, mlx_array update, const int *start,
                         const int *stop, size_t n);
mlx_array m_concat(const mlx_array *arrays, size_t count, int axis);
mlx_array m_concat2(mlx_array a, mlx_array b, int axis);
mlx_array m_stack(const mlx_array *arrays, size_t count, int axis);
/* Splits a along axis at the given indices; writes count+1 parts into out. */
void m_split(mlx_array a, const int *indices, size_t count, int axis, mlx_array *out);
mlx_array m_take(mlx_array a, mlx_array indices, int axis);
mlx_array m_take_along_axis(mlx_array a, mlx_array indices, int axis);
mlx_array m_repeat(mlx_array a, int repeats, int axis);
mlx_array m_pad(mlx_array a, int axis, int low, int high, float value);

/* ---- reductions ---- */
mlx_array m_sum(mlx_array a, int axis, bool keepdims);
mlx_array m_mean(mlx_array a, int axis, bool keepdims);
mlx_array m_max(mlx_array a, int axis, bool keepdims);
mlx_array m_argmax(mlx_array a, int axis, bool keepdims);
mlx_array m_all(mlx_array a);
mlx_array m_any(mlx_array a);
mlx_array m_cumsum(mlx_array a, int axis);
mlx_array m_softmax(mlx_array a, int axis);
mlx_array m_sort(mlx_array a, int axis);
mlx_array m_argsort(mlx_array a, int axis);
mlx_array m_argpartition(mlx_array a, int kth, int axis);

/* ---- linear algebra ---- */
mlx_array m_matmul(mlx_array a, mlx_array b);
/* x @ w.T */
mlx_array m_linear(mlx_array x, mlx_array w);
mlx_array m_gather_qmm(mlx_array x, mlx_array w, mlx_array scales, mlx_array biases,
                       mlx_array rhs_indices, int group_size, int bits, bool sorted);

/* ---- fast ---- */
mlx_array m_rms_norm(mlx_array x, mlx_array weight, float eps);
mlx_array m_layer_norm(mlx_array x, mlx_array weight, mlx_array bias, float eps);
mlx_array m_rope(mlx_array x, int dims, bool traditional, float base, int offset);
mlx_array m_sdpa(mlx_array q, mlx_array k, mlx_array v, float scale,
                 const char *mask_mode, mlx_array mask);

/* ---- convolution (channels last: x [N, L, C], w [O, K, C/groups]) ---- */
mlx_array m_conv1d(mlx_array x, mlx_array w, int stride, int padding, int dilation,
                   int groups);
mlx_array m_conv_transpose1d(mlx_array x, mlx_array w, int stride, int padding);

/* ---- random ---- */
void m_seed(uint64_t seed);
mlx_array m_uniform(const int *shape, size_t n, mlx_dtype dtype);

/* ---- evaluation & host access ---- */
void m_eval(mlx_array a);
void m_eval_many(const mlx_array *arrays, size_t count);
void m_async_eval_many(const mlx_array *arrays, size_t count);
int m_ndim(mlx_array a);
int m_dim(mlx_array a, int axis);
size_t m_size(mlx_array a);
mlx_dtype m_dtype(mlx_array a);
/* Evaluates, converts to the given dtype and copies into a malloc'd buffer. */
int32_t *m_to_int32(mlx_array a, size_t *count);
float *m_to_float(mlx_array a, size_t *count);
int m_item_int(mlx_array a);
float m_item_float(mlx_array a);
void m_print_shape(const char *label, mlx_array a);

#endif
