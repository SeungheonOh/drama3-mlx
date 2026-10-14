/* S2-Pro codec, following fish_speech models/dac (modded_dac.py, rvq.py) and the
 * descript-audio-codec layers it builds on. Activations are channels-last
 * [1, L, C]; weights come from tools/convert.py (weight norm folded). */
#include "codec.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "weights.h"

#define SAMPLE_RATE 44100
#define ENCODER_DIM 64
#define LATENT_DIM 1024
#define DECODER_DIM 1536
#define N_RESIDUAL 9
#define SEMANTIC_CODEBOOK 4096
#define RESIDUAL_CODEBOOK 1024
#define T_HEAD_DIM 64
#define T_ROPE_BASE 10000.0f
#define T_NORM_EPS 1e-5f
#define ENCODER_WINDOW 512
#define QUANTIZER_WINDOW 128

static const int ENCODER_RATES[4] = {2, 4, 8, 8};
static const int DECODER_RATES[4] = {8, 8, 4, 2};

typedef struct {
  int dilation;
} UnitCtx;

struct Codec {
  /* every weight, converted once to dtype at load */
  mlx_map_string_to_array weights;
  mlx_dtype dtype;
  int hop, frame_length;
  /* RoPE tables shared by all codec transformers: [max_len, 32] in bf16 values */
  mlx_array rope_cos, rope_sin;
  int rope_len;
  /* compiled blocks: one fused elementwise kernel between convolutions */
  struct {
    Compiled snake, unit[3], convnext;
    UnitCtx unit_ctx[3];
  } compiled;
};

static mlx_array W(Codec *c, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

static mlx_array W(Codec *c, const char *fmt, ...) {
  char name[256];
  va_list args;
  va_start(args, fmt);
  vsnprintf(name, sizeof(name), fmt, args);
  va_end(args);
  mlx_array w = mlx_array_new();
  if (mlx_map_string_to_array_get(&w, c->weights, name) || !w.ctx) die("codec: missing %s", name);
  return track(w);
}

Codec *codec_load(const char *path, mlx_dtype dtype) {
  mlxu_init();
  Codec *c = calloc(1, sizeof(Codec));
  FILE *probe = fopen(path, "rb");
  if (!probe) die("cannot open %s (run tools/convert.py, or pass --model DIR)", path);
  fclose(probe);
  mlx_map_string_to_array raw = mlx_map_string_to_array_new();
  mlx_map_string_to_string meta = mlx_map_string_to_string_new();
  mlx_stream cpu = mlx_default_cpu_stream_new();
  if (mlx_load_safetensors(&raw, &meta, path, cpu)) die("cannot load %s", path);
  mlx_stream_free(cpu);
  mlx_map_string_to_string_free(meta);
  c->weights = mlx_map_string_to_array_new();
  mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(raw);
  const char *key;
  mlx_array value = mlx_array_new();
  mlx_array converted[1024];
  size_t n = 0;
  size_t mark = arena_mark();
  while (mlx_map_string_to_array_iterator_next(&key, &value, it) == 0 && key) {
    mlx_array a = m_astype(value, dtype);
    mlx_map_string_to_array_insert(c->weights, key, a);
    if (n < 1024) converted[n++] = a;
  }
  m_eval_many(converted, n);
  arena_release(mark);
  mlx_array_free(value);
  mlx_map_string_to_array_iterator_free(it);
  mlx_map_string_to_array_free(raw);
  c->dtype = dtype;
  c->hop = 1;
  for (int i = 0; i < 4; i++) c->hop *= ENCODER_RATES[i];
  c->frame_length = c->hop * 4;
  return c;
}

void codec_free(Codec *c) {
  if (!c) return;
  Compiled *compiled[] = {&c->compiled.snake, &c->compiled.convnext, &c->compiled.unit[0],
                          &c->compiled.unit[1], &c->compiled.unit[2]};
  for (size_t i = 0; i < sizeof(compiled) / sizeof(compiled[0]); i++)
    if (compiled[i]->ready) mlx_closure_free(compiled[i]->closure);
  release(&c->rope_cos);
  release(&c->rope_sin);
  mlx_map_string_to_array_free(c->weights);
  free(c);
}

int codec_sample_rate(const Codec *c) { return SAMPLE_RATE; }
int codec_frame_length(const Codec *c) { return c->frame_length; }

/* ---- layers ----
 * Each takes its weights explicitly; snake, residual units and ConvNeXt blocks run
 * as compiled closures, so the elementwise work between convolutions is fused
 * (with the same per-op roundings). */

static mlx_array snake_w(mlx_array x, mlx_array alpha) {
  mlx_array inv = m_reciprocal(m_add(alpha, m_scalar(1e-9f, m_dtype(alpha))));
  mlx_array s = m_sin(m_mul(alpha, x));
  return m_add(x, m_mul(inv, m_square(s)));
}

/* CausalConvNet: left pad (k_eff - stride), right pad to whole frames. */
static mlx_array causal_conv_w(mlx_array x, mlx_array w, mlx_array b, int stride, int dilation,
                               int groups) {
  int k = m_dim(w, 1);
  int k_eff = (k - 1) * dilation + 1;
  int padding = k_eff - stride;
  int length = m_dim(x, 1);
  double n_frames = (double)(length - k_eff + padding) / stride + 1.0;
  int ideal = (int)(ceil(n_frames) - 1) * stride + (k_eff - padding);
  int extra = ideal - length;
  if (padding > 0 || extra > 0) x = m_pad(m_pad(x, 1, padding, 0, 0.0f), 1, 0, extra, 0.0f);
  return m_add(m_conv1d(x, w, stride, 0, dilation, groups), b);
}

/* in: x, alpha  out: snake(x) */
static void snake_fn(const mlx_array *in, mlx_array *out, const void *ctx) {
  (void)ctx;
  out[0] = snake_w(in[0], in[1]);
}

/* in: x, alpha1, conv1 w / b, alpha2, conv2 w / b  out: the residual unit */
static void unit_fn(const mlx_array *in, mlx_array *out, const void *ctx) {
  const UnitCtx *u = ctx;
  mlx_array x = in[0];
  mlx_array y = causal_conv_w(snake_w(x, in[1]), in[2], in[3], 1, u->dilation, 1);
  y = causal_conv_w(snake_w(y, in[4]), in[5], in[6], 1, 1, 1);
  int pad = m_dim(x, 1) - m_dim(y, 1);
  if (pad > 0) x = m_slice_axis(x, 1, 0, m_dim(x, 1) - pad);
  out[0] = m_add(x, y);
}

/* in: x, dwconv w / b, norm w / b, pwconv1 w / b, pwconv2 w / b, gamma */
static void convnext_fn(const mlx_array *in, mlx_array *out, const void *ctx) {
  (void)ctx;
  mlx_array x = in[0];
  int dim = m_dim(x, 2);
  mlx_array y = causal_conv_w(x, in[1], in[2], 1, 1, dim);
  y = m_layer_norm(y, in[3], in[4], 1e-6f);
  y = m_gelu(m_add(m_linear(y, in[5]), in[6]));
  y = m_add(m_linear(y, in[7]), in[8]);
  out[0] = m_add(x, m_mul(in[9], y));
}

static void ensure_compiled(Codec *c) {
  if (c->compiled.snake.fn) return;
  static const int dil[3] = {1, 3, 9};
  c->compiled.snake = (Compiled){.fn = snake_fn, .n_out = 1};
  c->compiled.convnext = (Compiled){.fn = convnext_fn, .n_out = 1};
  for (int i = 0; i < 3; i++) {
    c->compiled.unit_ctx[i].dilation = dil[i];
    c->compiled.unit[i] = (Compiled){.fn = unit_fn, .ctx = &c->compiled.unit_ctx[i], .n_out = 1};
  }
}

static mlx_array snake(Codec *c, mlx_array x, const char *prefix) {
  ensure_compiled(c);
  mlx_array in[2] = {x, W(c, "%s.alpha", prefix)}, out;
  compiled_call(&c->compiled.snake, in, 2, &out);
  return out;
}

static mlx_array causal_conv(Codec *c, mlx_array x, const char *prefix, int stride,
                             int dilation, int groups) {
  return causal_conv_w(x, W(c, "%s.weight", prefix), W(c, "%s.bias", prefix), stride, dilation,
                       groups);
}

/* CausalTransConvNet: transposed conv, then drop (kernel - stride) trailing samples. */
static mlx_array causal_trans_conv(Codec *c, mlx_array x, const char *prefix, int stride) {
  mlx_array w = W(c, "%s.weight", prefix);
  mlx_array b = W(c, "%s.bias", prefix);
  int k = m_dim(w, 1);
  mlx_array y = m_add(m_conv_transpose1d(x, w, stride, 0), b);
  int pad = k - stride;
  if (pad > 0) y = m_slice_axis(y, 1, 0, m_dim(y, 1) - pad);
  return y;
}

static mlx_array residual_unit(Codec *c, mlx_array x, const char *prefix, int dilation) {
  ensure_compiled(c);
  int which = dilation == 1 ? 0 : dilation == 3 ? 1 : dilation == 9 ? 2 : -1;
  if (which < 0) die("codec: unexpected dilation %d", dilation);
  mlx_array in[7] = {x,
                     W(c, "%s.block.0.alpha", prefix),
                     W(c, "%s.block.1.conv.weight", prefix),
                     W(c, "%s.block.1.conv.bias", prefix),
                     W(c, "%s.block.2.alpha", prefix),
                     W(c, "%s.block.3.conv.weight", prefix),
                     W(c, "%s.block.3.conv.bias", prefix)};
  mlx_array out;
  compiled_call(&c->compiled.unit[which], in, 7, &out);
  return out;
}

static mlx_array linear_bias(Codec *c, mlx_array x, const char *prefix) {
  return m_add(m_linear(x, W(c, "%s.weight", prefix)), W(c, "%s.bias", prefix));
}

static mlx_array convnext(Codec *c, mlx_array x, const char *prefix) {
  ensure_compiled(c);
  mlx_array in[10] = {x,
                      W(c, "%s.dwconv.conv.weight", prefix),
                      W(c, "%s.dwconv.conv.bias", prefix),
                      W(c, "%s.norm.weight", prefix),
                      W(c, "%s.norm.bias", prefix),
                      W(c, "%s.pwconv1.weight", prefix),
                      W(c, "%s.pwconv1.bias", prefix),
                      W(c, "%s.pwconv2.weight", prefix),
                      W(c, "%s.pwconv2.bias", prefix),
                      W(c, "%s.gamma", prefix)};
  mlx_array out;
  compiled_call(&c->compiled.convnext, in, 10, &out);
  return out;
}

/* ---- window-limited causal transformer ---- */

static void ensure_rope(Codec *c, int length) {
  if (c->rope_len >= length) return;
  int half = T_HEAD_DIM / 2;
  int n = ((length + 1023) / 1024) * 1024;
  float *cosv = malloc((size_t)n * half * sizeof(float));
  float *sinv = malloc((size_t)n * half * sizeof(float));
  for (int i = 0; i < half; i++) {
    float freq = 1.0f / powf(T_ROPE_BASE, (float)(2 * i) / (float)T_HEAD_DIM);
    for (int t = 0; t < n; t++) {
      float angle = (float)t * freq;
      cosv[(size_t)t * half + i] = cosf(angle);
      sinv[(size_t)t * half + i] = sinf(angle);
    }
  }
  size_t mark = arena_mark();
  /* precompute_freqs_cis stores the table in bfloat16 */
  mlx_array cs = m_astype(m_astype(m_from_data(cosv, SH(n, half), MLX_FLOAT32), MLX_BFLOAT16),
                          MLX_FLOAT32);
  mlx_array sn = m_astype(m_astype(m_from_data(sinv, SH(n, half), MLX_FLOAT32), MLX_BFLOAT16),
                          MLX_FLOAT32);
  m_eval(cs);
  m_eval(sn);
  assign(&c->rope_cos, cs);
  assign(&c->rope_sin, sn);
  arena_release(mark);
  free(cosv);
  free(sinv);
  c->rope_len = n;
}

/* Interleaved RoPE on x [1, T, H, 64] with the bf16 table, computed in fp32. */
static mlx_array codec_rope(Codec *c, mlx_array x) {
  int T = m_dim(x, 1), H = m_dim(x, 2), half = T_HEAD_DIM / 2;
  mlx_array cs = m_reshape(m_slice_axis(c->rope_cos, 0, 0, T), SH(1, T, 1, half));
  mlx_array sn = m_reshape(m_slice_axis(c->rope_sin, 0, 0, T), SH(1, T, 1, half));
  mlx_array xf = m_reshape(m_astype(x, MLX_FLOAT32), SH(1, T, H, half, 2));
  mlx_array x0 = m_squeeze(m_slice_axis(xf, 4, 0, 1), 4);
  mlx_array x1 = m_squeeze(m_slice_axis(xf, 4, 1, 2), 4);
  mlx_array o0 = m_sub(m_mul(x0, cs), m_mul(x1, sn));
  mlx_array o1 = m_add(m_mul(x1, cs), m_mul(x0, sn));
  mlx_array out = m_reshape(m_stack((mlx_array[]){o0, o1}, 2, -1), SH(1, T, H, T_HEAD_DIM));
  return m_astype(out, m_dtype(x));
}

/* RMSNorm: fp32 norm cast back to x's dtype, then times the weight. */
static mlx_array codec_rms(Codec *c, mlx_array x, mlx_array w) {
  mlx_array n = m_astype(m_rms_norm(m_astype(x, MLX_FLOAT32), null_array(), T_NORM_EPS),
                         m_dtype(x));
  return m_mul(n, w);
}

static mlx_array window_mask(int T, int window) {
  mlx_array rows = m_reshape(m_arange(0, T, 1, MLX_INT32), SH(T, 1));
  mlx_array cols = m_reshape(m_arange(0, T, 1, MLX_INT32), SH(1, T));
  mlx_array causal = m_less_equal(cols, rows);
  mlx_array lower = m_maximum(m_sub(rows, m_int(window - 1)), m_int(0));
  return m_logical_and(causal, m_greater_equal(cols, lower));
}

/* WindowLimitedTransformer over x [1, T, dim] (input/output projections are
 * identities for the codec's configurations). */
static mlx_array transformer(Codec *c, mlx_array x, const char *prefix, int n_layers,
                             int window) {
  int T = m_dim(x, 1), dim = m_dim(x, 2);
  int heads = dim / T_HEAD_DIM;
  ensure_rope(c, T);
  mlx_array mask = window_mask(T, window);
  for (int i = 0; i < n_layers; i++) {
    char p[256];
    snprintf(p, sizeof(p), "%s.layers.%d", prefix, i);
    mlx_array h = codec_rms(c, x, W(c, "%s.attention_norm.weight", p));
    mlx_array parts[3];
    m_split(m_linear(h, W(c, "%s.attention.wqkv.weight", p)), (int[]){dim, 2 * dim}, 2, -1,
            parts);
    mlx_array q = codec_rope(c, m_reshape(parts[0], SH(1, T, heads, T_HEAD_DIM)));
    mlx_array k = codec_rope(c, m_reshape(parts[1], SH(1, T, heads, T_HEAD_DIM)));
    mlx_array v = m_reshape(parts[2], SH(1, T, heads, T_HEAD_DIM));
    q = m_transpose(q, SH(0, 2, 1, 3));
    k = m_transpose(k, SH(0, 2, 1, 3));
    v = m_transpose(v, SH(0, 2, 1, 3));
    mlx_array a = m_sdpa(q, k, v, 1.0f / sqrtf((float)T_HEAD_DIM), "array", mask);
    a = m_reshape(m_transpose(a, SH(0, 2, 1, 3)), SH(1, T, dim));
    a = m_linear(a, W(c, "%s.attention.wo.weight", p));
    x = m_add(x, m_mul(a, W(c, "%s.attention_layer_scale.gamma", p)));
    h = codec_rms(c, x, W(c, "%s.ffn_norm.weight", p));
    mlx_array ff = m_linear(m_mul(m_silu_f32(m_linear(h, W(c, "%s.feed_forward.w1.weight", p))),
                                  m_linear(h, W(c, "%s.feed_forward.w3.weight", p))),
                            W(c, "%s.feed_forward.w2.weight", p));
    x = m_add(x, m_mul(ff, W(c, "%s.ffn_layer_scale.gamma", p)));
  }
  return codec_rms(c, x, W(c, "%s.norm.weight", prefix));
}

/* ---- decoder ---- */

static mlx_array quantizer_decode(Codec *c, const Codes *codes) {
  int T = codes->frames;
  /* from_codes of the semantic quantizer, plus from_codes of the residual one:
   * z_q = semantic + (r0 + r1 + ... + r8) */
  mlx_array semantic = null_array(), residual = null_array();
  for (int q = 0; q <= N_RESIDUAL; q++) {
    int limit = q == 0 ? SEMANTIC_CODEBOOK - 1 : RESIDUAL_CODEBOOK - 1;
    mlx_array ids = m_from_int32(codes->data + (size_t)q * T, T);
    ids = m_clip(ids, m_int(0), m_int(limit));
    char p[128];
    if (q == 0)
      snprintf(p, sizeof(p), "quantizer.semantic_quantizer.quantizers.0");
    else
      snprintf(p, sizeof(p), "quantizer.quantizer.quantizers.%d", q - 1);
    mlx_array e = m_take(W(c, "%s.codebook.weight", p), ids, 0); /* [T, 8] */
    char op[160];
    snprintf(op, sizeof(op), "%s.out_proj", p);
    mlx_array zq = linear_bias(c, e, op); /* [T, 1024] */
    if (q == 0)
      semantic = zq;
    else
      residual = is_null(residual) ? zq : m_add(residual, zq);
  }
  return m_expand_dims(m_add(semantic, residual), 0);
}

float *codec_decode(Codec *c, const Codes *codes, size_t *samples) {
  if (codes->n_codebooks != N_RESIDUAL + 1) die("codec: expected %d codebooks", N_RESIDUAL + 1);
  size_t mark = arena_mark();
  mlx_array z = quantizer_decode(c, codes);
  z = transformer(c, z, "quantizer.post_module", 8, QUANTIZER_WINDOW);
  for (int i = 0; i < 2; i++) {
    char p[128];
    snprintf(p, sizeof(p), "quantizer.upsample.%d.0.conv", i);
    z = causal_trans_conv(c, z, p, 2);
    snprintf(p, sizeof(p), "quantizer.upsample.%d.1", i);
    z = convnext(c, z, p);
  }
  mlx_array x = causal_conv(c, z, "decoder.model.0.conv", 1, 1, 1);
  for (int b = 0; b < 4; b++) {
    char p[128];
    snprintf(p, sizeof(p), "decoder.model.%d.block.0", b + 1);
    x = snake(c, x, p);
    snprintf(p, sizeof(p), "decoder.model.%d.block.1.conv", b + 1);
    x = causal_trans_conv(c, x, p, DECODER_RATES[b]);
    static const int dil[3] = {1, 3, 9};
    for (int r = 0; r < 3; r++) {
      snprintf(p, sizeof(p), "decoder.model.%d.block.%d", b + 1, r + 2);
      x = residual_unit(c, x, p, dil[r]);
    }
  }
  x = snake(c, x, "decoder.model.5");
  x = causal_conv(c, x, "decoder.model.6.conv", 1, 1, 1);
  x = m_tanh(x);
  float *out = m_to_float(x, samples);
  arena_release(mark);
  return out;
}

/* ---- encoder ---- */

/* VectorQuantize: project, pick the nearest L2-normalized code, return
 * (indices [T], out-projected z_q [1, T, 1024]). */
static void vector_quantize(Codec *c, mlx_array z, const char *prefix, mlx_array *ids,
                            mlx_array *zq) {
  char p[192];
  snprintf(p, sizeof(p), "%s.in_proj", prefix);
  mlx_array ze = m_squeeze(linear_bias(c, z, p), 0); /* [T, 8] */
  mlx_array codebook = W(c, "%s.codebook.weight", prefix);
  /* F.normalize: x / max(||x||, 1e-12), in the codec dtype like the reference */
  mlx_array eps = m_scalar(1e-12f, c->dtype);
  mlx_array enc = m_div(ze, m_maximum(m_sqrt(m_sum(m_square(ze), -1, true)), eps));
  mlx_array cb =
      m_div(codebook, m_maximum(m_sqrt(m_sum(m_square(codebook), -1, true)), eps));
  mlx_array dist = m_add(m_sub(m_sum(m_square(enc), -1, true),
                               m_mul(m_scalar(2.0f, c->dtype),
                                     m_matmul(enc, m_swapaxes(cb, 0, 1)))),
                         m_reshape(m_sum(m_square(cb), -1, false), SH(1, -1)));
  *ids = m_astype(m_argmax(m_neg(dist), -1, false), MLX_INT32);
  mlx_array q = m_take(codebook, *ids, 0);
  snprintf(p, sizeof(p), "%s.out_proj", prefix);
  *zq = m_expand_dims(linear_bias(c, q, p), 0);
}

Codes codec_encode(Codec *c, const float *audio, size_t samples) {
  size_t mark = arena_mark();
  size_t padded = ((samples + c->frame_length - 1) / c->frame_length) * c->frame_length;
  if (padded == 0) die("codec: empty audio");
  float *buf = calloc(padded, sizeof(float));
  memcpy(buf, audio, samples * sizeof(float));
  mlx_array x = m_astype(m_from_data(buf, SH(1, (int)padded, 1), MLX_FLOAT32), c->dtype);
  free(buf);

  x = causal_conv(c, x, "encoder.block.0.conv", 1, 1, 1);
  for (int b = 0; b < 4; b++) {
    char p[128];
    static const int dil[3] = {1, 3, 9};
    for (int r = 0; r < 3; r++) {
      snprintf(p, sizeof(p), "encoder.block.%d.block.%d", b + 1, r);
      x = residual_unit(c, x, p, dil[r]);
    }
    snprintf(p, sizeof(p), "encoder.block.%d.block.3", b + 1);
    x = snake(c, x, p);
    snprintf(p, sizeof(p), "encoder.block.%d.block.4.conv", b + 1);
    x = causal_conv(c, x, p, ENCODER_RATES[b], 1, 1);
    if (b == 3) x = transformer(c, x, "encoder.block.4.block.5", 4, ENCODER_WINDOW);
  }
  x = snake(c, x, "encoder.block.5");
  x = causal_conv(c, x, "encoder.block.6.conv", 1, 1, 1);

  for (int i = 0; i < 2; i++) {
    char p[128];
    snprintf(p, sizeof(p), "quantizer.downsample.%d.0.conv", i);
    x = causal_conv(c, x, p, 2, 1, 1);
    snprintf(p, sizeof(p), "quantizer.downsample.%d.1", i);
    x = convnext(c, x, p);
  }
  x = transformer(c, x, "quantizer.pre_module", 8, QUANTIZER_WINDOW);

  int frames = m_dim(x, 1);
  mlx_array ids[N_RESIDUAL + 1];
  mlx_array sem_zq;
  vector_quantize(c, x, "quantizer.semantic_quantizer.quantizers.0", &ids[0], &sem_zq);
  mlx_array residual = m_sub(x, sem_zq);
  for (int q = 0; q < N_RESIDUAL; q++) {
    char p[128];
    snprintf(p, sizeof(p), "quantizer.quantizer.quantizers.%d", q);
    mlx_array zq;
    vector_quantize(c, residual, p, &ids[q + 1], &zq);
    residual = m_sub(residual, zq);
  }
  mlx_array all = m_stack(ids, N_RESIDUAL + 1, 0); /* [10, frames] */
  size_t count;
  Codes out = {0};
  out.data = m_to_int32(all, &count);
  out.n_codebooks = N_RESIDUAL + 1;
  out.frames = frames;
  /* DAC.encode reports ceil(length / frame_length) frames of the unpadded input */
  int valid = (int)((samples + c->frame_length - 1) / c->frame_length);
  if (valid < frames) {
    int32_t *trimmed = malloc((size_t)out.n_codebooks * valid * sizeof(int32_t));
    for (int q = 0; q < out.n_codebooks; q++)
      memcpy(trimmed + (size_t)q * valid, out.data + (size_t)q * frames,
             valid * sizeof(int32_t));
    free(out.data);
    out.data = trimmed;
    out.frames = valid;
  }
  arena_release(mark);
  return out;
}
