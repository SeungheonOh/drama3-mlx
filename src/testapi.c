/* Entry points for the Python parity tests (loaded through ctypes). */
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"
#include "unicode.h"

#define API __attribute__((visibility("default")))

API void *d3_tok_load(const char *path) { return tok_load(path); }

API void d3_tok_free(void *tok) { tok_free(tok); }

/* Returns the id count (ids copied up to cap), or -1 for invalid UTF-8. */
API long d3_tok_encode(void *tok, const char *text, size_t len, int *out, size_t cap) {
  size_t n;
  int32_t *ids = tok_encode(tok, text, len, &n);
  if (!ids) return -1;
  memcpy(out, ids, (n < cap ? n : cap) * sizeof(int32_t));
  free(ids);
  return (long)n;
}

API long d3_tok_token_to_id(void *tok, const char *token) {
  return tok_token_to_id(tok, token);
}

/* Code-point offsets of the pre-tokens: [start0, end0, start1, ...]. */
API long d3_pretokenize(const char *text, size_t len, size_t *out, size_t cap) {
  size_t *offsets;
  size_t n = tok_pretokenize(text, len, &offsets);
  if (!offsets) return -1;
  memcpy(out, offsets, (2 * n < cap ? 2 * n : cap) * sizeof(size_t));
  free(offsets);
  return (long)n;
}

/* NFC of a UTF-8 string into out (UTF-8); returns the byte length. */
API long d3_nfc(const char *text, size_t len, char *out, size_t cap) {
  uint32_t *cps = malloc((len + 1) * sizeof(uint32_t));
  size_t n = utf8_decode(text, len, cps);
  if (n == SIZE_MAX) {
    free(cps);
    return -1;
  }
  size_t nn;
  uint32_t *norm = uni_nfc(cps, n, &nn);
  char *buf = malloc(nn * 4 + 1);
  size_t bytes = utf8_encode(norm, nn, buf);
  memcpy(out, buf, bytes < cap ? bytes : cap);
  free(buf);
  free(norm);
  free(cps);
  return (long)bytes;
}

/* ---- model ---- */
#include "codec.h"
#include "generate.h"
#include "model.h"
#include "prompt.h"
#include "resample.h"

static void copy_out(mlx_array a, float *out, size_t cap) {
  size_t n;
  float *v = m_to_float(a, &n);
  memcpy(out, v, (n < cap ? n : cap) * sizeof(float));
  free(v);
}

API void *d3_model_load(const char *dir) { return model_load(dir); }
API void *d3_model_load_light(const char *dir) { return model_load_light(dir); }
API void d3_model_free(void *m) { model_free(m); }
API void d3_reset(void *m, int capacity) { model_reset_caches(m, capacity); }

/* frames int32 [T, W] -> backbone input embeddings [T, H] */
API void d3_embed_frames(void *mp, const int *frames, int T, float *out, size_t cap) {
  Model *m = mp;
  size_t mark = arena_mark();
  mlx_array f = m_from_data(frames, SH(T, 1 + m->cfg.n_codebooks), MLX_INT32);
  copy_out(backbone_embed_frames(m, f), out, cap);
  arena_release(mark);
}

static mlx_array input_bf16(Model *m, const float *x, int T) {
  return m_astype(m_from_data(x, SH(1, T, m->cfg.hidden), MLX_FLOAT32), MLX_BFLOAT16);
}

static void eval_caches(Model *m, mlx_array result) {
  mlx_array arrays[4 * MAX_LAYERS + 4 * MAX_CODEBOOKS + 1];
  size_t n = model_cache_arrays(m, arrays, sizeof(arrays) / sizeof(arrays[0]) - 1);
  arrays[n++] = result;
  m_eval_many(arrays, n);
}

/* which: 0 whole layer, 1 mixer (attention / delta net), 2 MoE */
API void d3_layer(void *mp, int layer, int which, const float *x, int T, int pos, float *out,
                  size_t cap) {
  Model *m = mp;
  size_t mark = arena_mark();
  mlx_array in = input_bf16(m, x, T);
  mlx_array r = which == 0   ? backbone_layer_forward(m, layer, in, pos)
                : which == 1 ? backbone_mixer_forward(m, layer, in, pos)
                             : backbone_moe_forward(m, layer, in);
  eval_caches(m, r);
  copy_out(r, out, cap);
  arena_release(mark);
}

API void d3_backbone(void *mp, const float *x, int T, int pos, float *out, size_t cap) {
  Model *m = mp;
  size_t mark = arena_mark();
  mlx_array r = backbone_forward(m, input_bf16(m, x, T), pos);
  eval_caches(m, r);
  copy_out(r, out, cap);
  arena_release(mark);
}

API void d3_audio_logits(void *mp, const float *hidden, int N, float *out, size_t cap) {
  Model *m = mp;
  size_t mark = arena_mark();
  mlx_array h = m_astype(m_from_data(hidden, SH(N, m->cfg.hidden), MLX_FLOAT32), MLX_BFLOAT16);
  copy_out(backbone_audio_logits(m, h), out, cap);
  arena_release(mark);
}

API void d3_tower_prefill(void *mp, const float *hidden, int T, int pos, const int *frames,
                          const int *next_frames) {
  Model *m = mp;
  size_t mark = arena_mark();
  int W = 1 + m->cfg.n_codebooks;
  mlx_array h = m_astype(m_from_data(hidden, SH(T, m->cfg.hidden), MLX_FLOAT32), MLX_BFLOAT16);
  tower_prefill(m, h, pos, m_from_data(frames, SH(T, W), MLX_INT32),
                m_from_data(next_frames, SH(T, W), MLX_INT32));
  mlx_array arrays[4 * MAX_LAYERS + 4 * MAX_CODEBOOKS];
  size_t n = model_cache_arrays(m, arrays, sizeof(arrays) / sizeof(arrays[0]));
  m_eval_many(arrays, n);
  arena_release(mark);
}

API void d3_tower_forced(void *mp, const float *hidden, int pos, const int *frame,
                         const int *next_frame, float *out, size_t cap) {
  Model *m = mp;
  size_t mark = arena_mark();
  int W = 1 + m->cfg.n_codebooks;
  mlx_array h = m_astype(m_from_data(hidden, SH(1, m->cfg.hidden), MLX_FLOAT32), MLX_BFLOAT16);
  mlx_array r = tower_forced_logits(m, h, pos, m_from_data(frame, SH(1, W), MLX_INT32),
                                    m_from_data(next_frame, SH(1, W), MLX_INT32));
  eval_caches(m, r);
  copy_out(r, out, cap);
  arena_release(mark);
}

/* Sampling: ids for each row of logits [N, V]. */
API void d3_sample(const float *logits, int N, int V, float temperature, int top_k,
                   float top_p, unsigned long long seed, int *out) {
  mlxu_init();
  size_t mark = arena_mark();
  m_seed(seed);
  mlx_array l = m_from_data(logits, SH(N, V), MLX_FLOAT32);
  size_t n;
  int32_t *ids = m_to_int32(sample_top_k_top_p(l, temperature, top_k, top_p), &n);
  memcpy(out, ids, n * sizeof(int32_t));
  free(ids);
  arena_release(mark);
}

/* ---- prompt / chunking ---- */

/* refs: n_refs transcripts + codes [N, frames_i] concatenated; returns rows or -1/-2. */
API long d3_build_prompt(void *mp, const char *text, int n_refs, const char **ref_texts,
                         const int *ref_codes, const int *ref_frames, int n_hist,
                         const char **hist_texts, const int *hist_codes,
                         const int *hist_frames, int *out, size_t cap, char *err,
                         size_t err_len) {
  Model *m = mp;
  int N = m->cfg.n_codebooks;
  Turn refs[64], hist[64];
  const int *p = ref_codes;
  for (int i = 0; i < n_refs; i++) {
    refs[i].text = (char *)ref_texts[i];
    refs[i].codes = (Codes){(int32_t *)p, N, ref_frames[i]};
    p += (size_t)N * ref_frames[i];
  }
  p = hist_codes;
  for (int i = 0; i < n_hist; i++) {
    hist[i].text = (char *)hist_texts[i];
    hist[i].codes = (Codes){(int32_t *)p, N, hist_frames[i]};
    p += (size_t)N * hist_frames[i];
  }
  Rows rows;
  PromptStatus s = build_prompt(m, text, refs, n_refs, hist, n_hist, &rows, err, err_len);
  if (s != PROMPT_OK) return -(long)s;
  size_t count = (size_t)rows.rows * rows.width;
  memcpy(out, rows.data, (count < cap ? count : cap) * sizeof(int32_t));
  long r = rows.rows;
  rows_free(&rows);
  return r;
}

/* Chunks joined by \x1e into out; returns the chunk count. */
API long d3_split_text(const char *text, size_t max_bytes, int max_speakers, char *out,
                       size_t cap) {
  StrList l = split_drama3_text(text, max_bytes, max_speakers);
  size_t used = 0;
  for (size_t i = 0; i < l.count; i++) {
    size_t len = strlen(l.items[i]);
    if (used + len + 2 > cap) break;
    if (i) out[used++] = '\x1e';
    memcpy(out + used, l.items[i], len);
    used += len;
  }
  out[used] = 0;
  long n = (long)l.count;
  strlist_free(&l);
  return n;
}

/* ---- codec / audio ---- */

API void *d3_codec_load(const char *path, int dtype) {
  return codec_load(path, dtype == 0 ? MLX_FLOAT32 : dtype == 1 ? MLX_BFLOAT16 : MLX_FLOAT16);
}
API void d3_codec_free(void *c) {
  codec_free(c);
  mlx_clear_cache();
}

API size_t d3_active_memory(void) {
  size_t n = 0;
  mlx_get_active_memory(&n);
  return n;
}

API long d3_codec_decode(void *c, const int *codes, int frames, float *out, size_t cap) {
  Codes in = {(int32_t *)codes, 10, frames};
  size_t n;
  float *audio = codec_decode(c, &in, &n);
  memcpy(out, audio, (n < cap ? n : cap) * sizeof(float));
  free(audio);
  return (long)n;
}

API long d3_codec_encode(void *c, const float *audio, size_t samples, int *out, size_t cap) {
  Codes codes = codec_encode(c, audio, samples);
  size_t n = (size_t)codes.n_codebooks * codes.frames;
  memcpy(out, codes.data, (n < cap ? n : cap) * sizeof(int32_t));
  long frames = codes.frames;
  codes_free(&codes);
  return frames;
}

API long d3_resample(const float *in, size_t n, int orig, int new_rate, float *out, size_t cap) {
  size_t count;
  float *r = resample(in, n, orig, new_rate, &count);
  memcpy(out, r, (count < cap ? count : cap) * sizeof(float));
  free(r);
  return (long)count;
}

API void d3_tower_decode(void *mp, const float *logits, const float *hidden, int pos,
                         const int *frame, const int *history, int n_history,
                         float temperature, int top_k, float top_p, unsigned long long seed,
                         int *out) {
  Model *m = mp;
  size_t mark = arena_mark();
  m_seed(seed);
  int W = 1 + m->cfg.n_codebooks;
  mlx_array l = m_from_data(logits, SH(1, 1 + m->cfg.code_vocab), MLX_FLOAT32);
  mlx_array h = m_astype(m_from_data(hidden, SH(1, m->cfg.hidden), MLX_FLOAT32), MLX_BFLOAT16);
  SamplingParams p = {temperature, top_p, top_k};
  mlx_array r = tower_decode_step(m, m_astype(l, MLX_BFLOAT16), h, pos,
                                  m_from_data(frame, SH(1, W), MLX_INT32),
                                  m_from_data(history, SH(n_history), MLX_INT32), &p);
  eval_caches(m, r);
  size_t n;
  int32_t *v = m_to_int32(r, &n);
  memcpy(out, v, n * sizeof(int32_t));
  free(v);
  arena_release(mark);
}

/* generate_frames on prompt rows; codes [n_codebooks, cap / n_codebooks] (frames-major
 * per codebook). Returns the number of frames (negative when eos was not reached). */
API long d3_generate(void *mp, const int *rows, int R, int max_new, float temperature,
                     int top_k, float top_p, unsigned long long seed, int *out, size_t cap) {
  Model *m = mp;
  m_seed(seed);
  Rows prompt = {(int32_t *)rows, R, 1 + m->cfg.n_codebooks};
  SamplingParams p = {temperature, top_p, top_k};
  FinishReason reason;
  Codes codes = generate_frames(m, &prompt, max_new, &p, false, &reason);
  size_t n = (size_t)codes.n_codebooks * codes.frames;
  memcpy(out, codes.data, (n < cap ? n : cap) * sizeof(int32_t));
  long frames = codes.frames;
  codes_free(&codes);
  return reason == FINISH_EOS ? frames : -frames;
}

/* Production chunked prefill of prompt rows; returns every row's hidden [R, H]. */
API void d3_prefill(void *mp, const int *rows, int R, int capacity, float *out_hidden) {
  Model *m = mp;
  model_reset_caches(m, capacity);
  mlx_array last = prefill_rows(m, rows, R, 1 + m->cfg.n_codebooks, out_hidden);
  release(&last);
}

#include "reference.h"

/* WAV file -> codes through the CLI's reference path; returns frames or -1. */
API long d3_encode_reference(void *c, const char *path, int *out, size_t cap, char *err,
                             size_t err_len) {
  Codes codes;
  if (!encode_reference(c, path, &codes, false, err, err_len)) return -1;
  size_t n = (size_t)codes.n_codebooks * codes.frames;
  memcpy(out, codes.data, (n < cap ? n : cap) * sizeof(int32_t));
  long frames = codes.frames;
  codes_free(&codes);
  return frames;
}
