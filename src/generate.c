/* Generation loop, following fish_omni generate.py generate_drama3 and
 * inference.py generate_long_drama3. */
#include "generate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static mlx_array rows_array(const int32_t *data, int rows, int width) {
  return m_from_data(data, SH(rows, width), MLX_INT32);
}

mlx_array prefill_rows(Model *m, const int32_t *frames, int R, int W, float *all_hidden) {
  mlx_array hidden = null_array(); /* owned: last row's backbone hidden [1, H] */
  mlx_array cache_arrays[4 * MAX_LAYERS + 4 * MAX_CODEBOOKS + 4];
  for (int start = 0; start < R; start += DRAMA3_PREFILL_CHUNK_SIZE) {
    size_t mark = arena_mark();
    int end = start + DRAMA3_PREFILL_CHUNK_SIZE < R ? start + DRAMA3_PREFILL_CHUNK_SIZE : R;
    mlx_array chunk = rows_array(frames + (size_t)start * W, end - start, W);
    mlx_array h = backbone_forward_frames(m, chunk, start);
    h = m_squeeze(h, 0); /* [T, H] */
    /* The tower cache of row p is labelled with frame p + 1; the last prompt row
     * has no label yet, so the first decode step writes that slot. */
    int paired_end = end < R - 1 ? end : R - 1;
    if (paired_end > start) {
      int n = paired_end - start;
      mlx_array next = rows_array(frames + (size_t)(start + 1) * W, n, W);
      tower_prefill(m, m_slice_axis(h, 0, 0, n), start, m_slice_axis(chunk, 0, 0, n), next);
    }
    mlx_array last = m_slice_axis(h, 0, end - start - 1, end - start);
    size_t n = model_cache_arrays(m, cache_arrays, sizeof(cache_arrays) / sizeof(mlx_array));
    cache_arrays[n++] = last;
    cache_arrays[n++] = h;
    /* queue the chunk and build the next one while the GPU runs it */
    if (all_hidden) m_eval_many(cache_arrays, n);
    else m_async_eval_many(cache_arrays, n);
    if (all_hidden) {
      size_t count;
      float *v = m_to_float(h, &count);
      memcpy(all_hidden + (size_t)start * m->cfg.hidden, v, count * sizeof(float));
      free(v);
    }
    assign(&hidden, last);
    arena_release(mark);
  }
  return hidden;
}

Codes generate_frames(Model *m, const Rows *prompt, int max_new_tokens,
                      const SamplingParams *sampling, bool verbose, FinishReason *reason) {
  const Config *c = &m->cfg;
  int R = prompt->rows, W = prompt->width;
  int capacity = R + max_new_tokens;
  if (max_new_tokens < 1 || capacity > c->max_positions)
    die("max_new_tokens=%d does not fit after %d prompt rows", max_new_tokens, R);
  model_reset_caches(m, capacity);
  int32_t *frames = malloc((size_t)capacity * W * sizeof(int32_t));
  memcpy(frames, prompt->data, (size_t)R * W * sizeof(int32_t));

  double t0 = now_seconds();
  mlx_array hidden = prefill_rows(m, frames, R, W, NULL);
  if (verbose) log_info("prefill: %d rows in %.2fs", R, now_seconds() - t0);

  /* Main tokens of the last RAS_WINDOW_FRAMES rows; kept on the device so a step
   * can be queued before the previous one has been read back. */
  int hist_len = R < RAS_WINDOW_FRAMES ? R : RAS_WINDOW_FRAMES;
  mlx_array hist = null_array(), frame = null_array();
  {
    int32_t history[RAS_WINDOW_FRAMES];
    for (int i = 0; i < hist_len; i++) history[i] = frames[(size_t)(R - hist_len + i) * W];
    size_t mark = arena_mark();
    assign(&hist, m_from_int32(history, hist_len));
    assign(&frame, rows_array(frames + (size_t)(R - 1) * W, 1, W));
    arena_release(mark);
  }

  /* Pipelined decode: step t+1's graph is built and queued while the GPU still
   * runs step t; step t's frame is read back afterwards. A step queued after the
   * eos frame is discarded. The random draws happen in the same order as an
   * unpipelined loop, so a seed gives the same frames. */
  int produced = 0;
  *reason = FINISH_LENGTH;
  double t1 = now_seconds();
  mlx_array pending = null_array(), pending_finite = null_array();
  bool done = false;
  for (int step = 0; step <= max_new_tokens && !done; step++) {
    mlx_array queued = null_array(), queued_finite = null_array();
    if (step < max_new_tokens) {
      size_t mark = arena_mark();
      int position = R - 1 + step;
      mlx_array logits = backbone_audio_logits(m, hidden);
      mlx_array finite = m_all(m_isfinite(logits));
      mlx_array prefix;
      mlx_array sampled =
          tower_decode_step(m, logits, hidden, position, frame, hist, sampling, &prefix);
      mlx_array main_tok = m_reshape(m_slice_axis(sampled, 1, 0, 1), SH(1));
      mlx_array next_hist = hist_len == RAS_WINDOW_FRAMES
                                ? m_concat2(m_slice_axis(hist, 0, 1, RAS_WINDOW_FRAMES), main_tok, 0)
                                : m_concat2(hist, main_tok, 0);
      if (hist_len < RAS_WINDOW_FRAMES) hist_len++;
      /* Evaluation order: the frame's leading codes (all the next backbone step
       * reads), the rest of the tower, then the next backbone step, so the frame
       * can be read back before the backbone step finishes. */
      mlx_array caches[4 * MAX_LAYERS + 4 * MAX_CODEBOOKS];
      size_t n_caches = model_cache_arrays(m, caches, sizeof(caches) / sizeof(caches[0]));
      size_t n_backbone = 2 * (size_t)c->n_layers;
      mlx_array outs[4 * MAX_LAYERS + 4 * MAX_CODEBOOKS + 8];
      size_t n = 0;
      outs[n++] = prefix;
      outs[n++] = sampled;
      outs[n++] = next_hist;
      for (size_t i = n_backbone; i < n_caches; i++) outs[n++] = caches[i];
      outs[n++] = finite;
      if (step + 1 < max_new_tokens) {
        mlx_array h = backbone_forward_frames(m, prefix, position + 1);
        assign(&hidden, m_reshape(h, SH(1, c->hidden)));
        n_caches = model_cache_arrays(m, caches, sizeof(caches) / sizeof(caches[0]));
      }
      for (size_t i = 0; i < n_backbone; i++) outs[n++] = caches[i];
      if (step + 1 < max_new_tokens) outs[n++] = hidden;
      m_async_eval_many(outs, n);
      queued = own(sampled);
      queued_finite = own(finite);
      assign(&frame, sampled);
      assign(&hist, next_hist);
      arena_release(mark);
    }
    if (!is_null(pending)) {
      int position = R - 1 + step - 1;
      if (!m_item_int(pending_finite)) die("non-finite logits; refusing to sample invalid audio");
      size_t count;
      int32_t *row = m_to_int32(pending, &count);
      if (row[0] == c->eos_id) {
        *reason = FINISH_EOS;
        done = true;
      } else {
        memcpy(frames + (size_t)(position + 1) * W, row, W * sizeof(int32_t));
        produced++;
        if (verbose && (produced % 50 == 0))
          log_info("  %d frames (%.1f frames/s)", produced, produced / (now_seconds() - t1));
      }
      free(row);
      release(&pending);
      release(&pending_finite);
    }
    pending = queued;
    pending_finite = queued_finite;
  }
  release(&pending);
  release(&pending_finite);
  release(&frame);
  release(&hist);
  /* the step queued after eos may still be running; let it finish before the
   * caches are reused */
  mlx_synchronize(g_stream);
  if (verbose && produced)
    log_info("decode: %d frames in %.2fs (%.2f frames/s), finish=%s", produced,
             now_seconds() - t1, produced / (now_seconds() - t1),
             *reason == FINISH_EOS ? "eos" : "length");
  release(&hidden);

  Codes out = {0};
  out.n_codebooks = c->n_codebooks;
  out.frames = produced;
  out.data = malloc(((size_t)c->n_codebooks * produced + 1) * sizeof(int32_t));
  for (int t = 0; t < produced; t++)
    for (int k = 0; k < c->n_codebooks; k++)
      out.data[(size_t)k * produced + t] = frames[(size_t)(R + t) * W + 1 + k];
  free(frames);
  return out;
}

/* ---- long text ---- */

/* The first chunk stays pinned, plus the latest chunk. */
typedef struct {
  Turn pinned, recent;
  bool has_pinned, has_recent;
} History;

static void history_push(History *h, const char *text, Codes codes) {
  Turn t = {strdup(text), codes};
  if (!h->has_pinned) {
    h->pinned = t;
    h->has_pinned = true;
  } else {
    if (h->has_recent) {
      free(h->recent.text);
      codes_free(&h->recent.codes);
    }
    h->recent = t;
    h->has_recent = true;
  }
}

static void history_free(History *h) {
  if (h->has_pinned) {
    free(h->pinned.text);
    codes_free(&h->pinned.codes);
  }
  if (h->has_recent) {
    free(h->recent.text);
    codes_free(&h->recent.codes);
  }
}

/* Earlier takes (previous_takes) come first, then the chunk history; a prompt
 * that is too long drops the oldest recent turns, then the pinned one. */
static PromptStatus chunk_prompt(Model *m, const char *chunk, const Turn *refs, int n_refs,
                                 const Turn *takes, int n_takes, const History *h, Rows *out,
                                 char *err, size_t err_len) {
  Turn turns[64];
  int candidates[3][2]; /* (use pinned, use recent) */
  int n_candidates = 0;
  if (h->has_pinned) {
    if (h->has_recent) {
      candidates[n_candidates][0] = 1;
      candidates[n_candidates++][1] = 1;
    }
    candidates[n_candidates][0] = 1;
    candidates[n_candidates++][1] = 0;
  }
  candidates[n_candidates][0] = 0;
  candidates[n_candidates++][1] = 0;
  PromptStatus status = PROMPT_TOO_LONG;
  for (int ci = 0; ci < n_candidates; ci++) {
    int n = 0;
    for (int i = 0; i < n_takes && n < 60; i++) turns[n++] = takes[i];
    if (candidates[ci][0]) turns[n++] = h->pinned;
    if (candidates[ci][1]) turns[n++] = h->recent;
    status = build_prompt(m, chunk, refs, n_refs, turns, n, out, err, err_len);
    if (status != PROMPT_TOO_LONG) return status;
  }
  return status;
}

bool generate_long(Model *m, const char *text, const Turn *refs, int n_refs,
                   const Turn *takes, int n_takes, const GenOptions *opt, Codes *out,
                   char *err, size_t err_len) {
  const Config *c = &m->cfg;
  StrList chunks = {0};
  if (!opt->single_chunk)
    chunks = split_drama3_text(text, DRAMA3_CHUNK_BYTES, DRAMA3_CHUNK_MAX_SPEAKERS);
  if (!chunks.count) {
    chunks.items = malloc(sizeof(char *));
    chunks.items[0] = strdup(text);
    chunks.count = 1;
  }
  int capacity = DRAMA3_GRAPH_CAPACITY < c->max_positions ? DRAMA3_GRAPH_CAPACITY
                                                           : c->max_positions;
  /* Reject bad text in any chunk before generating. */
  for (size_t i = 0; i < chunks.count; i++) {
    Rows r;
    PromptStatus s =
        build_prompt(m, chunks.items[i], refs, n_refs, takes, n_takes, &r, err, err_len);
    if (s != PROMPT_OK) {
      strlist_free(&chunks);
      return false;
    }
    bool full = r.rows >= capacity;
    int rows = r.rows;
    rows_free(&r);
    if (full) {
      snprintf(err, err_len, "prompt has %d rows; no room left to generate", rows);
      strlist_free(&chunks);
      return false;
    }
  }
  int requested = opt->max_new_tokens;
  if (requested <= 0)
    requested = opt->single_chunk ? DRAMA3_DEFAULT_MAX_NEW_TOKENS : DRAMA3_CHUNK_MAX_NEW_TOKENS;
  if (opt->verbose)
    log_info("drama3: %zu chunk(s), %d reference(s), max_new_tokens=%d per chunk",
             chunks.count, n_refs, requested);

  History history = {0};
  Codes all = {NULL, c->n_codebooks, 0};
  bool ok = true;
  for (size_t i = 0; i < chunks.count; i++) {
    Rows prompt;
    if (chunk_prompt(m, chunks.items[i], refs, n_refs, takes, n_takes, &history, &prompt, err,
                     err_len) != PROMPT_OK) {
      ok = false;
      break;
    }
    int limit = capacity - prompt.rows;
    if (requested > limit && opt->verbose)
      log_info("max_new_tokens=%d exceeds the %d frames left after the %d-row prompt; "
               "clamping",
               requested, limit, prompt.rows);
    int budget = requested < limit ? requested : limit;
    if (opt->verbose)
      log_info("chunk %zu/%zu (%d-row prompt): %s", i + 1, chunks.count, prompt.rows,
               chunks.items[i]);
    FinishReason reason;
    Codes codes = generate_frames(m, &prompt, budget, &opt->sampling, opt->verbose, &reason);
    rows_free(&prompt);
    if (reason != FINISH_EOS)
      log_info("warning: reached max_new_tokens before EOS; audio may be truncated");
    if (!codes.frames) {
      log_info("warning: the model emitted EOS before any audio frame");
      codes_free(&codes);
      continue;
    }
    /* append to the running output */
    int total = all.frames + codes.frames;
    int32_t *merged = malloc((size_t)c->n_codebooks * total * sizeof(int32_t));
    for (int k = 0; k < c->n_codebooks; k++) {
      memcpy(merged + (size_t)k * total, all.data + (size_t)k * all.frames,
             all.frames * sizeof(int32_t));
      memcpy(merged + (size_t)k * total + all.frames, codes.data + (size_t)k * codes.frames,
             codes.frames * sizeof(int32_t));
    }
    free(all.data);
    all.data = merged;
    all.frames = total;
    history_push(&history, chunks.items[i], codes);
  }
  history_free(&history);
  strlist_free(&chunks);
  if (!ok) {
    codes_free(&all);
    return false;
  }
  *out = all;
  return true;
}
