/* Drama 3 generation: prompt rows -> frames, and long text -> codes. */
#ifndef DRAMA3_GENERATE_H
#define DRAMA3_GENERATE_H

#include <stdbool.h>

#include "model.h"
#include "prompt.h"

#define DRAMA3_PREFILL_CHUNK_SIZE 128
#define RAS_WINDOW_FRAMES 16

typedef struct {
  SamplingParams sampling;
  /* frames per chunk; <= 0 means 600 per chunk (2048 with single_chunk) */
  int max_new_tokens;
  bool single_chunk;
  bool verbose;
} GenOptions;

typedef enum { FINISH_EOS, FINISH_LENGTH } FinishReason;

/* Chunked prefill (DRAMA3_PREFILL_CHUNK_SIZE rows at a time) of rows [R, W] into
 * the backbone and tower caches, which must already hold R rows. Returns the
 * last row's hidden [1, H] (owned); all_hidden, if given, receives [R, H]. */
mlx_array prefill_rows(Model *m, const int32_t *frames, int R, int W, float *all_hidden);

/* Samples frames after the prompt rows; codes [n_codebooks, frames] without eos. */
Codes generate_frames(Model *m, const Rows *prompt, int max_new_tokens,
                      const SamplingParams *sampling, bool verbose, FinishReason *reason);

/* Chunked generation of text with optional voice references; returns the
 * concatenated codes of every chunk. history (optional) seeds earlier takes. */
bool generate_long(Model *m, const char *text, const Turn *refs, int n_refs,
                   const Turn *takes, int n_takes, const GenOptions *opt, Codes *out,
                   char *err, size_t err_len);

#endif
