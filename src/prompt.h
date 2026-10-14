/* Drama 3 prompt rows (reference-bracket-v1) and long-text chunking. */
#ifndef DRAMA3_PROMPT_H
#define DRAMA3_PROMPT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "model.h"

#define DRAMA3_MAX_PROMPT_ROWS 8192
#define DRAMA3_DEFAULT_MAX_NEW_TOKENS 2048
#define DRAMA3_GRAPH_CAPACITY (DRAMA3_MAX_PROMPT_ROWS + DRAMA3_DEFAULT_MAX_NEW_TOKENS)
#define DRAMA3_CHUNK_BYTES 300
#define DRAMA3_CHUNK_MAX_SPEAKERS 3
#define DRAMA3_CHUNK_MAX_NEW_TOKENS 600
#define DRAMA3_MAX_SPEAKER_ID 15

/* Audio codes of one clip: [n_codebooks, frames], codebook-major. */
typedef struct {
  int32_t *data;
  int n_codebooks, frames;
} Codes;

/* A transcript with its codes: a voice reference, or an earlier generated turn. */
typedef struct {
  char *text;
  Codes codes;
} Turn;

/* Prompt rows [rows, 1 + n_codebooks], row-major. */
typedef struct {
  int32_t *data;
  int rows, width;
} Rows;

typedef enum { PROMPT_OK = 0, PROMPT_INVALID = 1, PROMPT_TOO_LONG = 2 } PromptStatus;

PromptStatus build_prompt(Model *m, const char *text, const Turn *refs, int n_refs,
                          const Turn *history, int n_history, Rows *out, char *err,
                          size_t err_len);
void rows_free(Rows *r);
void codes_free(Codes *c);

/* ---- text helpers shared with the chunker ---- */
bool is_unicode_white_space(uint32_t cp);
/* Byte length of the leading speaker tag at s (0 if none), id in *speaker. */
size_t match_speaker_tag(const char *s, size_t len, char *speaker, size_t speaker_cap);
/* Validates caller text: non-empty, no structural tokens, canonical tags. */
bool validate_text(const char *text, const char *location, char *err, size_t err_len);
/* malloc'd copy of text with "<|speaker:0|> " prefixed unless a tag leads it. */
char *with_speaker_tag(const char *text);

/* ---- chunking (fish_omni chunking.py) ---- */
typedef struct {
  char **items;
  size_t count;
} StrList;

/* Generation chunks for text, in order. */
StrList split_drama3_text(const char *text, size_t max_bytes, int max_speakers);
void strlist_free(StrList *l);

#endif
