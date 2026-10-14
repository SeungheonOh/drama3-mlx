/* Byte-level BPE tokenizer reading Hugging Face tokenizer.json (Qwen3.5 layout:
 * NFC normalizer, the Qwen3.5 split regex, ByteLevel, BPE, added tokens). */
#ifndef DRAMA3_TOKENIZER_H
#define DRAMA3_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct Tokenizer Tokenizer;

Tokenizer *tok_load(const char *path);
void tok_free(Tokenizer *tok);

/* Encodes UTF-8 text. Returns a malloc'd id array (count in *n), or NULL when
 * the text is not valid UTF-8. */
int32_t *tok_encode(const Tokenizer *tok, const char *text, size_t len, size_t *n);

/* Id of an added token or vocabulary entry, -1 if absent. */
int32_t tok_token_to_id(const Tokenizer *tok, const char *token);

/* Splits text into pre-tokens (for tests): returns the number of pieces and
 * writes code-point offsets [start, end) pairs into *offsets (malloc'd). */
size_t tok_pretokenize(const char *text, size_t len, size_t **offsets);

#endif
