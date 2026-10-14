/* Unicode support for the tokenizer: UTF-8, general categories, NFC. */
#ifndef DRAMA3_UNICODE_H
#define DRAMA3_UNICODE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum { UNI_L = 1, UNI_M = 2, UNI_N = 4, UNI_WS = 8, UNI_DIGIT = 16, UNI_NUMERIC = 32 };

extern const uint32_t UNI_CAT_RANGES[][3];
extern const size_t UNI_CAT_RANGES_LEN;
extern const uint32_t UNI_CCC_RANGES[][3];
extern const size_t UNI_CCC_RANGES_LEN;
extern const uint32_t UNI_DECOMP_INDEX[][3];
extern const size_t UNI_DECOMP_INDEX_LEN;
extern const uint32_t UNI_DECOMP_POOL[];
extern const uint32_t UNI_COMPOSE[][3];
extern const size_t UNI_COMPOSE_LEN;

/* Category flags (UNI_L | UNI_M | UNI_N | UNI_WS) of a code point. */
unsigned uni_flags(uint32_t cp);
unsigned uni_ccc(uint32_t cp);

/* Decodes UTF-8 into code points (out must hold len entries).
 * Returns the number of code points, or SIZE_MAX for invalid UTF-8. */
size_t utf8_decode(const char *s, size_t len, uint32_t *out);
/* Decodes the code point at s[*i] of valid UTF-8 and advances *i past it. */
uint32_t utf8_next(const char *s, size_t len, size_t *i);
/* Encodes code points; out must hold 4 * count bytes. Returns bytes written. */
size_t utf8_encode(const uint32_t *cps, size_t count, char *out);

/* NFC-normalizes code points; returns a malloc'd array and its length. */
uint32_t *uni_nfc(const uint32_t *cps, size_t count, size_t *out_count);

#endif
