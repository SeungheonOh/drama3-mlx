/* Prompt rows in the reference-bracket-v1 layout (fish_omni generate.py). */
#include "prompt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "unicode.h"

static const char *TTS_INSTRUCTION = "convert the provided text to speech";
static const char *REFERENCE_INSTRUCTION =
    "convert the provided text to speech reference to the following:";
static const char *MESSAGE_END = "<|im_end|>\n";
static const char *GENERATION_START = "<|im_start|>assistant\n<|voice|>";
/* Only the prompt builder writes these; other special tokens in caller text are content. */
static const char *STRUCTURAL_TOKENS[] = {"<|im_start|>", "<|im_end|>", "<|voice|>",
                                          "<|text|>", "<|interleave|>", "<|semantic:"};

bool is_unicode_white_space(uint32_t cp) {
  return (cp >= 0x9 && cp <= 0xD) || cp == 0x20 || cp == 0x85 || cp == 0xA0 ||
         cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

/* Byte offset of the first non-white-space code point. */
static size_t lstrip_offset(const char *s, size_t len) {
  size_t i = 0;
  while (i < len) {
    size_t j = i;
    if (!is_unicode_white_space(utf8_next(s, len, &j))) break;
    i = j;
  }
  return i;
}

static bool is_blank(const char *s, size_t len) { return lstrip_offset(s, len) == len; }

size_t match_speaker_tag(const char *s, size_t len, char *speaker, size_t speaker_cap) {
  static const char prefix[] = "<|speaker:";
  size_t plen = sizeof(prefix) - 1;
  if (len < plen || memcmp(s, prefix, plen)) return 0;
  size_t i = plen;
  while (i < len && s[i] >= '0' && s[i] <= '9') i++;
  if (i == plen || i + 2 > len || s[i] != '|' || s[i + 1] != '>') return 0;
  if (speaker) {
    size_t n = i - plen < speaker_cap - 1 ? i - plen : speaker_cap - 1;
    memcpy(speaker, s + plen, n);
    speaker[n] = 0;
  }
  return i + 2;
}

bool validate_text(const char *text, const char *location, char *err, size_t err_len) {
  size_t len = strlen(text);
  uint32_t *tmp = malloc((len + 1) * sizeof(uint32_t));
  bool valid_utf8 = utf8_decode(text, len, tmp) != SIZE_MAX;
  free(tmp);
  if (!valid_utf8) {
    snprintf(err, err_len, "%s: text is not valid UTF-8", location);
    return false;
  }
  if (is_blank(text, len)) {
    snprintf(err, err_len, "%s: text must not be empty", location);
    return false;
  }
  for (size_t i = 0; i < sizeof(STRUCTURAL_TOKENS) / sizeof(STRUCTURAL_TOKENS[0]); i++) {
    if (strstr(text, STRUCTURAL_TOKENS[i])) {
      snprintf(err, err_len, "%s: %s is written by the prompt builder", location,
               STRUCTURAL_TOKENS[i]);
      return false;
    }
  }
  for (const char *p = strstr(text, "<|speaker"); p; p = strstr(p + 1, "<|speaker")) {
    char digits[32] = "";
    size_t n = match_speaker_tag(p, len - (size_t)(p - text), digits, sizeof(digits));
    bool canonical = n && (digits[0] != '0' || digits[1] == 0) && strlen(digits) <= 2 &&
                     atoi(digits) <= DRAMA3_MAX_SPEAKER_ID;
    if (!canonical) {
      snprintf(err, err_len,
               "%s: speaker tags must be <|speaker:0|> to <|speaker:%d|>", location,
               DRAMA3_MAX_SPEAKER_ID);
      return false;
    }
  }
  return true;
}

char *with_speaker_tag(const char *text) {
  size_t len = strlen(text);
  size_t off = lstrip_offset(text, len);
  if (match_speaker_tag(text + off, len - off, NULL, 0)) return strdup(text);
  static const char prefix[] = "<|speaker:0|> ";
  char *out = malloc(len + sizeof(prefix));
  memcpy(out, prefix, sizeof(prefix) - 1);
  memcpy(out + sizeof(prefix) - 1, text, len + 1);
  return out;
}

/* ---- rows ---- */

typedef struct {
  int32_t *data;
  int rows, cap, width;
} RowBuf;

static void rowbuf_reserve(RowBuf *b, int extra) {
  if (b->rows + extra <= b->cap) return;
  while (b->cap < b->rows + extra) b->cap = b->cap ? b->cap * 2 : 256;
  b->data = realloc(b->data, (size_t)b->cap * b->width * sizeof(int32_t));
}

static void append_text(Model *m, RowBuf *b, const char *s) {
  size_t n;
  int32_t *ids = tok_encode(m->tok, s, strlen(s), &n);
  if (!ids) die("prompt segment is not valid UTF-8");
  rowbuf_reserve(b, (int)n);
  for (size_t i = 0; i < n; i++) {
    int32_t *row = b->data + (size_t)b->rows * b->width;
    memset(row, 0, b->width * sizeof(int32_t));
    row[0] = ids[i];
    b->rows++;
  }
  free(ids);
}

static void append_codes(Model *m, RowBuf *b, const Codes *c) {
  rowbuf_reserve(b, c->frames);
  for (int t = 0; t < c->frames; t++) {
    int32_t *row = b->data + (size_t)b->rows * b->width;
    row[0] = c->data[t] + m->cfg.semantic_start;
    for (int k = 0; k < c->n_codebooks; k++) row[1 + k] = c->data[(size_t)k * c->frames + t];
    b->rows++;
  }
}

static bool check_codes(Model *m, const Codes *c, const char *location, char *err,
                        size_t err_len) {
  if (c->n_codebooks != m->cfg.n_codebooks || c->frames < 1) {
    snprintf(err, err_len, "%s: codes must be [%d, frames >= 1]; got [%d, %d]", location,
             m->cfg.n_codebooks, c->n_codebooks, c->frames);
    return false;
  }
  for (size_t i = 0; i < (size_t)c->n_codebooks * c->frames; i++) {
    if (c->data[i] < 0 || c->data[i] >= m->cfg.code_vocab) {
      snprintf(err, err_len, "%s: codes must be in [0, %d)", location, m->cfg.code_vocab);
      return false;
    }
  }
  return true;
}

PromptStatus build_prompt(Model *m, const char *text, const Turn *refs, int n_refs,
                          const Turn *history, int n_history, Rows *out, char *err,
                          size_t err_len) {
  if (!validate_text(text, "text", err, err_len)) return PROMPT_INVALID;
  RowBuf b = {NULL, 0, 0, 1 + m->cfg.n_codebooks};
  append_text(m, &b, "<|im_start|>system\n");
  append_text(m, &b, n_refs ? REFERENCE_INSTRUCTION : TTS_INSTRUCTION);
  for (int i = 0; i < n_refs; i++) {
    char location[64];
    snprintf(location, sizeof(location), "reference %d", i + 1);
    if (!validate_text(refs[i].text, location, err, err_len) ||
        !check_codes(m, &refs[i].codes, location, err, err_len)) {
      free(b.data);
      return PROMPT_INVALID;
    }
    /* The tag must follow "Text: " directly; leading white space moves behind it. */
    char *tagged = with_speaker_tag(refs[i].text);
    size_t len = strlen(tagged);
    size_t off = lstrip_offset(tagged, len);
    size_t tag_len = match_speaker_tag(tagged + off, len - off, NULL, 0);
    size_t seg_len = 64 + len;
    char *seg = malloc(seg_len);
    int k = snprintf(seg, seg_len, "\n\n[Reference %d]\nText: ", i + 1);
    memcpy(seg + k, tagged + off, tag_len);
    k += (int)tag_len;
    memcpy(seg + k, tagged, off);
    k += (int)off;
    size_t rest = len - off - tag_len;
    memcpy(seg + k, tagged + off + tag_len, rest);
    k += (int)rest;
    strcpy(seg + k, "\nSpeech: ");
    append_text(m, &b, seg);
    append_codes(m, &b, &refs[i].codes);
    free(seg);
    free(tagged);
  }
  append_text(m, &b, MESSAGE_END);
  for (int i = 0; i < n_history; i++) {
    char location[64];
    snprintf(location, sizeof(location), "history %d", i + 1);
    if (!validate_text(history[i].text, location, err, err_len) ||
        !check_codes(m, &history[i].codes, location, err, err_len)) {
      free(b.data);
      return PROMPT_INVALID;
    }
    char *tagged = with_speaker_tag(history[i].text);
    append_text(m, &b, "<|im_start|>user\n");
    append_text(m, &b, tagged);
    append_text(m, &b, MESSAGE_END);
    append_text(m, &b, GENERATION_START);
    append_codes(m, &b, &history[i].codes);
    append_text(m, &b, MESSAGE_END);
    free(tagged);
  }
  char *tagged = with_speaker_tag(text);
  append_text(m, &b, "<|im_start|>user\n");
  append_text(m, &b, tagged);
  append_text(m, &b, MESSAGE_END);
  append_text(m, &b, GENERATION_START);
  free(tagged);
  if (b.rows > DRAMA3_MAX_PROMPT_ROWS) {
    snprintf(err, err_len, "prompt has %d rows; at most %d", b.rows, DRAMA3_MAX_PROMPT_ROWS);
    free(b.data);
    return PROMPT_TOO_LONG;
  }
  out->data = b.data;
  out->rows = b.rows;
  out->width = b.width;
  return PROMPT_OK;
}

void rows_free(Rows *r) {
  free(r->data);
  r->data = NULL;
  r->rows = 0;
}

void codes_free(Codes *c) {
  free(c->data);
  c->data = NULL;
  c->frames = 0;
}
