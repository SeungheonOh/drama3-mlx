/* Long-text chunking for Drama 3, ported from fish_omni chunking.py.
 *
 * Text is cut into chunks of about DRAMA3_CHUNK_BYTES UTF-8 bytes with at most
 * DRAMA3_CHUNK_MAX_SPEAKERS speakers, preferring sentence ends, then clause
 * marks, then spaces. Work happens on code points, like the Python original.
 */
#include <stdlib.h>
#include <string.h>

#include "prompt.h"
#include "unicode.h"

typedef struct {
  uint32_t *v;
  size_t n, cap;
} U32;

static void u32_push(U32 *s, uint32_t c) {
  if (s->n == s->cap) {
    s->cap = s->cap ? s->cap * 2 : 32;
    s->v = realloc(s->v, s->cap * sizeof(uint32_t));
  }
  s->v[s->n++] = c;
}

static void u32_append(U32 *s, const uint32_t *v, size_t n) {
  for (size_t i = 0; i < n; i++) u32_push(s, v[i]);
}

static U32 u32_copy(const uint32_t *v, size_t n) {
  U32 s = {0};
  u32_append(&s, v, n);
  return s;
}

static size_t cp_bytes(uint32_t c) { return c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4; }

static size_t utf8_len32(const uint32_t *v, size_t n) {
  size_t total = 0;
  for (size_t i = 0; i < n; i++) total += cp_bytes(v[i]);
  return total;
}

static bool blank32(const uint32_t *v, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (!is_unicode_white_space(v[i])) return false;
  return true;
}

static char *to_utf8(const uint32_t *v, size_t n) {
  char *out = malloc(n * 4 + 1);
  size_t len = utf8_encode(v, n, out);
  out[len] = 0;
  return out;
}

static void strlist_push(StrList *l, char *s) {
  l->items = realloc(l->items, (l->count + 1) * sizeof(char *));
  l->items[l->count++] = s;
}

void strlist_free(StrList *l) {
  for (size_t i = 0; i < l->count; i++) free(l->items[i]);
  free(l->items);
  l->items = NULL;
  l->count = 0;
}

/* ---- pieces ---- */

typedef struct {
  U32 text;
  bool protected_;
} Piece;

typedef struct {
  Piece *v;
  size_t n, cap;
} Pieces;

static void pieces_push(Pieces *p, const uint32_t *v, size_t n, bool protected_) {
  if (p->n == p->cap) {
    p->cap = p->cap ? p->cap * 2 : 16;
    p->v = realloc(p->v, p->cap * sizeof(Piece));
  }
  p->v[p->n].text = u32_copy(v, n);
  p->v[p->n].protected_ = protected_;
  p->n++;
}

static void pieces_free(Pieces *p) {
  for (size_t i = 0; i < p->n; i++) free(p->v[i].text.v);
  free(p->v);
}

/* Python's \w for str patterns: alphanumeric (isalnum) or underscore. */
static bool is_word(uint32_t c) {
  return c == '_' || (uni_flags(c) & (UNI_L | UNI_NUMERIC));
}

static bool ascii_letter(uint32_t c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

/* PROTECTED_SPAN = <\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]
 * (re.S). Returns the match length at i, 0 if none. */
static size_t protected_at(const uint32_t *t, size_t n, size_t i) {
  if (t[i] == '<' && i + 1 < n && t[i + 1] == '|') {
    size_t j = i + 2;
    while (j < n && t[j] != '|' && t[j] != '>') j++;
    if (j + 1 < n && t[j] == '|' && t[j + 1] == '>') return j + 2 - i;
  }
  if (t[i] == '<' && i + 1 < n && ascii_letter(t[i + 1])) {
    size_t e_max = i + 2;
    while (e_max < n && (is_word(t[e_max]) || t[e_max] == '-')) e_max++;
    for (size_t e = e_max; e >= i + 2; e--) {
      bool before = is_word(t[e - 1]);
      bool after = e < n && is_word(t[e]);
      if (before == after) continue; /* \b */
      size_t g = e;
      while (g < n && t[g] != '>') g++;
      if (g >= n) break; /* no '>' at all: no shorter name can match either */
      size_t name_len = e - (i + 1);
      for (size_t s = g + 1; s + name_len + 3 <= n; s++) {
        if (t[s] == '<' && t[s + 1] == '/' &&
            !memcmp(t + s + 2, t + i + 1, name_len * sizeof(uint32_t)) &&
            t[s + 2 + name_len] == '>')
          return s + 3 + name_len - i;
      }
    }
  }
  if (t[i] == '[') {
    size_t j = i + 1;
    while (j < n && t[j] != '[' && t[j] != ']') j++;
    if (j < n && t[j] == ']') return j + 1 - i;
  }
  return 0;
}

static Pieces protected_pieces(const uint32_t *t, size_t n) {
  Pieces out = {0};
  size_t last = 0;
  for (size_t i = 0; i < n;) {
    size_t len = protected_at(t, n, i);
    if (!len) {
      i++;
      continue;
    }
    if (i > last) pieces_push(&out, t + last, i - last, false);
    pieces_push(&out, t + i, len, true);
    i += len;
    last = i;
  }
  if (last < n) pieces_push(&out, t + last, n - last, false);
  return out;
}

static bool in_marks(uint32_t c, const uint32_t *marks) {
  for (; *marks; marks++)
    if (*marks == c) return true;
  return false;
}

static bool is_digit(uint32_t c) { return uni_flags(c) & UNI_DIGIT; }

static bool breaks_at(const U32 *t, size_t i, const uint32_t *marks) {
  if (!in_marks(t->v[i], marks)) return false;
  /* Keep decimals such as 3.14 in one piece. */
  return !(t->v[i] == '.' && i > 0 && i + 1 < t->n && is_digit(t->v[i - 1]) &&
           is_digit(t->v[i + 1]));
}

static Pieces break_at_marks(Pieces in, size_t length, const uint32_t *marks) {
  Pieces out = {0};
  for (size_t p = 0; p < in.n; p++) {
    U32 *t = &in.v[p].text;
    if (in.v[p].protected_ || utf8_len32(t->v, t->n) <= length) {
      pieces_push(&out, t->v, t->n, in.v[p].protected_);
      continue;
    }
    size_t start = 0;
    for (size_t i = 0; i < t->n; i++) {
      if (breaks_at(t, i, marks)) {
        pieces_push(&out, t->v + start, i + 1 - start, false);
        start = i + 1;
      }
    }
    if (start < t->n) pieces_push(&out, t->v + start, t->n - start, false);
  }
  pieces_free(&in);
  return out;
}

static Pieces break_by_length(Pieces in, size_t length) {
  Pieces out = {0};
  for (size_t p = 0; p < in.n; p++) {
    U32 *t = &in.v[p].text;
    if (in.v[p].protected_ || utf8_len32(t->v, t->n) <= length) {
      pieces_push(&out, t->v, t->n, in.v[p].protected_);
      continue;
    }
    size_t start = 0, size = 0;
    for (size_t i = 0; i < t->n; i++) {
      size_t b = cp_bytes(t->v[i]);
      if (i > start && size + b > length) {
        pieces_push(&out, t->v + start, i - start, false);
        start = i;
        size = 0;
      }
      size += b;
    }
    pieces_push(&out, t->v + start, t->n - start, false);
  }
  pieces_free(&in);
  return out;
}

static const uint32_t SENTENCE_ENDS[] = {'.', '!', '?', ';', 0x3002, 0xFF01, 0xFF1F,
                                         0xFF1B, 0x2026, '\n', 0};
static const uint32_t CLAUSE_MARKS[] = {',', ':', 0xFF0C, 0xFF1A, 0x3001, 0};
static const uint32_t SPACES[] = {' ', 0};

static void strip32(const uint32_t *v, size_t n, size_t *start, size_t *end) {
  size_t s = 0, e = n;
  while (s < e && is_unicode_white_space(v[s])) s++;
  while (e > s && is_unicode_white_space(v[e - 1])) e--;
  *start = s;
  *end = e;
}

/* split_by_length: chunks of about length bytes, packed sentence by sentence. */
static void split_by_length(const uint32_t *t, size_t n, size_t length, StrList *out) {
  if (blank32(t, n)) return;
  if (utf8_len32(t, n) <= length) {
    strlist_push(out, to_utf8(t, n));
    return;
  }
  Pieces pieces = break_at_marks(protected_pieces(t, n), 0, SENTENCE_ENDS);
  pieces = break_at_marks(pieces, length, CLAUSE_MARKS);
  pieces = break_at_marks(pieces, length, SPACES);
  pieces = break_by_length(pieces, length);

  U32 current = {0};
  for (size_t p = 0; p <= pieces.n; p++) {
    bool flush_last = p == pieces.n;
    if (!flush_last) {
      U32 *piece = &pieces.v[p].text;
      if (current.n &&
          utf8_len32(current.v, current.n) + utf8_len32(piece->v, piece->n) > length) {
        size_t s, e;
        strip32(current.v, current.n, &s, &e);
        if (e > s) strlist_push(out, to_utf8(current.v + s, e - s));
        current.n = 0;
      }
      u32_append(&current, piece->v, piece->n);
    } else {
      size_t s, e;
      strip32(current.v, current.n, &s, &e);
      if (e > s) strlist_push(out, to_utf8(current.v + s, e - s));
    }
  }
  free(current.v);
  pieces_free(&pieces);
}

/* <\|speaker:([0-9]+)\|> at i; returns the match length, digits in [ds, de). */
static size_t speaker_tag_at(const uint32_t *t, size_t n, size_t i, size_t *ds, size_t *de) {
  static const char prefix[] = "<|speaker:";
  size_t plen = sizeof(prefix) - 1;
  if (i + plen > n) return 0;
  for (size_t k = 0; k < plen; k++)
    if (t[i + k] != (uint32_t)prefix[k]) return 0;
  size_t j = i + plen;
  while (j < n && t[j] >= '0' && t[j] <= '9') j++;
  if (j == i + plen || j + 1 >= n || t[j] != '|' || t[j + 1] != '>') return 0;
  *ds = i + plen;
  *de = j;
  return j + 2 - i;
}

StrList split_drama3_text(const char *text, size_t max_bytes, int max_speakers) {
  StrList chunks = {0};
  size_t len = strlen(text);
  uint32_t *t = malloc((len + 1) * sizeof(uint32_t));
  size_t n = utf8_decode(text, len, t);
  if (n == SIZE_MAX) {
    free(t);
    return chunks;
  }
  /* Find tags (leftmost, non-overlapping). */
  size_t *tag_start = NULL, *tag_len = NULL, *dig_s = NULL, *dig_e = NULL;
  size_t tags = 0;
  for (size_t i = 0; i < n;) {
    size_t ds, de, l = speaker_tag_at(t, n, i, &ds, &de);
    if (!l) {
      i++;
      continue;
    }
    tag_start = realloc(tag_start, (tags + 1) * sizeof(size_t));
    tag_len = realloc(tag_len, (tags + 1) * sizeof(size_t));
    dig_s = realloc(dig_s, (tags + 1) * sizeof(size_t));
    dig_e = realloc(dig_e, (tags + 1) * sizeof(size_t));
    tag_start[tags] = i;
    tag_len[tags] = l;
    dig_s[tags] = ds;
    dig_e[tags] = de;
    tags++;
    i += l;
  }
  if (!tags) {
    split_by_length(t, n, max_bytes, &chunks);
    free(t);
    return chunks;
  }

  /* Speaker segments: (speaker, "<|speaker:S|> piece"). */
  typedef struct {
    char speaker[32];
    char *text;
  } Segment;
  Segment *segs = NULL;
  size_t nsegs = 0;
  for (size_t s = 0; s <= tags; s++) {
    size_t body_start = s == 0 ? 0 : tag_start[s - 1] + tag_len[s - 1];
    size_t body_end = s < tags ? tag_start[s] : n;
    char speaker[32] = "0";
    if (s > 0) {
      size_t dl = dig_e[s - 1] - dig_s[s - 1];
      if (dl > sizeof(speaker) - 1) dl = sizeof(speaker) - 1;
      for (size_t k = 0; k < dl; k++) speaker[k] = (char)t[dig_s[s - 1] + k];
      speaker[dl] = 0;
    }
    if (blank32(t + body_start, body_end - body_start)) continue;
    StrList pieces = {0};
    split_by_length(t + body_start, body_end - body_start, max_bytes, &pieces);
    for (size_t p = 0; p < pieces.count; p++) {
      size_t sz = strlen(pieces.items[p]) + strlen(speaker) + 16;
      char *seg = malloc(sz);
      snprintf(seg, sz, "<|speaker:%s|> %s", speaker, pieces.items[p]);
      segs = realloc(segs, (nsegs + 1) * sizeof(Segment));
      strcpy(segs[nsegs].speaker, speaker);
      segs[nsegs].text = seg;
      nsegs++;
    }
    strlist_free(&pieces);
  }

  /* Regroup: a new chunk starts when a segment brings one speaker too many or
   * would exceed max_bytes, so A, B, A can share a chunk. */
  char speakers[DRAMA3_CHUNK_MAX_SPEAKERS + 64][32];
  int n_speakers = 0;
  size_t size = 0, batch_start = 0, batch_count = 0;
  for (size_t i = 0; i <= nsegs; i++) {
    bool done = i == nsegs;
    bool new_speaker = false;
    if (!done) {
      new_speaker = true;
      for (int k = 0; k < n_speakers; k++)
        if (!strcmp(speakers[k], segs[i].speaker)) new_speaker = false;
      bool too_many = new_speaker && n_speakers >= max_speakers;
      bool too_long = batch_count && size + strlen(segs[i].text) > max_bytes;
      if (!too_many && !too_long) goto add;
    } else if (!batch_count) {
      break;
    }
    {
      size_t total = 1;
      for (size_t k = batch_start; k < batch_start + batch_count; k++)
        total += strlen(segs[k].text) + 1;
      char *joined = malloc(total);
      joined[0] = 0;
      for (size_t k = batch_start; k < batch_start + batch_count; k++) {
        if (k > batch_start) strcat(joined, "\n");
        strcat(joined, segs[k].text);
      }
      strlist_push(&chunks, joined);
      if (done) break;
      batch_start = i;
      batch_count = 0;
      n_speakers = 0;
      size = 0;
      new_speaker = true;
    }
  add:
    batch_count++;
    if (new_speaker && n_speakers < (int)(sizeof(speakers) / sizeof(speakers[0])))
      strcpy(speakers[n_speakers++], segs[i].speaker);
    size += strlen(segs[i].text);
  }
  for (size_t i = 0; i < nsegs; i++) free(segs[i].text);
  free(segs);
  free(tag_start);
  free(tag_len);
  free(dig_s);
  free(dig_e);
  free(t);
  return chunks;
}
