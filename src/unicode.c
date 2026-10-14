#include "unicode.h"

#include <stdlib.h>
#include <string.h>

static unsigned range_lookup(const uint32_t (*ranges)[3], size_t len, uint32_t cp) {
  size_t lo = 0, hi = len;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < ranges[mid][0])
      hi = mid;
    else if (cp > ranges[mid][1])
      lo = mid + 1;
    else
      return ranges[mid][2];
  }
  return 0;
}

unsigned uni_flags(uint32_t cp) {
  if (cp < 0x80) {
    /* ASCII fast path */
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return UNI_L;
    if (cp >= '0' && cp <= '9') return UNI_N | UNI_DIGIT | UNI_NUMERIC;
    if (cp == ' ' || (cp >= 0x09 && cp <= 0x0D)) return UNI_WS;
    return 0;
  }
  return range_lookup(UNI_CAT_RANGES, UNI_CAT_RANGES_LEN, cp);
}

unsigned uni_ccc(uint32_t cp) {
  if (cp < 0x300) return 0;
  return range_lookup(UNI_CCC_RANGES, UNI_CCC_RANGES_LEN, cp);
}

size_t utf8_decode(const char *str, size_t len, uint32_t *out) {
  static const uint32_t min_value[4] = {0, 0x80, 0x800, 0x10000};
  const unsigned char *s = (const unsigned char *)str;
  size_t i = 0, n = 0;
  while (i < len) {
    unsigned c = s[i];
    uint32_t cp;
    size_t need;
    if (c < 0x80) {
      out[n++] = c;
      i++;
      continue;
    } else if ((c & 0xE0) == 0xC0) {
      cp = c & 0x1F;
      need = 1;
    } else if ((c & 0xF0) == 0xE0) {
      cp = c & 0x0F;
      need = 2;
    } else if ((c & 0xF8) == 0xF0) {
      cp = c & 0x07;
      need = 3;
    } else {
      return SIZE_MAX;
    }
    if (i + need >= len) return SIZE_MAX; /* truncated sequence */
    for (size_t k = 1; k <= need; k++) {
      if ((s[i + k] & 0xC0) != 0x80) return SIZE_MAX;
      cp = (cp << 6) | (s[i + k] & 0x3F);
    }
    if (cp < min_value[need] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
      return SIZE_MAX;
    out[n++] = cp;
    i += need + 1;
  }
  return n;
}

uint32_t utf8_next(const char *str, size_t len, size_t *i) {
  const unsigned char *s = (const unsigned char *)str;
  unsigned c = s[*i];
  size_t need = c < 0x80 ? 0 : (c & 0xE0) == 0xC0 ? 1 : (c & 0xF0) == 0xE0 ? 2 : 3;
  uint32_t cp = need == 0 ? c : need == 1 ? (c & 0x1F) : need == 2 ? (c & 0x0F) : (c & 0x07);
  (*i)++;
  for (size_t k = 0; k < need && *i < len; k++, (*i)++) cp = (cp << 6) | (s[*i] & 0x3F);
  return cp;
}

size_t utf8_encode(const uint32_t *cps, size_t count, char *out) {
  size_t n = 0;
  for (size_t i = 0; i < count; i++) {
    uint32_t cp = cps[i];
    if (cp < 0x80) {
      out[n++] = (char)cp;
    } else if (cp < 0x800) {
      out[n++] = (char)(0xC0 | (cp >> 6));
      out[n++] = (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out[n++] = (char)(0xE0 | (cp >> 12));
      out[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
      out[n++] = (char)(0x80 | (cp & 0x3F));
    } else {
      out[n++] = (char)(0xF0 | (cp >> 18));
      out[n++] = (char)(0x80 | ((cp >> 12) & 0x3F));
      out[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
      out[n++] = (char)(0x80 | (cp & 0x3F));
    }
  }
  return n;
}

/* ---- NFC ---- */

enum {
  S_BASE = 0xAC00,
  L_BASE = 0x1100,
  V_BASE = 0x1161,
  T_BASE = 0x11A7,
  L_COUNT = 19,
  V_COUNT = 21,
  T_COUNT = 28,
  N_COUNT = V_COUNT * T_COUNT,
  S_COUNT = L_COUNT * N_COUNT,
};

typedef struct {
  uint32_t *v;
  size_t n, cap;
} Buf;

static void buf_push(Buf *b, uint32_t cp) {
  if (b->n == b->cap) {
    b->cap = b->cap ? b->cap * 2 : 64;
    b->v = realloc(b->v, b->cap * sizeof(uint32_t));
  }
  b->v[b->n++] = cp;
}

static const uint32_t *decomposition(uint32_t cp, size_t *len) {
  size_t lo = 0, hi = UNI_DECOMP_INDEX_LEN;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    uint32_t key = UNI_DECOMP_INDEX[mid][0];
    if (cp < key)
      hi = mid;
    else if (cp > key)
      lo = mid + 1;
    else {
      *len = UNI_DECOMP_INDEX[mid][2];
      return UNI_DECOMP_POOL + UNI_DECOMP_INDEX[mid][1];
    }
  }
  return NULL;
}

static void decompose_one(uint32_t cp, Buf *out) {
  if (cp >= S_BASE && cp < S_BASE + S_COUNT) {
    uint32_t s = cp - S_BASE;
    buf_push(out, L_BASE + s / N_COUNT);
    buf_push(out, V_BASE + (s % N_COUNT) / T_COUNT);
    if (s % T_COUNT) buf_push(out, T_BASE + s % T_COUNT);
    return;
  }
  size_t len;
  const uint32_t *d = cp >= 0xC0 ? decomposition(cp, &len) : NULL;
  if (!d) {
    buf_push(out, cp);
    return;
  }
  for (size_t i = 0; i < len; i++) buf_push(out, d[i]);
}

static uint32_t compose_pair(uint32_t a, uint32_t b) {
  if (a >= L_BASE && a < L_BASE + L_COUNT && b >= V_BASE && b < V_BASE + V_COUNT)
    return S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT;
  if (a >= S_BASE && a < S_BASE + S_COUNT && (a - S_BASE) % T_COUNT == 0 &&
      b > T_BASE && b < T_BASE + T_COUNT)
    return a + (b - T_BASE);
  size_t lo = 0, hi = UNI_COMPOSE_LEN;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    const uint32_t *e = UNI_COMPOSE[mid];
    if (a < e[0] || (a == e[0] && b < e[1]))
      hi = mid;
    else if (a > e[0] || (a == e[0] && b > e[1]))
      lo = mid + 1;
    else
      return e[2];
  }
  return 0;
}

uint32_t *uni_nfc(const uint32_t *cps, size_t count, size_t *out_count) {
  Buf d = {0};
  for (size_t i = 0; i < count; i++) decompose_one(cps[i], &d);

  /* Canonical ordering: stable insertion sort of each run of non-starters. */
  for (size_t i = 1; i < d.n; i++) {
    unsigned c = uni_ccc(d.v[i]);
    if (c == 0) continue;
    size_t j = i;
    uint32_t cp = d.v[i];
    while (j > 0) {
      unsigned prev = uni_ccc(d.v[j - 1]);
      if (prev == 0 || prev <= c) break;
      d.v[j] = d.v[j - 1];
      j--;
    }
    d.v[j] = cp;
  }

  /* Canonical composition. */
  if (d.n == 0) {
    *out_count = 0;
    return d.v ? d.v : malloc(sizeof(uint32_t));
  }
  size_t starter = SIZE_MAX;
  int last_ccc = -1;
  size_t w = 0;
  for (size_t i = 0; i < d.n; i++) {
    uint32_t cp = d.v[i];
    int c = (int)uni_ccc(cp);
    if (starter != SIZE_MAX) {
      bool blocked = last_ccc != -1 && (last_ccc == 0 || last_ccc >= c);
      if (!blocked) {
        uint32_t composed = compose_pair(d.v[starter], cp);
        if (composed) {
          d.v[starter] = composed;
          continue;
        }
      }
    }
    if (c == 0) {
      starter = w;
      last_ccc = -1;
    } else {
      last_ccc = c;
    }
    d.v[w++] = cp;
  }
  *out_count = w;
  return d.v;
}
