#include "tokenizer.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "unicode.h"

/* ---- string -> id hash map ---- */

typedef struct {
  char *key; /* NULL = empty slot */
  uint32_t len;
  int32_t value;
} StrSlot;

typedef struct {
  StrSlot *slots;
  size_t cap, count;
} StrMap;

static uint64_t hash_bytes(const char *s, size_t len) {
  uint64_t h = 1469598103934665603ULL; /* FNV-1a */
  for (size_t i = 0; i < len; i++) {
    h ^= (unsigned char)s[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static void strmap_init(StrMap *m, size_t expected) {
  m->cap = 16;
  while (m->cap < expected * 2) m->cap *= 2;
  m->slots = calloc(m->cap, sizeof(StrSlot));
  m->count = 0;
}

static void strmap_put(StrMap *m, const char *key, size_t len, int32_t value) {
  size_t i = hash_bytes(key, len) & (m->cap - 1);
  while (m->slots[i].key) {
    if (m->slots[i].len == len && !memcmp(m->slots[i].key, key, len)) {
      m->slots[i].value = value;
      return;
    }
    i = (i + 1) & (m->cap - 1);
  }
  char *copy = malloc(len + 1);
  memcpy(copy, key, len);
  copy[len] = 0;
  m->slots[i] = (StrSlot){copy, (uint32_t)len, value};
  m->count++;
}

static int32_t strmap_get(const StrMap *m, const char *key, size_t len) {
  size_t i = hash_bytes(key, len) & (m->cap - 1);
  while (m->slots[i].key) {
    if (m->slots[i].len == len && !memcmp(m->slots[i].key, key, len))
      return m->slots[i].value;
    i = (i + 1) & (m->cap - 1);
  }
  return -1;
}

static void strmap_free(StrMap *m) {
  for (size_t i = 0; i < m->cap; i++) free(m->slots[i].key);
  free(m->slots);
}

/* ---- merge (pair of ids) -> (rank, new id) hash map ---- */

typedef struct {
  uint64_t key; /* (left << 32 | right) + 1; 0 = empty */
  int32_t rank, id;
} MergeSlot;

typedef struct {
  MergeSlot *slots;
  size_t cap;
} MergeMap;

static uint64_t mix64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

static void mergemap_put(MergeMap *m, int32_t a, int32_t b, int32_t rank, int32_t id) {
  uint64_t key = (((uint64_t)(uint32_t)a << 32) | (uint32_t)b) + 1;
  size_t i = mix64(key) & (m->cap - 1);
  while (m->slots[i].key && m->slots[i].key != key) i = (i + 1) & (m->cap - 1);
  if (m->slots[i].key == key) return; /* first (lowest) rank wins */
  m->slots[i] = (MergeSlot){key, rank, id};
}

static const MergeSlot *mergemap_get(const MergeMap *m, int32_t a, int32_t b) {
  uint64_t key = (((uint64_t)(uint32_t)a << 32) | (uint32_t)b) + 1;
  size_t i = mix64(key) & (m->cap - 1);
  while (m->slots[i].key) {
    if (m->slots[i].key == key) return &m->slots[i];
    i = (i + 1) & (m->cap - 1);
  }
  return NULL;
}

/* ---- added-token trie (bytes) ---- */

typedef struct {
  int32_t first_child, next_sibling;
  int32_t token_id; /* -1 if no token ends here */
  unsigned char byte;
} TrieNode;

typedef struct {
  TrieNode *nodes;
  size_t count, cap;
} Trie;

static int32_t trie_new_node(Trie *t, unsigned char byte) {
  if (t->count == t->cap) {
    t->cap = t->cap ? t->cap * 2 : 1024;
    t->nodes = realloc(t->nodes, t->cap * sizeof(TrieNode));
  }
  t->nodes[t->count] = (TrieNode){-1, -1, -1, byte};
  return (int32_t)t->count++;
}

static void trie_insert(Trie *t, const char *s, size_t len, int32_t id) {
  int32_t node = 0;
  for (size_t i = 0; i < len; i++) {
    unsigned char c = (unsigned char)s[i];
    int32_t child = t->nodes[node].first_child;
    while (child >= 0 && t->nodes[child].byte != c) child = t->nodes[child].next_sibling;
    if (child < 0) {
      child = trie_new_node(t, c);
      t->nodes[child].next_sibling = t->nodes[node].first_child;
      t->nodes[node].first_child = child;
    }
    node = child;
  }
  t->nodes[node].token_id = id;
}

/* Longest added token starting at s; returns its byte length (0 if none). */
static size_t trie_longest(const Trie *t, const char *s, size_t len, int32_t *id) {
  int32_t node = 0;
  size_t best = 0;
  for (size_t i = 0; i < len; i++) {
    unsigned char c = (unsigned char)s[i];
    int32_t child = t->nodes[node].first_child;
    while (child >= 0 && t->nodes[child].byte != c) child = t->nodes[child].next_sibling;
    if (child < 0) break;
    node = child;
    if (t->nodes[node].token_id >= 0) {
      best = i + 1;
      *id = t->nodes[node].token_id;
    }
  }
  return best;
}

struct Tokenizer {
  StrMap vocab;   /* byte-level token string -> id */
  StrMap added;   /* added token content -> id */
  MergeMap merges;
  Trie trie;
  int32_t byte_ids[256]; /* id of each single byte-level symbol */
  uint32_t byte_char[256];
};

/* GPT-2 bytes_to_unicode. */
static void build_byte_chars(uint32_t *table) {
  int n = 0;
  for (int b = 0; b < 256; b++) {
    bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
    table[b] = printable ? (uint32_t)b : (uint32_t)(256 + n++);
  }
}

static void die_tok(const char *msg) {
  fprintf(stderr, "tokenizer: %s\n", msg);
  exit(1);
}

Tokenizer *tok_load(const char *path) {
  char err[256];
  JDoc *doc = json_parse_file(path, err, sizeof(err));
  if (!doc) {
    fprintf(stderr, "tokenizer: %s\n", err);
    return NULL;
  }
  JNode *root = json_root(doc);
  JNode *model = json_get(root, "model");
  const char *type = json_str(model, "type");
  if (!type || strcmp(type, "BPE")) die_tok("model.type must be BPE");
  JNode *vocab = json_get(model, "vocab");
  JNode *merges = json_get(model, "merges");
  JNode *added = json_get(root, "added_tokens");
  if (!vocab || vocab->type != J_OBJ || !merges || merges->type != J_ARR)
    die_tok("missing vocab / merges");
  JNode *norm = json_get(root, "normalizer");
  if (!norm || !json_str(norm, "type") || strcmp(json_str(norm, "type"), "NFC"))
    die_tok("expected an NFC normalizer");

  Tokenizer *tok = calloc(1, sizeof(Tokenizer));
  strmap_init(&tok->vocab, vocab->count);
  for (size_t i = 0; i < vocab->count; i++) {
    const char *key = vocab->keys[i];
    strmap_put(&tok->vocab, key, strlen(key), (int32_t)vocab->items[i]->number);
  }

  tok->merges.cap = 16;
  while (tok->merges.cap < merges->count * 2) tok->merges.cap *= 2;
  tok->merges.slots = calloc(tok->merges.cap, sizeof(MergeSlot));
  char buf[1024];
  for (size_t r = 0; r < merges->count; r++) {
    JNode *m = merges->items[r];
    const char *left, *right;
    if (m->type == J_ARR && m->count == 2) {
      left = m->items[0]->str;
      right = m->items[1]->str;
    } else if (m->type == J_STR) {
      /* "left right" legacy form */
      const char *sp = strchr(m->str, ' ');
      if (!sp) die_tok("bad merge");
      size_t ll = (size_t)(sp - m->str);
      memcpy(buf, m->str, ll);
      buf[ll] = 0;
      left = buf;
      right = sp + 1;
    } else {
      die_tok("bad merge entry");
    }
    int32_t a = strmap_get(&tok->vocab, left, strlen(left));
    int32_t b = strmap_get(&tok->vocab, right, strlen(right));
    size_t ll = strlen(left), rl = strlen(right);
    char *joined = malloc(ll + rl + 1);
    memcpy(joined, left, ll);
    memcpy(joined + ll, right, rl + 1);
    int32_t id = strmap_get(&tok->vocab, joined, ll + rl);
    free(joined);
    if (a < 0 || b < 0 || id < 0) die_tok("merge refers to unknown tokens");
    mergemap_put(&tok->merges, a, b, (int32_t)r, id);
  }

  strmap_init(&tok->added, added ? added->count : 1);
  trie_new_node(&tok->trie, 0);
  for (size_t i = 0; added && i < added->count; i++) {
    JNode *t = added->items[i];
    const char *content = json_str(t, "content");
    int32_t id = (int32_t)json_num(t, "id", -1);
    if (!content || id < 0) die_tok("bad added token");
    if (json_bool(t, "normalized", false) || json_bool(t, "lstrip", false) ||
        json_bool(t, "rstrip", false) || json_bool(t, "single_word", false))
      die_tok("added tokens with normalized/lstrip/rstrip/single_word are unsupported");
    strmap_put(&tok->added, content, strlen(content), id);
    trie_insert(&tok->trie, content, strlen(content), id);
  }

  build_byte_chars(tok->byte_char);
  for (int b = 0; b < 256; b++) {
    char s[8];
    size_t len = utf8_encode(&tok->byte_char[b], 1, s);
    tok->byte_ids[b] = strmap_get(&tok->vocab, s, len);
    if (tok->byte_ids[b] < 0) die_tok("byte-level symbol missing from vocab");
  }
  json_free(doc);
  return tok;
}

void tok_free(Tokenizer *tok) {
  if (!tok) return;
  strmap_free(&tok->vocab);
  strmap_free(&tok->added);
  free(tok->merges.slots);
  free(tok->trie.nodes);
  free(tok);
}

int32_t tok_token_to_id(const Tokenizer *tok, const char *token) {
  int32_t id = strmap_get(&tok->added, token, strlen(token));
  if (id >= 0) return id;
  return strmap_get(&tok->vocab, token, strlen(token));
}

/* ---- pre-tokenizer: the Qwen3.5 split regex ----
 *
 * (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
 * | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 *
 * Each alternative is resolved here with Oniguruma's leftmost-first
 * backtracking semantics.
 */

static bool is_crlf(uint32_t c) { return c == '\r' || c == '\n'; }

static bool fold_eq(uint32_t c, char lower) {
  if (c == (uint32_t)lower || c == (uint32_t)(lower - 32)) return true;
  return lower == 's' && c == 0x017F; /* LATIN SMALL LETTER LONG S folds to s */
}

static size_t match_at(const uint32_t *cp, size_t n, size_t i) {
  unsigned f = uni_flags(cp[i]);
  /* 1: contractions */
  if (cp[i] == '\'' && i + 1 < n) {
    uint32_t c1 = cp[i + 1];
    if (fold_eq(c1, 's') || fold_eq(c1, 't') || fold_eq(c1, 'm') || fold_eq(c1, 'd'))
      return 2;
    if (i + 2 < n) {
      uint32_t c2 = cp[i + 2];
      if ((fold_eq(c1, 'r') && fold_eq(c2, 'e')) || (fold_eq(c1, 'v') && fold_eq(c2, 'e')) ||
          (fold_eq(c1, 'l') && fold_eq(c2, 'l')))
        return 3;
    }
  }
  /* 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ */
  {
    size_t start = i;
    bool optional = !is_crlf(cp[i]) && !(f & (UNI_L | UNI_N));
    if (optional && i + 1 < n && (uni_flags(cp[i + 1]) & (UNI_L | UNI_M))) start = i + 1;
    if (start != i || (f & (UNI_L | UNI_M))) {
      size_t j = start;
      while (j < n && (uni_flags(cp[j]) & (UNI_L | UNI_M))) j++;
      return j - i;
    }
  }
  /* 3: \p{N} */
  if (f & UNI_N) return 1;
  /* 4:  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* */
  {
    const unsigned other = UNI_WS | UNI_L | UNI_M | UNI_N;
    size_t start = i;
    if (cp[i] == ' ' && i + 1 < n && !(uni_flags(cp[i + 1]) & other)) start = i + 1;
    if (!(uni_flags(cp[start]) & other)) {
      size_t j = start;
      while (j < n && !(uni_flags(cp[j]) & other)) j++;
      while (j < n && is_crlf(cp[j])) j++;
      return j - i;
    }
  }
  if (f & UNI_WS) {
    size_t j = i;
    while (j < n && (uni_flags(cp[j]) & UNI_WS)) j++;
    /* 5: \s*[\r\n]+ -- ends after the last newline of the run */
    for (size_t k = j; k > i; k--)
      if (is_crlf(cp[k - 1])) return k - i;
    /* 6: \s+(?!\S) */
    if (j == n) return j - i;
    if (j - i >= 2) return j - 1 - i;
    /* 7: \s+ */
    return j - i;
  }
  return 1; /* unreachable: every code point matches one alternative */
}

static size_t pretokenize_cps(const uint32_t *cp, size_t n, size_t **offsets) {
  size_t cap = 16, count = 0;
  size_t *out = malloc(cap * 2 * sizeof(size_t));
  for (size_t i = 0; i < n;) {
    size_t len = match_at(cp, n, i);
    if (count == cap) {
      cap *= 2;
      out = realloc(out, cap * 2 * sizeof(size_t));
    }
    out[2 * count] = i;
    out[2 * count + 1] = i + len;
    count++;
    i += len;
  }
  *offsets = out;
  return count;
}

size_t tok_pretokenize(const char *text, size_t len, size_t **offsets) {
  uint32_t *cps = malloc((len + 1) * sizeof(uint32_t));
  size_t n = utf8_decode(text, len, cps);
  if (n == SIZE_MAX) {
    free(cps);
    *offsets = NULL;
    return 0;
  }
  size_t count = pretokenize_cps(cps, n, offsets);
  free(cps);
  return count;
}

/* ---- BPE ---- */

typedef struct {
  int32_t id, prev, next;
  bool removed;
} Symbol;

typedef struct {
  int32_t rank, pos, new_id;
} Merge;

static bool merge_before(const Merge *a, const Merge *b) {
  return a->rank != b->rank ? a->rank < b->rank : a->pos < b->pos;
}

typedef struct {
  Merge *v;
  size_t n, cap;
} Heap;

static void heap_push(Heap *h, Merge m) {
  if (h->n == h->cap) {
    h->cap = h->cap ? h->cap * 2 : 64;
    h->v = realloc(h->v, h->cap * sizeof(Merge));
  }
  size_t i = h->n++;
  h->v[i] = m;
  while (i > 0) {
    size_t parent = (i - 1) / 2;
    if (!merge_before(&h->v[i], &h->v[parent])) break;
    Merge t = h->v[i];
    h->v[i] = h->v[parent];
    h->v[parent] = t;
    i = parent;
  }
}

static Merge heap_pop(Heap *h) {
  Merge top = h->v[0];
  h->v[0] = h->v[--h->n];
  size_t i = 0;
  for (;;) {
    size_t l = 2 * i + 1, r = l + 1, best = i;
    if (l < h->n && merge_before(&h->v[l], &h->v[best])) best = l;
    if (r < h->n && merge_before(&h->v[r], &h->v[best])) best = r;
    if (best == i) break;
    Merge t = h->v[i];
    h->v[i] = h->v[best];
    h->v[best] = t;
    i = best;
  }
  return top;
}

typedef struct {
  int32_t *v;
  size_t n, cap;
} IdBuf;

static void ids_push(IdBuf *b, int32_t id) {
  if (b->n == b->cap) {
    b->cap = b->cap ? b->cap * 2 : 64;
    b->v = realloc(b->v, b->cap * sizeof(int32_t));
  }
  b->v[b->n++] = id;
}

/* BPE over the bytes of one pre-token, following tokenizers' Word::merge_all. */
static void bpe_word(const Tokenizer *tok, const unsigned char *bytes, size_t len,
                     IdBuf *out) {
  Symbol *sym = malloc(len * sizeof(Symbol));
  for (size_t i = 0; i < len; i++)
    sym[i] = (Symbol){tok->byte_ids[bytes[i]], (int32_t)i - 1,
                      i + 1 < len ? (int32_t)i + 1 : -1, false};
  Heap heap = {0};
  for (size_t i = 0; i + 1 < len; i++) {
    const MergeSlot *m = mergemap_get(&tok->merges, sym[i].id, sym[i + 1].id);
    if (m) heap_push(&heap, (Merge){m->rank, (int32_t)i, m->id});
  }
  while (heap.n) {
    Merge top = heap_pop(&heap);
    Symbol *cur = &sym[top.pos];
    if (cur->removed || cur->next < 0) continue;
    Symbol *right = &sym[cur->next];
    const MergeSlot *m = mergemap_get(&tok->merges, cur->id, right->id);
    if (!m || m->id != top.new_id) continue; /* stale entry */
    cur->id = top.new_id;
    right->removed = true;
    int32_t right_next = right->next;
    cur->next = right_next;
    if (right_next >= 0) sym[right_next].prev = top.pos;
    if (cur->prev >= 0) {
      const MergeSlot *pm = mergemap_get(&tok->merges, sym[cur->prev].id, cur->id);
      if (pm) heap_push(&heap, (Merge){pm->rank, cur->prev, pm->id});
    }
    if (cur->next >= 0) {
      const MergeSlot *nm = mergemap_get(&tok->merges, cur->id, sym[cur->next].id);
      if (nm) heap_push(&heap, (Merge){nm->rank, top.pos, nm->id});
    }
  }
  for (size_t i = 0; i < len; i++)
    if (!sym[i].removed) ids_push(out, sym[i].id);
  free(heap.v);
  free(sym);
}

static bool encode_plain(const Tokenizer *tok, const char *text, size_t len, IdBuf *out) {
  if (len == 0) return true;
  uint32_t *cps = malloc((len + 1) * sizeof(uint32_t));
  size_t n = utf8_decode(text, len, cps);
  if (n == SIZE_MAX) {
    free(cps);
    return false;
  }
  size_t nn;
  uint32_t *norm = uni_nfc(cps, n, &nn);
  free(cps);
  size_t *offsets;
  size_t pieces = pretokenize_cps(norm, nn, &offsets);
  char *bytes = malloc(nn * 4 + 1);
  for (size_t p = 0; p < pieces; p++) {
    size_t blen = utf8_encode(norm + offsets[2 * p], offsets[2 * p + 1] - offsets[2 * p], bytes);
    bpe_word(tok, (const unsigned char *)bytes, blen, out);
  }
  free(bytes);
  free(offsets);
  free(norm);
  return true;
}

int32_t *tok_encode(const Tokenizer *tok, const char *text, size_t len, size_t *n) {
  IdBuf out = {0};
  size_t seg_start = 0, i = 0;
  while (i < len) {
    int32_t id;
    size_t match = trie_longest(&tok->trie, text + i, len - i, &id);
    if (!match) {
      i++;
      while (i < len && ((unsigned char)text[i] & 0xC0) == 0x80) i++;
      continue;
    }
    if (!encode_plain(tok, text + seg_start, i - seg_start, &out)) {
      free(out.v);
      return NULL;
    }
    ids_push(&out, id);
    i += match;
    seg_start = i;
  }
  if (!encode_plain(tok, text + seg_start, len - seg_start, &out)) {
    free(out.v);
    return NULL;
  }
  *n = out.n;
  if (!out.v) out.v = malloc(sizeof(int32_t));
  return out.v;
}
