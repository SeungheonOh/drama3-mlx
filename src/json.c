#include "json.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Bump allocator: a document's nodes and strings are freed together. */
typedef struct Block {
  struct Block *next;
  size_t used, size;
  char data[];
} Block;

struct JDoc {
  Block *blocks;
  JNode *root;
};

typedef struct {
  JDoc *doc;
  const char *begin, *p, *end;
  char *err;
  size_t err_len;
  bool failed;
} Parser;

static void *doc_alloc(JDoc *doc, size_t size) {
  size = (size + 15) & ~(size_t)15;
  Block *b = doc->blocks;
  if (!b || b->used + size > b->size) {
    size_t cap = size > (1 << 20) ? size : (1 << 20);
    Block *nb = malloc(sizeof(Block) + cap);
    if (!nb) return NULL;
    nb->next = doc->blocks;
    nb->used = 0;
    nb->size = cap;
    doc->blocks = nb;
    b = nb;
  }
  void *ptr = b->data + b->used;
  b->used += size;
  return ptr;
}

static void fail(Parser *ps, const char *msg) {
  if (ps->failed) return;
  ps->failed = true;
  if (ps->err && ps->err_len)
    snprintf(ps->err, ps->err_len, "%s at byte %zu", msg, (size_t)(ps->p - ps->begin));
}

static void skip_ws(Parser *ps) {
  while (ps->p < ps->end &&
         (*ps->p == ' ' || *ps->p == '\n' || *ps->p == '\r' || *ps->p == '\t'))
    ps->p++;
}

static int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static bool read_hex4(Parser *ps, uint32_t *out) {
  if (ps->end - ps->p < 4) return false;
  uint32_t v = 0;
  for (int i = 0; i < 4; i++) {
    int d = hex_digit(ps->p[i]);
    if (d < 0) return false;
    v = v * 16 + (uint32_t)d;
  }
  ps->p += 4;
  *out = v;
  return true;
}

static size_t put_utf8(char *out, uint32_t cp) {
  if (cp < 0x80) {
    out[0] = (char)cp;
    return 1;
  }
  if (cp < 0x800) {
    out[0] = (char)(0xC0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
  }
  if (cp < 0x10000) {
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
  }
  out[0] = (char)(0xF0 | (cp >> 18));
  out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
  out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
  out[3] = (char)(0x80 | (cp & 0x3F));
  return 4;
}

/* Parses a string starting at the opening quote. */
static const char *parse_string(Parser *ps, size_t *len_out) {
  ps->p++; /* opening quote */
  const char *start = ps->p;
  bool escaped = false;
  while (ps->p < ps->end && *ps->p != '"') {
    if (*ps->p == '\\') {
      escaped = true;
      ps->p++;
    }
    ps->p++;
  }
  if (ps->p >= ps->end) {
    fail(ps, "unterminated string");
    return NULL;
  }
  size_t span = (size_t)(ps->p - start);
  char *out = doc_alloc(ps->doc, span + 1);
  if (!escaped) {
    memcpy(out, start, span);
    out[span] = 0;
    ps->p++;
    *len_out = span;
    return out;
  }
  const char *q = start;
  size_t n = 0;
  while (q < ps->p) {
    if (*q != '\\') {
      out[n++] = *q++;
      continue;
    }
    q++;
    char c = *q++;
    switch (c) {
      case '"': out[n++] = '"'; break;
      case '\\': out[n++] = '\\'; break;
      case '/': out[n++] = '/'; break;
      case 'b': out[n++] = '\b'; break;
      case 'f': out[n++] = '\f'; break;
      case 'n': out[n++] = '\n'; break;
      case 'r': out[n++] = '\r'; break;
      case 't': out[n++] = '\t'; break;
      case 'u': {
        Parser sub = *ps;
        sub.p = q;
        uint32_t cp;
        if (!read_hex4(&sub, &cp)) {
          fail(ps, "bad \\u escape");
          return NULL;
        }
        q = sub.p;
        if (cp >= 0xD800 && cp <= 0xDBFF && q + 6 <= ps->p && q[0] == '\\' &&
            q[1] == 'u') {
          sub.p = q + 2;
          uint32_t lo;
          if (read_hex4(&sub, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            q = sub.p;
          }
        }
        n += put_utf8(out + n, cp);
        break;
      }
      default:
        fail(ps, "bad escape");
        return NULL;
    }
  }
  out[n] = 0;
  ps->p++; /* closing quote */
  *len_out = n;
  return out;
}

typedef struct {
  JNode **items;
  const char **keys;
  size_t count, cap;
} Children;

static void children_push(Children *c, const char *key, JNode *node) {
  if (c->count == c->cap) {
    c->cap = c->cap ? c->cap * 2 : 8;
    c->items = realloc(c->items, c->cap * sizeof(JNode *));
    c->keys = realloc(c->keys, c->cap * sizeof(char *));
  }
  c->items[c->count] = node;
  c->keys[c->count] = key;
  c->count++;
}

static JNode *parse_value(Parser *ps, int depth);

static JNode *finish_children(Parser *ps, JNode *node, Children *c, bool object) {
  node->count = c->count;
  node->items = doc_alloc(ps->doc, (c->count ? c->count : 1) * sizeof(JNode *));
  memcpy(node->items, c->items, c->count * sizeof(JNode *));
  if (object) {
    node->keys = doc_alloc(ps->doc, (c->count ? c->count : 1) * sizeof(char *));
    memcpy(node->keys, c->keys, c->count * sizeof(char *));
  }
  free(c->items);
  free(c->keys);
  return node;
}

static JNode *parse_value(Parser *ps, int depth) {
  if (depth > 512) {
    fail(ps, "nesting too deep");
    return NULL;
  }
  skip_ws(ps);
  if (ps->p >= ps->end) {
    fail(ps, "unexpected end");
    return NULL;
  }
  JNode *node = doc_alloc(ps->doc, sizeof(JNode));
  memset(node, 0, sizeof(*node));
  char c = *ps->p;
  if (c == '{' || c == '[') {
    bool object = c == '{';
    node->type = object ? J_OBJ : J_ARR;
    ps->p++;
    Children ch = {0};
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == (object ? '}' : ']')) {
      ps->p++;
      return finish_children(ps, node, &ch, object);
    }
    for (;;) {
      const char *key = NULL;
      if (object) {
        skip_ws(ps);
        if (ps->p >= ps->end || *ps->p != '"') {
          fail(ps, "expected key");
          free(ch.items);
          free(ch.keys);
          return NULL;
        }
        size_t klen;
        key = parse_string(ps, &klen);
        skip_ws(ps);
        if (!key || ps->p >= ps->end || *ps->p != ':') {
          fail(ps, "expected ':'");
          free(ch.items);
          free(ch.keys);
          return NULL;
        }
        ps->p++;
      }
      JNode *child = parse_value(ps, depth + 1);
      if (!child) {
        free(ch.items);
        free(ch.keys);
        return NULL;
      }
      children_push(&ch, key, child);
      skip_ws(ps);
      if (ps->p < ps->end && *ps->p == ',') {
        ps->p++;
        continue;
      }
      if (ps->p < ps->end && *ps->p == (object ? '}' : ']')) {
        ps->p++;
        return finish_children(ps, node, &ch, object);
      }
      fail(ps, "expected ',' or close");
      free(ch.items);
      free(ch.keys);
      return NULL;
    }
  }
  if (c == '"') {
    node->type = J_STR;
    node->str = parse_string(ps, &node->len);
    return node->str ? node : NULL;
  }
  if (c == 't' && ps->end - ps->p >= 4 && !memcmp(ps->p, "true", 4)) {
    ps->p += 4;
    node->type = J_BOOL;
    node->boolean = true;
    return node;
  }
  if (c == 'f' && ps->end - ps->p >= 5 && !memcmp(ps->p, "false", 5)) {
    ps->p += 5;
    node->type = J_BOOL;
    return node;
  }
  if (c == 'n' && ps->end - ps->p >= 4 && !memcmp(ps->p, "null", 4)) {
    ps->p += 4;
    node->type = J_NULL;
    return node;
  }
  if (c == '-' || (c >= '0' && c <= '9')) {
    char buf[64];
    size_t n = 0;
    while (ps->p < ps->end && n < sizeof(buf) - 1 &&
           (strchr("+-.eE", *ps->p) || (*ps->p >= '0' && *ps->p <= '9')))
      buf[n++] = *ps->p++;
    buf[n] = 0;
    node->type = J_NUM;
    node->number = strtod(buf, NULL);
    return node;
  }
  fail(ps, "unexpected character");
  return NULL;
}

JDoc *json_parse(const char *text, size_t len, char *err, size_t err_len) {
  JDoc *doc = calloc(1, sizeof(JDoc));
  Parser ps = {doc, text, text, text + len, err, err_len, false};
  doc->root = parse_value(&ps, 0);
  if (!doc->root || ps.failed) {
    json_free(doc);
    return NULL;
  }
  return doc;
}

JDoc *json_parse_file(const char *path, char *err, size_t err_len) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    if (err) snprintf(err, err_len, "cannot open %s", path);
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *text = malloc((size_t)size + 1);
  if (fread(text, 1, (size_t)size, f) != (size_t)size) {
    fclose(f);
    free(text);
    if (err) snprintf(err, err_len, "cannot read %s", path);
    return NULL;
  }
  fclose(f);
  JDoc *doc = json_parse(text, (size_t)size, err, err_len);
  free(text);
  return doc;
}

JNode *json_root(JDoc *doc) { return doc->root; }

void json_free(JDoc *doc) {
  if (!doc) return;
  Block *b = doc->blocks;
  while (b) {
    Block *next = b->next;
    free(b);
    b = next;
  }
  free(doc);
}

JNode *json_get(const JNode *obj, const char *key) {
  if (!obj || obj->type != J_OBJ) return NULL;
  for (size_t i = 0; i < obj->count; i++)
    if (!strcmp(obj->keys[i], key)) return obj->items[i];
  return NULL;
}

double json_num(const JNode *obj, const char *key, double fallback) {
  JNode *n = json_get(obj, key);
  return n && n->type == J_NUM ? n->number : fallback;
}

const char *json_str(const JNode *obj, const char *key) {
  JNode *n = json_get(obj, key);
  return n && n->type == J_STR ? n->str : NULL;
}

bool json_bool(const JNode *obj, const char *key, bool fallback) {
  JNode *n = json_get(obj, key);
  return n && n->type == J_BOOL ? n->boolean : fallback;
}
