/* Minimal JSON DOM parser (RFC 8259) for config.json and tokenizer.json. */
#ifndef DRAMA3_JSON_H
#define DRAMA3_JSON_H

#include <stdbool.h>
#include <stddef.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } JType;

typedef struct JNode JNode;
struct JNode {
  JType type;
  bool boolean;
  double number;
  const char *str; /* J_STR: unescaped UTF-8, NUL terminated */
  size_t len;      /* J_STR: byte length */
  size_t count;    /* J_ARR / J_OBJ: number of children */
  JNode **items;   /* children */
  const char **keys; /* J_OBJ: keys, parallel to items */
};

typedef struct JDoc JDoc;

/* Parses text; returns NULL and fills err (if given) on failure. */
JDoc *json_parse(const char *text, size_t len, char *err, size_t err_len);
JDoc *json_parse_file(const char *path, char *err, size_t err_len);
JNode *json_root(JDoc *doc);
void json_free(JDoc *doc);

JNode *json_get(const JNode *obj, const char *key);
double json_num(const JNode *obj, const char *key, double fallback);
const char *json_str(const JNode *obj, const char *key);
bool json_bool(const JNode *obj, const char *key, bool fallback);

#endif
