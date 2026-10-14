/* Generates src/unicode_data.c from Unicode Character Database files.
 *
 * The tokenizer must reproduce Hugging Face tokenizers exactly:
 *   - its pre-tokenizer regex runs on Oniguruma 6.9.10, whose \p{L}, \p{M},
 *     \p{N} and \s (White_Space) come from Unicode 16.0;
 *   - its NFC normalizer is the unicode-normalization-alignments crate, whose
 *     tables are Unicode 9.0.
 *
 * usage: gen_unicode <ucd-16 dir> <ucd-9 dir> <output.c>
 * (tools/fetch_ucd.sh downloads both directories.)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CP 0x110000

/* CAT_DIGIT / CAT_NUMERIC: Python str.isdigit() / str.isnumeric() (UnicodeData
 * fields 7 / 8), used by the text chunker. */
enum { CAT_L = 1, CAT_M = 2, CAT_N = 4, CAT_WS = 8, CAT_DIGIT = 16, CAT_NUMERIC = 32 };

static uint8_t cat_flags[MAX_CP];
static uint8_t ccc9[MAX_CP];
static int32_t decomp_off[MAX_CP]; /* single-level canonical mapping, 9.0 */
static uint8_t decomp_len[MAX_CP];
static uint32_t decomp_pool[MAX_CP];
static size_t decomp_pool_used;
static uint8_t full_exclusion[MAX_CP];

static FILE *open_file(const char *dir, const char *name) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    exit(1);
  }
  return f;
}

/* Splits a UnicodeData line into its ';' separated fields (in place). */
static int split_fields(char *line, char **fields, int max) {
  int n = 0;
  fields[n++] = line;
  for (char *p = line; *p && n < max; p++) {
    if (*p == ';') {
      *p = 0;
      fields[n++] = p + 1;
    }
  }
  return n;
}

static uint8_t category_flag(const char *gc) {
  switch (gc[0]) {
    case 'L': return CAT_L;
    case 'M': return CAT_M;
    case 'N': return CAT_N;
    default: return 0;
  }
}

static void read_categories(const char *dir) {
  FILE *f = open_file(dir, "UnicodeData.txt");
  char line[1024];
  uint32_t range_start = 0;
  while (fgets(line, sizeof(line), f)) {
    char *fields[16];
    if (split_fields(line, fields, 16) < 9) continue;
    uint32_t cp = (uint32_t)strtoul(fields[0], NULL, 16);
    uint8_t flag = category_flag(fields[2]);
    if (fields[7][0]) flag |= CAT_DIGIT;
    if (fields[8][0]) flag |= CAT_NUMERIC;
    const char *name = fields[1];
    if (strstr(name, ", First>")) {
      range_start = cp;
      continue;
    }
    if (strstr(name, ", Last>")) {
      for (uint32_t c = range_start; c <= cp; c++) cat_flags[c] |= flag;
      continue;
    }
    cat_flags[cp] |= flag;
  }
  fclose(f);
}

static void read_white_space(const char *dir) {
  FILE *f = open_file(dir, "PropList.txt");
  char line[1024];
  while (fgets(line, sizeof(line), f)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    if (!strstr(line, "; White_Space")) continue;
    uint32_t lo = (uint32_t)strtoul(line, NULL, 16), hi = lo;
    char *dots = strstr(line, "..");
    if (dots && dots < strchr(line, ';')) hi = (uint32_t)strtoul(dots + 2, NULL, 16);
    for (uint32_t c = lo; c <= hi; c++) cat_flags[c] |= CAT_WS;
  }
  fclose(f);
}

static void read_normalization(const char *dir) {
  memset(decomp_off, -1, sizeof(decomp_off));
  FILE *f = open_file(dir, "UnicodeData.txt");
  char line[1024];
  uint32_t range_start = 0;
  while (fgets(line, sizeof(line), f)) {
    char *fields[16];
    if (split_fields(line, fields, 16) < 6) continue;
    uint32_t cp = (uint32_t)strtoul(fields[0], NULL, 16);
    int ccc = atoi(fields[3]);
    if (strstr(fields[1], ", First>")) {
      range_start = cp;
      continue;
    }
    if (strstr(fields[1], ", Last>")) {
      for (uint32_t c = range_start; c <= cp; c++) ccc9[c] = (uint8_t)ccc;
      continue;
    }
    ccc9[cp] = (uint8_t)ccc;
    const char *d = fields[5];
    if (*d == 0 || *d == '<') continue; /* none, or compatibility mapping */
    decomp_off[cp] = (int32_t)decomp_pool_used;
    while (*d) {
      char *end;
      uint32_t v = (uint32_t)strtoul(d, &end, 16);
      if (end == d) break;
      decomp_pool[decomp_pool_used++] = v;
      decomp_len[cp]++;
      d = end;
      while (*d == ' ') d++;
    }
  }
  fclose(f);

  f = open_file(dir, "DerivedNormalizationProps.txt");
  while (fgets(line, sizeof(line), f)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    char *semi = strchr(line, ';');
    if (!semi) continue;
    char prop[128];
    if (sscanf(semi + 1, " %127[A-Za-z_]", prop) != 1) continue;
    if (strcmp(prop, "Full_Composition_Exclusion") != 0) continue;
    uint32_t lo = (uint32_t)strtoul(line, NULL, 16), hi = lo;
    char *dots = strstr(line, "..");
    if (dots && dots < semi) hi = (uint32_t)strtoul(dots + 2, NULL, 16);
    for (uint32_t c = lo; c <= hi; c++) full_exclusion[c] = 1;
  }
  fclose(f);
}

/* Appends the full canonical decomposition of cp (Hangul excluded). */
static void full_decompose(uint32_t cp, uint32_t *out, size_t *n) {
  if (decomp_off[cp] < 0) {
    out[(*n)++] = cp;
    return;
  }
  for (int i = 0; i < decomp_len[cp]; i++)
    full_decompose(decomp_pool[decomp_off[cp] + i], out, n);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <ucd-16 dir> <ucd-9 dir> <output.c>\n", argv[0]);
    return 2;
  }
  read_categories(argv[1]);
  read_white_space(argv[1]);
  read_normalization(argv[2]);

  FILE *out = fopen(argv[3], "w");
  if (!out) {
    fprintf(stderr, "cannot write %s\n", argv[3]);
    return 1;
  }
  fprintf(out,
          "/* Generated by tools/gen_unicode.c -- do not edit.\n"
          " * Derived from the Unicode Character Database, Copyright (c) Unicode, Inc.,\n"
          " * under the Unicode License v3 (https://www.unicode.org/license.txt).\n"
          " * Categories / White_Space: Unicode 16.0 (Oniguruma 6.9.10).\n"
          " * NFC: Unicode 9.0 (unicode-normalization-alignments 0.1.12). */\n"
          "#include \"unicode.h\"\n\n");

  /* Category ranges: runs of equal non-zero flags. */
  fprintf(out, "const uint32_t UNI_CAT_RANGES[][3] = {\n");
  size_t count = 0;
  for (uint32_t c = 0; c < MAX_CP;) {
    uint8_t flag = cat_flags[c];
    uint32_t start = c;
    while (c < MAX_CP && cat_flags[c] == flag) c++;
    if (flag) {
      fprintf(out, "  {0x%X, 0x%X, %u},\n", start, c - 1, flag);
      count++;
    }
  }
  fprintf(out, "};\nconst size_t UNI_CAT_RANGES_LEN = %zu;\n\n", count);

  fprintf(out, "const uint32_t UNI_CCC_RANGES[][3] = {\n");
  count = 0;
  for (uint32_t c = 0; c < MAX_CP;) {
    uint8_t v = ccc9[c];
    uint32_t start = c;
    while (c < MAX_CP && ccc9[c] == v) c++;
    if (v) {
      fprintf(out, "  {0x%X, 0x%X, %u},\n", start, c - 1, v);
      count++;
    }
  }
  fprintf(out, "};\nconst size_t UNI_CCC_RANGES_LEN = %zu;\n\n", count);

  /* Full canonical decompositions. */
  static uint32_t pool[MAX_CP];
  size_t pool_used = 0;
  fprintf(out, "const uint32_t UNI_DECOMP_INDEX[][3] = {\n");
  count = 0;
  for (uint32_t c = 0; c < MAX_CP; c++) {
    if (decomp_off[c] < 0) continue;
    size_t n = 0;
    uint32_t buf[64];
    full_decompose(c, buf, &n);
    fprintf(out, "  {0x%X, %zu, %zu},\n", c, pool_used, n);
    memcpy(pool + pool_used, buf, n * sizeof(uint32_t));
    pool_used += n;
    count++;
  }
  fprintf(out, "};\nconst size_t UNI_DECOMP_INDEX_LEN = %zu;\n\n", count);
  fprintf(out, "const uint32_t UNI_DECOMP_POOL[] = {\n");
  for (size_t i = 0; i < pool_used; i++)
    fprintf(out, "%s0x%X,%s", i % 8 ? " " : "  ", pool[i], i % 8 == 7 ? "\n" : "");
  fprintf(out, "\n};\n\n");

  /* Primary composites: two-code-point canonical mappings not excluded. */
  typedef struct {
    uint32_t a, b, c;
  } Pair;
  Pair *pairs = malloc(sizeof(Pair) * 4096);
  size_t npairs = 0;
  for (uint32_t c = 0; c < MAX_CP; c++) {
    if (decomp_off[c] < 0 || decomp_len[c] != 2 || full_exclusion[c]) continue;
    pairs[npairs].a = decomp_pool[decomp_off[c]];
    pairs[npairs].b = decomp_pool[decomp_off[c] + 1];
    pairs[npairs].c = c;
    npairs++;
  }
  /* sort by (a, b) */
  for (size_t i = 1; i < npairs; i++) {
    Pair key = pairs[i];
    size_t j = i;
    while (j > 0 && (pairs[j - 1].a > key.a ||
                     (pairs[j - 1].a == key.a && pairs[j - 1].b > key.b))) {
      pairs[j] = pairs[j - 1];
      j--;
    }
    pairs[j] = key;
  }
  fprintf(out, "const uint32_t UNI_COMPOSE[][3] = {\n");
  for (size_t i = 0; i < npairs; i++)
    fprintf(out, "  {0x%X, 0x%X, 0x%X},\n", pairs[i].a, pairs[i].b, pairs[i].c);
  fprintf(out, "};\nconst size_t UNI_COMPOSE_LEN = %zu;\n", npairs);
  fclose(out);
  free(pairs);
  return 0;
}
