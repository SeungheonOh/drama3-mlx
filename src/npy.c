#include "npy.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool npy_read_codes(const char *path, Codes *out, char *err, size_t err_len) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    snprintf(err, err_len, "cannot open %s", path);
    return false;
  }
  unsigned char magic[10];
  if (fread(magic, 1, 10, f) != 10 || memcmp(magic, "\x93NUMPY", 6)) {
    fclose(f);
    snprintf(err, err_len, "%s is not a .npy file", path);
    return false;
  }
  size_t header_len;
  if (magic[6] == 1) {
    header_len = magic[8] | (magic[9] << 8);
  } else {
    unsigned char extra[2];
    if (fread(extra, 1, 2, f) != 2) {
      fclose(f);
      snprintf(err, err_len, "%s: truncated header", path);
      return false;
    }
    header_len = magic[8] | (magic[9] << 8) | (extra[0] << 16) | ((size_t)extra[1] << 24);
  }
  char *header = calloc(header_len + 1, 1);
  if (fread(header, 1, header_len, f) != header_len) {
    free(header);
    fclose(f);
    snprintf(err, err_len, "%s: truncated header", path);
    return false;
  }
  int bytes = 0;
  bool is_signed = true;
  const char *descr = strstr(header, "'descr'");
  if (descr) {
    const char *q = strchr(descr + 7, '\'');
    if (q && (q[1] == '<' || q[1] == '|')) {
      is_signed = q[2] == 'i';
      if (q[2] == 'i' || q[2] == 'u') bytes = q[3] - '0';
    }
  }
  bool fortran = strstr(header, "'fortran_order': True") != NULL;
  long rows = -1, cols = -1;
  const char *shape = strstr(header, "'shape'");
  if (shape) {
    const char *p = strchr(shape, '(');
    if (p) sscanf(p, "(%ld, %ld)", &rows, &cols);
  }
  free(header);
  if (!(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8) || fortran || rows <= 0 ||
      cols <= 0) {
    fclose(f);
    snprintf(err, err_len, "%s: expected a C-order 2-D integer array", path);
    return false;
  }
  size_t count = (size_t)rows * cols;
  unsigned char *raw = malloc(count * bytes);
  if (fread(raw, bytes, count, f) != count) {
    free(raw);
    fclose(f);
    snprintf(err, err_len, "%s: truncated data", path);
    return false;
  }
  fclose(f);
  out->data = malloc(count * sizeof(int32_t));
  for (size_t i = 0; i < count; i++) {
    int64_t v = 0;
    const unsigned char *p = raw + i * bytes;
    if (bytes == 1) v = is_signed ? (int8_t)p[0] : p[0];
    if (bytes == 2) {
      uint16_t u = (uint16_t)(p[0] | (p[1] << 8));
      v = is_signed ? (int16_t)u : u;
    }
    if (bytes == 4) {
      uint32_t u = p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24);
      v = is_signed ? (int32_t)u : u;
    }
    if (bytes == 8) {
      uint64_t u = 0;
      for (int k = 7; k >= 0; k--) u = (u << 8) | p[k];
      v = (int64_t)u;
    }
    out->data[i] = (int32_t)v;
  }
  free(raw);
  out->n_codebooks = (int)rows;
  out->frames = (int)cols;
  return true;
}

bool npy_write_codes(const char *path, const Codes *codes, char *err, size_t err_len) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    snprintf(err, err_len, "cannot write %s", path);
    return false;
  }
  char header[128];
  int n = snprintf(header, sizeof(header),
                   "{'descr': '<i8', 'fortran_order': False, 'shape': (%d, %d), }",
                   codes->n_codebooks, codes->frames);
  int total = 10 + n + 1;
  int pad = (64 - total % 64) % 64;
  unsigned char pre[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0};
  int hlen = n + pad + 1;
  pre[8] = hlen & 0xFF;
  pre[9] = (hlen >> 8) & 0xFF;
  fwrite(pre, 1, 10, f);
  fwrite(header, 1, n, f);
  for (int i = 0; i < pad; i++) fputc(' ', f);
  fputc('\n', f);
  size_t count = (size_t)codes->n_codebooks * codes->frames;
  for (size_t i = 0; i < count; i++) {
    int64_t v = codes->data[i];
    unsigned char b[8];
    for (int k = 0; k < 8; k++) b[k] = (unsigned char)((uint64_t)v >> (8 * k));
    fwrite(b, 1, 8, f);
  }
  bool ok = !ferror(f);
  fclose(f);
  if (!ok) snprintf(err, err_len, "short write to %s", path);
  return ok;
}
