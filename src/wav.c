#include "wav.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t le32(const unsigned char *p) {
  return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t le16(const unsigned char *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

bool wav_read(const char *path, Wav *out, char *err, size_t err_len) {
  memset(out, 0, sizeof(*out));
  FILE *f = fopen(path, "rb");
  if (!f) {
    snprintf(err, err_len, "cannot open %s", path);
    return false;
  }
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *buf = malloc((size_t)size);
  if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
    fclose(f);
    free(buf);
    snprintf(err, err_len, "cannot read %s", path);
    return false;
  }
  fclose(f);
  bool ok = false;
  if (size < 12 || memcmp(buf, "RIFF", 4) || memcmp(buf + 8, "WAVE", 4)) {
    snprintf(err, err_len, "%s is not a RIFF/WAVE file (only WAV input is supported)", path);
    goto done;
  }
  int format = 0, channels = 0, bits = 0, rate = 0;
  const unsigned char *data = NULL;
  size_t data_len = 0;
  for (size_t pos = 12; pos + 8 <= (size_t)size;) {
    uint32_t chunk = le32(buf + pos + 4);
    const unsigned char *body = buf + pos + 8;
    size_t avail = (size_t)size - pos - 8;
    if (chunk > avail) chunk = (uint32_t)avail; /* tolerate truncated data chunks */
    if (!memcmp(buf + pos, "fmt ", 4) && chunk >= 16) {
      format = le16(body);
      channels = le16(body + 2);
      rate = (int)le32(body + 4);
      bits = le16(body + 14);
      if (format == 0xFFFE && chunk >= 26) format = le16(body + 24); /* extensible */
    } else if (!memcmp(buf + pos, "data", 4)) {
      data = body;
      data_len = chunk;
    }
    pos += 8 + chunk + (chunk & 1);
  }
  if (!data || !channels || !rate) {
    snprintf(err, err_len, "%s: missing fmt or data chunk", path);
    goto done;
  }
  if (!((format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) ||
        (format == 3 && (bits == 32 || bits == 64)))) {
    snprintf(err, err_len, "%s: unsupported WAV encoding (format %d, %d bits)", path, format,
             bits);
    goto done;
  }
  size_t bytes = (size_t)bits / 8;
  size_t frames = data_len / (bytes * channels);
  float *s = malloc((frames * channels + 1) * sizeof(float));
  for (size_t i = 0; i < frames * channels; i++) {
    const unsigned char *p = data + i * bytes;
    float v;
    if (format == 3 && bits == 32) {
      memcpy(&v, p, 4);
    } else if (format == 3) {
      double d;
      memcpy(&d, p, 8);
      v = (float)d;
    } else if (bits == 8) {
      v = ((int)p[0] - 128) / 128.0f;
    } else if (bits == 16) {
      v = (int16_t)le16(p) / 32768.0f;
    } else if (bits == 24) {
      int32_t x = (int32_t)((uint32_t)p[0] << 8 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 24);
      v = (float)((double)(x >> 8) / 8388608.0);
    } else {
      v = (float)((double)(int32_t)le32(p) / 2147483648.0);
    }
    s[i] = v;
  }
  out->samples = s;
  out->frames = frames;
  out->channels = channels;
  out->sample_rate = rate;
  ok = true;
done:
  free(buf);
  return ok;
}

float *wav_mono(const Wav *w) {
  float *m = malloc((w->frames + 1) * sizeof(float));
  for (size_t i = 0; i < w->frames; i++) {
    float acc = 0.0f;
    for (int c = 0; c < w->channels; c++) acc += w->samples[i * w->channels + c];
    m[i] = acc / (float)w->channels;
  }
  return m;
}

void wav_free(Wav *w) {
  free(w->samples);
  w->samples = NULL;
}

static void put32(unsigned char *p, uint32_t v) {
  p[0] = v & 0xFF;
  p[1] = (v >> 8) & 0xFF;
  p[2] = (v >> 16) & 0xFF;
  p[3] = (v >> 24) & 0xFF;
}

static void put16(unsigned char *p, uint16_t v) {
  p[0] = v & 0xFF;
  p[1] = (v >> 8) & 0xFF;
}

bool wav_write(const char *path, const float *samples, size_t count, int sample_rate,
               int bits, char *err, size_t err_len) {
  if (bits != 16 && bits != 32) {
    snprintf(err, err_len, "unsupported output bit depth %d", bits);
    return false;
  }
  size_t bytes = (size_t)bits / 8;
  size_t data_len = count * bytes;
  unsigned char header[44];
  memcpy(header, "RIFF", 4);
  put32(header + 4, (uint32_t)(36 + data_len));
  memcpy(header + 8, "WAVEfmt ", 8);
  put32(header + 16, 16);
  put16(header + 20, bits == 32 ? 3 : 1);
  put16(header + 22, 1);
  put32(header + 24, (uint32_t)sample_rate);
  put32(header + 28, (uint32_t)(sample_rate * bytes));
  put16(header + 32, (uint16_t)bytes);
  put16(header + 34, (uint16_t)bits);
  memcpy(header + 36, "data", 4);
  put32(header + 40, (uint32_t)data_len);
  FILE *f = fopen(path, "wb");
  if (!f) {
    snprintf(err, err_len, "cannot write %s", path);
    return false;
  }
  fwrite(header, 1, sizeof(header), f);
  unsigned char *buf = malloc(data_len + 1);
  for (size_t i = 0; i < count; i++) {
    if (bits == 32) {
      memcpy(buf + i * 4, &samples[i], 4);
    } else {
      /* libsndfile float -> PCM_16: lrintf(x * 0x7FFF), clipped */
      float x = samples[i] * 32767.0f;
      long v = lrintf(x);
      if (v > 32767) v = 32767;
      if (v < -32768) v = -32768;
      put16(buf + i * 2, (uint16_t)(int16_t)v);
    }
  }
  bool ok = fwrite(buf, 1, data_len, f) == data_len;
  free(buf);
  fclose(f);
  if (!ok) snprintf(err, err_len, "short write to %s", path);
  return ok;
}
