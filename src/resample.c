/* Port of torchaudio.functional._get_sinc_resample_kernel /
 * _apply_sinc_resample_kernel for float32 input (kernel built in float32). */
#include "resample.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LOWPASS_FILTER_WIDTH 6
#define ROLLOFF 0.99

static long gcd(long a, long b) {
  while (b) {
    long t = a % b;
    a = b;
    b = t;
  }
  return a;
}

float *resample(const float *in, size_t count, int orig_rate, int new_rate, size_t *out_count) {
  if (orig_rate == new_rate) {
    float *out = malloc((count + 1) * sizeof(float));
    memcpy(out, in, count * sizeof(float));
    *out_count = count;
    return out;
  }
  long g = gcd(orig_rate, new_rate);
  int orig = (int)(orig_rate / g), nw = (int)(new_rate / g);
  double base_freq = (orig < nw ? orig : nw) * ROLLOFF;
  int width = (int)ceil(LOWPASS_FILTER_WIDTH * (double)orig / base_freq);
  int klen = 2 * width + orig;
  float *kernel = malloc((size_t)nw * klen * sizeof(float));
  const float base_f = (float)base_freq;
  const float pi_f = (float)M_PI;
  const float scale_f = (float)(base_freq / orig);
  for (int phase = 0; phase < nw; phase++) {
    float shift = (float)(-phase) / (float)nw;
    for (int j = 0; j < klen; j++) {
      float idx = (float)(j - width) / (float)orig;
      float t = (shift + idx) * base_f;
      if (t < -LOWPASS_FILTER_WIDTH) t = -LOWPASS_FILTER_WIDTH;
      if (t > LOWPASS_FILTER_WIDTH) t = LOWPASS_FILTER_WIDTH;
      float w = cosf(t * pi_f / (float)LOWPASS_FILTER_WIDTH / 2.0f);
      w = w * w;
      t *= pi_f;
      float k = t == 0.0f ? 1.0f : sinf(t) / t;
      kernel[(size_t)phase * klen + j] = k * (w * scale_f);
    }
  }
  /* pad (width, width + orig), conv with stride orig, interleave phases */
  size_t padded_len = count + 2 * (size_t)width + orig;
  float *padded = calloc(padded_len, sizeof(float));
  memcpy(padded + width, in, count * sizeof(float));
  size_t steps = (padded_len - klen) / orig + 1;
  size_t target = (size_t)ceil((double)nw * (double)count / (double)orig);
  float *out = malloc((target + 1) * sizeof(float));
  size_t n = 0;
  for (size_t s = 0; s < steps && n < target; s++) {
    const float *src = padded + s * orig;
    for (int phase = 0; phase < nw && n < target; phase++) {
      const float *k = kernel + (size_t)phase * klen;
      float acc = 0.0f;
      for (int j = 0; j < klen; j++) acc += k[j] * src[j];
      out[n++] = acc;
    }
  }
  free(padded);
  free(kernel);
  *out_count = n;
  return out;
}
