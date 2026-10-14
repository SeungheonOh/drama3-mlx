#include "reference.h"

#include <stdio.h>
#include <stdlib.h>

#include "resample.h"
#include "wav.h"

bool encode_reference(Codec *codec, const char *path, Codes *out, bool verbose, char *err,
                      size_t err_len) {
  Wav w;
  if (!wav_read(path, &w, err, err_len)) return false;
  if (!w.frames) {
    snprintf(err, err_len, "%s: no audio samples", path);
    wav_free(&w);
    return false;
  }
  float *mono = wav_mono(&w);
  size_t n;
  float *audio = resample(mono, w.frames, w.sample_rate, codec_sample_rate(codec), &n);
  double t0 = now_seconds();
  *out = codec_encode(codec, audio, n);
  if (verbose)
    log_info("encoded %s (%.2fs at %d Hz, %d ch) -> %d frames in %.2fs", path,
             (double)w.frames / w.sample_rate, w.sample_rate, w.channels, out->frames,
             now_seconds() - t0);
  free(audio);
  free(mono);
  wav_free(&w);
  return true;
}
