/* RIFF WAVE reading (PCM 8/16/24/32-bit, IEEE float 32/64) and writing. */
#ifndef DRAMA3_WAV_H
#define DRAMA3_WAV_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
  float *samples; /* interleaved, normalized to [-1, 1) like torchaudio.load */
  size_t frames;
  int channels, sample_rate;
} Wav;

bool wav_read(const char *path, Wav *out, char *err, size_t err_len);
/* Mono samples; bits 16 (PCM, libsndfile-style scaling by 32767) or 32 (float). */
bool wav_write(const char *path, const float *samples, size_t count, int sample_rate,
               int bits, char *err, size_t err_len);
/* Channel mean, like wav.mean(dim=0). Returns a malloc'd mono buffer. */
float *wav_mono(const Wav *w);
void wav_free(Wav *w);

#endif
