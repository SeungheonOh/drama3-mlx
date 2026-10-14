/* Fish S2-Pro codec (modded DAC + downsampling RVQ), 44.1 kHz, 2048 samples/frame. */
#ifndef DRAMA3_CODEC_H
#define DRAMA3_CODEC_H

#include <stdint.h>

#include "mlxu.h"
#include "prompt.h"

typedef struct Codec Codec;

/* dtype: MLX_FLOAT32, MLX_BFLOAT16 or MLX_FLOAT16 for weights and activations. */
Codec *codec_load(const char *path, mlx_dtype dtype);
void codec_free(Codec *c);
int codec_sample_rate(const Codec *c);
int codec_frame_length(const Codec *c);

/* Mono audio at the codec rate -> codes [n_codebooks, frames]. */
Codes codec_encode(Codec *c, const float *audio, size_t samples);
/* Codes [n_codebooks, frames] -> audio (malloc'd, frames * frame_length samples). */
float *codec_decode(Codec *c, const Codes *codes, size_t *samples);

#endif
