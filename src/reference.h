/* Reference audio for voice cloning: WAV -> codes, as fish-speech encode_audio does. */
#ifndef DRAMA3_REFERENCE_H
#define DRAMA3_REFERENCE_H

#include <stdbool.h>
#include <stddef.h>

#include "codec.h"

/* Reads a WAV file, averages its channels, resamples to the codec rate
 * (torchaudio sinc_interp_hann) and encodes it to codes [n_codebooks, frames]. */
bool encode_reference(Codec *codec, const char *path, Codes *out, bool verbose, char *err,
                      size_t err_len);

#endif
