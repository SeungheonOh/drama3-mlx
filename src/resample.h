/* torchaudio.functional.resample (sinc_interp_hann, lowpass_filter_width 6,
 * rolloff 0.99), as Fish uses on reference audio. */
#ifndef DRAMA3_RESAMPLE_H
#define DRAMA3_RESAMPLE_H

#include <stddef.h>

/* Returns a malloc'd buffer of ceil(new_rate * count / orig_rate) samples. */
float *resample(const float *in, size_t count, int orig_rate, int new_rate, size_t *out_count);

#endif
