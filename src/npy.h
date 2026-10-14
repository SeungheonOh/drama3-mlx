/* 2-D integer .npy files for audio codes (interoperable with fish-speech's codes_N.npy). */
#ifndef DRAMA3_NPY_H
#define DRAMA3_NPY_H

#include <stdbool.h>
#include <stddef.h>

#include "prompt.h"

bool npy_read_codes(const char *path, Codes *out, char *err, size_t err_len);
/* Writes int64 [n_codebooks, frames], like np.save of a torch.long tensor. */
bool npy_write_codes(const char *path, const Codes *codes, char *err, size_t err_len);

#endif
