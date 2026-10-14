/* Named tensors from safetensors files (a sharded index or a single file). */
#ifndef DRAMA3_WEIGHTS_H
#define DRAMA3_WEIGHTS_H

#include <stdbool.h>

#include "mlxu.h"

typedef struct WeightStore WeightStore;

/* Loads <dir>/model.safetensors.index.json shards. */
WeightStore *weights_open_index(const char *dir);
/* Loads a single safetensors file. */
WeightStore *weights_open_file(const char *path);
void weights_free(WeightStore *ws);

/* Owned handle to the named tensor; dies if missing. */
mlx_array weights_get(WeightStore *ws, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
bool weights_has(WeightStore *ws, const char *name);

#endif
