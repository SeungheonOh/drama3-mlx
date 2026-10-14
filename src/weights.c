#include "weights.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

struct WeightStore {
  mlx_map_string_to_array *maps;
  size_t count;
};

static void add_file(WeightStore *ws, const char *path) {
  FILE *probe = fopen(path, "rb");
  if (!probe) die("cannot open %s (run tools/convert.py, or pass --model DIR)", path);
  fclose(probe);
  mlx_map_string_to_array map = mlx_map_string_to_array_new();
  mlx_map_string_to_string meta = mlx_map_string_to_string_new();
  mlx_stream cpu = mlx_default_cpu_stream_new();
  if (mlx_load_safetensors(&map, &meta, path, cpu)) die("cannot load %s", path);
  mlx_stream_free(cpu);
  mlx_map_string_to_string_free(meta);
  ws->maps = realloc(ws->maps, (ws->count + 1) * sizeof(mlx_map_string_to_array));
  ws->maps[ws->count++] = map;
}

WeightStore *weights_open_file(const char *path) {
  WeightStore *ws = calloc(1, sizeof(WeightStore));
  add_file(ws, path);
  return ws;
}

WeightStore *weights_open_index(const char *dir) {
  char path[4096], err[256];
  snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
  JDoc *doc = json_parse_file(path, err, sizeof(err));
  if (!doc) die("%s", err);
  JNode *map = json_get(json_root(doc), "weight_map");
  if (!map || map->type != J_OBJ) die("%s has no weight_map", path);
  /* unique shard names */
  const char **files = calloc(map->count, sizeof(char *));
  size_t nfiles = 0;
  for (size_t i = 0; i < map->count; i++) {
    const char *f = map->items[i]->str;
    bool seen = false;
    for (size_t j = 0; j < nfiles && !seen; j++) seen = !strcmp(files[j], f);
    if (!seen) files[nfiles++] = f;
  }
  WeightStore *ws = calloc(1, sizeof(WeightStore));
  for (size_t i = 0; i < nfiles; i++) {
    snprintf(path, sizeof(path), "%s/%s", dir, files[i]);
    add_file(ws, path);
  }
  free(files);
  json_free(doc);
  return ws;
}

void weights_free(WeightStore *ws) {
  if (!ws) return;
  for (size_t i = 0; i < ws->count; i++) mlx_map_string_to_array_free(ws->maps[i]);
  free(ws->maps);
  free(ws);
}

static bool lookup(WeightStore *ws, const char *name, mlx_array *out) {
  for (size_t i = 0; i < ws->count; i++) {
    mlx_array a = mlx_array_new();
    if (mlx_map_string_to_array_get(&a, ws->maps[i], name) == 0 && a.ctx) {
      *out = a;
      return true;
    }
    mlx_array_free(a);
  }
  return false;
}

mlx_array weights_get(WeightStore *ws, const char *fmt, ...) {
  char name[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(name, sizeof(name), fmt, args);
  va_end(args);
  mlx_array a;
  if (!lookup(ws, name, &a)) die("missing weight %s", name);
  return a;
}

bool weights_has(WeightStore *ws, const char *name) {
  mlx_array a;
  if (!lookup(ws, name, &a)) return false;
  mlx_array_free(a);
  return true;
}
