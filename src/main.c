/* drama3-tts: Fish Audio Drama 3 text-to-speech on MLX. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "codec.h"
#include "generate.h"
#include "npy.h"
#include "reference.h"
#include "resample.h"
#include "wav.h"

#define MAX_ITEMS 32

static void usage(void) {
  fprintf(stderr,
          "usage:\n"
          "  drama3-tts tts (--text TEXT | --text-file PATH) --output OUT.wav [options]\n"
          "  drama3-tts encode --input IN.wav --output CODES.npy [--codec-dtype T]\n"
          "  drama3-tts decode --input CODES.npy --output OUT.wav [--codec-dtype T]\n"
          "\n"
          "tts options:\n"
          "  --model DIR           converted model directory (default: mlx_model)\n"
          "  --ref-audio PATH      reference clip (WAV) to clone; repeatable\n"
          "  --ref-codes PATH      reference codes (.npy) instead of audio; repeatable\n"
          "  --ref-text TEXT       transcript of each reference, in the same order\n"
          "  --take-text TEXT      text of a previous take (editing as continuation)\n"
          "  --take-codes PATH     codes (.npy) of that take, or --take-audio PATH (WAV)\n"
          "  --save-codes PATH     also save the generated codes as .npy\n"
          "  --temperature F       (default 0.7)\n"
          "  --top-p F             (default 0.7)\n"
          "  --top-k N             (default 30)\n"
          "  --max-new-tokens N    frames per text chunk; 0 = 600 (2048 single chunk)"
          " (default 1024)\n"
          "  --seed N              (default 42)\n"
          "  --single-chunk        generate the whole text in one pass\n"
          "  --codec-dtype T       float32 | bfloat16 | float16 (default bfloat16)\n"
          "  --wav-bits N          16 (PCM) or 32 (float) output (default 16)\n"
          "  --quiet               less logging\n");
  exit(2);
}

static char *read_file(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open %s", path);
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = malloc((size_t)size + 1);
  if (fread(buf, 1, (size_t)size, f) != (size_t)size) die("cannot read %s", path);
  buf[size] = 0;
  fclose(f);
  return buf;
}

static mlx_dtype parse_dtype(const char *s) {
  if (!strcmp(s, "float32")) return MLX_FLOAT32;
  if (!strcmp(s, "bfloat16")) return MLX_BFLOAT16;
  if (!strcmp(s, "float16")) return MLX_FLOAT16;
  die("unknown dtype %s (float32, bfloat16, float16)", s);
}

static Codes encode_file(Codec *codec, const char *path, bool verbose) {
  char err[512];
  Codes codes;
  if (!encode_reference(codec, path, &codes, verbose, err, sizeof(err))) die("%s", err);
  return codes;
}

static void decode_to_file(Codec *codec, const Codes *codes, const char *path, int bits,
                           bool verbose) {
  char err[512];
  double t0 = now_seconds();
  size_t n;
  float *audio = codec_decode(codec, codes, &n);
  if (verbose)
    log_info("decoded %d frames -> %.2fs of audio in %.2fs", codes->frames,
             (double)n / codec_sample_rate(codec), now_seconds() - t0);
  if (!wav_write(path, audio, n, codec_sample_rate(codec), bits, err, sizeof(err)))
    die("%s", err);
  free(audio);
}

typedef struct {
  const char *path;
  bool is_audio;
} Source;

int main(int argc, char **argv) {
  if (argc < 2) usage();
  const char *mode = argv[1];
  const char *model_dir = "mlx_model";
  const char *text = NULL, *text_file = NULL, *output = NULL, *input = NULL;
  const char *save_codes = NULL;
  Source ref_src[MAX_ITEMS], take_src[MAX_ITEMS];
  const char *ref_text[MAX_ITEMS], *take_text[MAX_ITEMS];
  int n_ref_src = 0, n_ref_text = 0, n_take_src = 0, n_take_text = 0;
  GenOptions opt = {{0.7f, 0.7f, 30}, 1024, false, true};
  long seed = 42;
  mlx_dtype codec_dtype = MLX_BFLOAT16;
  int wav_bits = 16;

  for (int i = 2; i < argc; i++) {
    const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (die("%s needs a value", a), ""))
    if (!strcmp(a, "--model")) model_dir = NEXT();
    else if (!strcmp(a, "--text")) text = NEXT();
    else if (!strcmp(a, "--text-file")) text_file = NEXT();
    else if (!strcmp(a, "--output") || !strcmp(a, "-o")) output = NEXT();
    else if (!strcmp(a, "--input") || !strcmp(a, "-i")) input = NEXT();
    else if (!strcmp(a, "--save-codes")) save_codes = NEXT();
    else if (!strcmp(a, "--ref-audio") || !strcmp(a, "--ref-codes")) {
      if (n_ref_src == MAX_ITEMS) die("too many references");
      ref_src[n_ref_src].is_audio = !strcmp(a, "--ref-audio");
      ref_src[n_ref_src++].path = NEXT();
    } else if (!strcmp(a, "--ref-text")) {
      if (n_ref_text == MAX_ITEMS) die("too many references");
      ref_text[n_ref_text++] = NEXT();
    } else if (!strcmp(a, "--take-audio") || !strcmp(a, "--take-codes")) {
      if (n_take_src == MAX_ITEMS) die("too many takes");
      take_src[n_take_src].is_audio = !strcmp(a, "--take-audio");
      take_src[n_take_src++].path = NEXT();
    } else if (!strcmp(a, "--take-text")) {
      if (n_take_text == MAX_ITEMS) die("too many takes");
      take_text[n_take_text++] = NEXT();
    } else if (!strcmp(a, "--temperature")) opt.sampling.temperature = (float)atof(NEXT());
    else if (!strcmp(a, "--top-p")) opt.sampling.top_p = (float)atof(NEXT());
    else if (!strcmp(a, "--top-k")) opt.sampling.top_k = atoi(NEXT());
    else if (!strcmp(a, "--max-new-tokens")) opt.max_new_tokens = atoi(NEXT());
    else if (!strcmp(a, "--seed")) seed = atol(NEXT());
    else if (!strcmp(a, "--single-chunk")) opt.single_chunk = true;
    else if (!strcmp(a, "--codec-dtype")) codec_dtype = parse_dtype(NEXT());
    else if (!strcmp(a, "--wav-bits")) wav_bits = atoi(NEXT());
    else if (!strcmp(a, "--quiet")) opt.verbose = false;
    else if (!strcmp(a, "--help") || !strcmp(a, "-h")) usage();
    else die("unknown option %s", a);
#undef NEXT
  }
  if (wav_bits != 16 && wav_bits != 32) die("--wav-bits must be 16 or 32");
  mlxu_init();
  char err[1024];
  char codec_path[4096];
  snprintf(codec_path, sizeof(codec_path), "%s/codec.safetensors", model_dir);

  if (!strcmp(mode, "encode") || !strcmp(mode, "decode")) {
    if (!input || !output) usage();
    Codec *codec = codec_load(codec_path, codec_dtype);
    if (!strcmp(mode, "encode")) {
      Codes codes = encode_file(codec, input, opt.verbose);
      if (!npy_write_codes(output, &codes, err, sizeof(err))) die("%s", err);
      codes_free(&codes);
    } else {
      Codes codes;
      if (!npy_read_codes(input, &codes, err, sizeof(err))) die("%s", err);
      decode_to_file(codec, &codes, output, wav_bits, opt.verbose);
      codes_free(&codes);
    }
    codec_free(codec);
    return 0;
  }
  if (strcmp(mode, "tts")) usage();
  if ((!text) == (!text_file) || !output) usage();
  if (n_ref_text != n_ref_src) die("give one --ref-text per --ref-audio / --ref-codes");
  if (n_take_text != n_take_src) die("give one --take-text per --take-audio / --take-codes");
  if (opt.sampling.top_k < 1) die("--top-k must be at least 1");
  if (!(opt.sampling.top_p > 0.0f && opt.sampling.top_p <= 1.0f))
    die("--top-p must be in (0, 1]");
  if (!(opt.sampling.temperature >= 0.0f)) die("--temperature must be >= 0");
  char *owned_text = text_file ? read_file(text_file) : NULL;
  if (owned_text) text = owned_text;

  double t0 = now_seconds();
  Model *model = model_load(model_dir);
  Codec *codec = codec_load(codec_path, codec_dtype);
  if (opt.verbose) log_info("model ready in %.1fs", now_seconds() - t0);

  Turn refs[MAX_ITEMS], takes[MAX_ITEMS];
  for (int i = 0; i < n_ref_src; i++) {
    refs[i].text = (char *)ref_text[i];
    if (ref_src[i].is_audio) {
      refs[i].codes = encode_file(codec, ref_src[i].path, opt.verbose);
    } else if (!npy_read_codes(ref_src[i].path, &refs[i].codes, err, sizeof(err))) {
      die("%s", err);
    }
  }
  for (int i = 0; i < n_take_src; i++) {
    takes[i].text = (char *)take_text[i];
    if (take_src[i].is_audio) {
      takes[i].codes = encode_file(codec, take_src[i].path, opt.verbose);
    } else if (!npy_read_codes(take_src[i].path, &takes[i].codes, err, sizeof(err))) {
      die("%s", err);
    }
  }

  m_seed((uint64_t)seed);
  Codes codes;
  double t1 = now_seconds();
  if (!generate_long(model, text, refs, n_ref_src, takes, n_take_src, &opt, &codes, err,
                     sizeof(err)))
    die("%s", err);
  if (opt.verbose) log_info("generated %d frames in %.2fs", codes.frames, now_seconds() - t1);
  if (!codes.frames) die("no audio was generated");
  if (save_codes && !npy_write_codes(save_codes, &codes, err, sizeof(err))) die("%s", err);
  decode_to_file(codec, &codes, output, wav_bits, opt.verbose);
  if (opt.verbose) log_info("wrote %s", output);

  codes_free(&codes);
  for (int i = 0; i < n_ref_src; i++) codes_free(&refs[i].codes);
  for (int i = 0; i < n_take_src; i++) codes_free(&takes[i].codes);
  codec_free(codec);
  model_free(model);
  free(owned_text);
  return 0;
}
