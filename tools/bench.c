/* drama3-bench: where does a decode step spend its time?
 *
 * Splits each phase into CPU graph construction (building the lazy MLX graph)
 * and execution (eval until the GPU is done), and compares the step with the
 * memory-bandwidth floor of the weights it reads.
 *
 * usage: drama3-bench [--model DIR] [--ref-codes CODES.npy] [--steps N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "generate.h"
#include "npy.h"

static size_t nbytes(mlx_array a) { return is_null(a) ? 0 : mlx_array_nbytes(a); }

static size_t tower_layer_bytes(const TowerLayer *t) {
  /* one embedding row is read, not the table */
  return nbytes(t->eh_proj) + nbytes(t->wqkv) + nbytes(t->wo) + nbytes(t->w1) + nbytes(t->w3) +
         nbytes(t->w2);
}

typedef struct {
  double build, run, encode, encode_cpu;
  int n;
} Timer;

static double thread_cpu_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void report(const char *name, const Timer *t) {
  printf("  %-26s build %6.2f ms   run %6.2f ms   total %6.2f ms\n", name, 1e3 * t->build / t->n,
         1e3 * t->run / t->n, 1e3 * (t->build + t->run) / t->n);
}

static void eval_with_caches_mode(Model *m, mlx_array *extra, size_t n_extra, bool async) {
  mlx_array arrays[4 * MAX_LAYERS + 4 * MAX_CODEBOOKS + 8];
  size_t n = model_cache_arrays(m, arrays, sizeof(arrays) / sizeof(arrays[0]) - n_extra);
  for (size_t i = 0; i < n_extra; i++) arrays[n++] = extra[i];
  if (async) m_async_eval_many(arrays, n);
  else m_eval_many(arrays, n);
}

static void eval_with_caches(Model *m, mlx_array *extra, size_t n_extra) {
  eval_with_caches_mode(m, extra, n_extra, false);
}

int main(int argc, char **argv) {
  const char *model_dir = "mlx_model", *ref_codes = NULL;
  int steps = 64;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
    else if (!strcmp(argv[i], "--ref-codes") && i + 1 < argc) ref_codes = argv[++i];
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else die("usage: drama3-bench [--model DIR] [--ref-codes CODES.npy] [--steps N]");
  }
  double t0 = now_seconds();
  Model *m = model_load(model_dir);
  const Config *c = &m->cfg;
  printf("load: %.2f s\n", now_seconds() - t0);

  char err[512];
  Turn ref = {0};
  int n_refs = 0;
  if (ref_codes) {
    if (!npy_read_codes(ref_codes, &ref.codes, err, sizeof(err))) die("%s", err);
    ref.text = "The old lighthouse keeper climbed the spiral stairs every evening.";
    n_refs = 1;
  }
  Rows prompt;
  if (build_prompt(m, "Get the trust fund to the bank early, before the storm arrives tonight.",
                   &ref, n_refs, NULL, 0, &prompt, err, sizeof(err)) != PROMPT_OK)
    die("%s", err);
  int R = prompt.rows, W = prompt.width;

  /* weight bytes read per decode step */
  size_t backbone = nbytes(m->final_norm) + nbytes(m->lm_head_audio), experts = 0;
  for (int i = 0; i < c->n_layers; i++) {
    const BackboneLayer *L = &m->layers[i];
    mlx_array dense[] = {L->in_qkv, L->in_z,   L->in_b,    L->in_a,   L->out_proj,
                         L->q_proj, L->k_proj, L->v_proj,  L->o_proj, L->router,
                         L->sh_gate, L->sh_up, L->sh_expert_gate, L->sh_down};
    for (size_t k = 0; k < sizeof(dense) / sizeof(dense[0]); k++) backbone += nbytes(dense[k]);
    size_t all = nbytes(L->gu_w) + nbytes(L->gu_s) + nbytes(L->gu_b) + nbytes(L->down_w) +
                 nbytes(L->down_s) + nbytes(L->down_b);
    experts += all / c->n_experts * c->top_k;
  }
  size_t tower = 0;
  for (int i = 0; i < c->n_shared; i++) tower += tower_layer_bytes(&m->shared[i]);
  for (int i = 0; i < c->n_codebooks - 1; i++)
    tower += tower_layer_bytes(&m->residual[i]) + nbytes(m->heads[i]);
  printf("weights read per step: backbone dense %.2f GB + experts %.2f GB, tower %.2f GB "
         "= %.2f GB\n",
         backbone / 1e9, experts / 1e9, tower / 1e9, (backbone + experts + tower) / 1e9);

  /* D3_BENCH_CAPACITY: cache rows to allocate (default: just enough) */
  const char *cap_env = getenv("D3_BENCH_CAPACITY");
  int capacity = cap_env ? atoi(cap_env) : 0;
  if (capacity < R + 3 * steps + 16) capacity = R + 3 * steps + 16;
  model_reset_caches(m, capacity);
  t0 = now_seconds();
  mlx_array hidden = prefill_rows(m, prompt.data, R, W, NULL);
  printf("prefill: %d rows in %.3f s\n", R, now_seconds() - t0);

  SamplingParams sp = {1.0f, 0.9f, 30};
  int32_t history[RAS_WINDOW_FRAMES];
  int hist_len = R < RAS_WINDOW_FRAMES ? R : RAS_WINDOW_FRAMES;
  for (int i = 0; i < hist_len; i++) history[i] = prompt.data[(size_t)(R - hist_len + i) * W];
  int32_t frame_host[16];
  memcpy(frame_host, prompt.data + (size_t)(R - 1) * W, W * sizeof(int32_t));

  Timer full = {0}, bb = {0}, tw = {0};
  int pos = R - 1;
  int warmup = 4;
  for (int s = 0; s < steps + warmup; s++) {
    /* full step, as generate_frames runs it */
    size_t mark = arena_mark();
    double a = now_seconds();
    mlx_array logits = backbone_audio_logits(m, hidden);
    mlx_array frame = m_from_data(frame_host, SH(1, W), MLX_INT32);
    mlx_array hist = m_from_int32(history, hist_len);
    mlx_array prefix;
    mlx_array sampled = tower_decode_step(m, logits, hidden, pos, frame, hist, &sp, &prefix);
    mlx_array h = m_reshape(backbone_forward_frames(m, prefix, pos + 1),
                            SH(1, c->hidden));
    double b = now_seconds();
    mlx_array outs[] = {sampled, h};
    const char *dump = getenv("D3_DUMP_GRAPH");
    if (dump && s == warmup) {
      FILE *f = fopen(dump, "w");
      mlx_array arrays[4 * MAX_LAYERS + 4 * MAX_CODEBOOKS + 8];
      size_t n = model_cache_arrays(m, arrays, sizeof(arrays) / sizeof(arrays[0]) - 2);
      arrays[n++] = sampled;
      arrays[n++] = h;
      mlx_vector_array vec = mlx_vector_array_new_data(arrays, n);
      mlx_node_namer namer = mlx_node_namer_new();
      mlx_print_graph(f, namer, vec);
      mlx_node_namer_free(namer);
      mlx_vector_array_free(vec);
      fclose(f);
    }
    /* async eval returns once the step is encoded and committed */
    double cpu0 = thread_cpu_seconds();
    eval_with_caches_mode(m, outs, 2, true);
    double enc = now_seconds();
    double cpu1 = thread_cpu_seconds();
    size_t cnt;
    int32_t *row = m_to_int32(sampled, &cnt);
    double e = now_seconds();
    if (s >= warmup) {
      full.build += b - a;
      full.encode += enc - b;
      full.encode_cpu += cpu1 - cpu0;
      full.run += e - b;
      full.n++;
    }
    memcpy(frame_host, row, W * sizeof(int32_t));
    free(row);
    assign(&hidden, h);
    pos++;
    arena_release(mark);

    /* tower alone */
    mark = arena_mark();
    a = now_seconds();
    logits = backbone_audio_logits(m, hidden);
    sampled = tower_decode_step(m, logits, hidden, pos, m_from_data(frame_host, SH(1, W), MLX_INT32),
                                m_from_int32(history, hist_len), &sp, NULL);
    b = now_seconds();
    eval_with_caches(m, &sampled, 1);
    e = now_seconds();
    if (s >= warmup) {
      tw.build += b - a;
      tw.run += e - b;
      tw.n++;
    }
    arena_release(mark);

    /* backbone alone, on the last sampled frame */
    mark = arena_mark();
    a = now_seconds();
    h = backbone_forward_frames(m, m_from_data(frame_host, SH(1, W), MLX_INT32), pos + 1);
    b = now_seconds();
    eval_with_caches(m, &h, 1);
    e = now_seconds();
    if (s >= warmup) {
      bb.build += b - a;
      bb.run += e - b;
      bb.n++;
    }
    arena_release(mark);
    pos++;
  }
  /* components of the backbone step, each alone in one graph
   * (D3_BENCH_STEPS_ONLY: decode steps only, e.g. under a GPU trace) */
  bool steps_only = getenv("D3_BENCH_STEPS_ONLY") != NULL;
  if (!steps_only) {
    Timer dense = {0}, gather = {0}, norms = {0};
    for (int s = 0; s < 24; s++) {
      size_t mark = arena_mark();
      mlx_array x = m_astype(m_uniform(SH(1, 1, c->hidden), MLX_FLOAT32), MLX_BFLOAT16);
      mlx_array x2 = m_astype(m_uniform(SH(1, 1, 4096), MLX_FLOAT32), MLX_BFLOAT16);
      mlx_array xi = m_astype(m_uniform(SH(1, 1, 1, c->hidden), MLX_FLOAT32), MLX_BFLOAT16);
      mlx_array xd = m_astype(m_uniform(SH(1, 8, 1, c->moe_inter), MLX_FLOAT32), MLX_BFLOAT16);
      mlx_array xs = m_astype(m_uniform(SH(1, 1, c->shared_inter), MLX_FLOAT32), MLX_BFLOAT16);
      mlx_array idx = m_from_int32((int32_t[]){3, 17, 40, 77, 100, 150, 200, 255}, 8);
      idx = m_reshape(idx, SH(1, 8));
      m_eval(x); m_eval(x2); m_eval(xi); m_eval(xd); m_eval(xs); m_eval(idx);
      mlx_array outs[4 * MAX_LAYERS];
      size_t n = 0;
      double a = now_seconds();
      for (int i = 0; i < c->n_layers; i++) {
        BackboneLayer *L = &m->layers[i];
        outs[n++] = m_linear(x, c->is_linear[i] ? L->in_qkv : L->q_proj);
        outs[n++] = m_linear(x2, c->is_linear[i] ? L->out_proj : L->o_proj);
        outs[n++] = m_linear(x, L->router);
        outs[n++] = m_linear(xs, L->sh_down);
      }
      m_eval_many(outs, n);
      double b = now_seconds();
      n = 0;
      for (int i = 0; i < c->n_layers; i++) {
        BackboneLayer *L = &m->layers[i];
        outs[n++] = m_gather_qmm(xi, L->gu_w, L->gu_s, L->gu_b, idx, c->expert_group,
                                 c->expert_bits, false);
        outs[n++] = m_gather_qmm(xd, L->down_w, L->down_s, L->down_b, idx, c->expert_group,
                                 c->expert_bits, false);
      }
      m_eval_many(outs, n);
      double e = now_seconds();
      n = 0;
      for (int i = 0; i < 2 * c->n_layers; i++)
        outs[n++] = m_rms_norm_f32(x, m->layers[i / 2].input_norm, c->rms_eps, MLX_BFLOAT16);
      m_eval_many(outs, n);
      double f = now_seconds();
      if (s >= 4) {
        dense.run += b - a; dense.n++;
        gather.run += e - b; gather.n++;
        norms.run += f - e; norms.n++;
      }
      arena_release(mark);
    }
    /* the same matmuls as dependent chains, as a decode step runs them: each
     * consumes a slice of the previous one's output */
    Timer dense_chain = {0}, gather_chain = {0};
    for (int s = 0; s < 24; s++) {
      size_t mark = arena_mark();
      mlx_array x = m_astype(m_uniform(SH(1, 1, c->hidden), MLX_FLOAT32), MLX_BFLOAT16);
      mlx_array idx = m_reshape(m_from_int32((int32_t[]){3, 17, 40, 77, 100, 150, 200, 255}, 8),
                                SH(1, 8));
      m_eval(x);
      m_eval(idx);
      double t0 = now_seconds();
      mlx_array y = x;
      for (int i = 0; i < c->n_layers; i++) {
        BackboneLayer *L = &m->layers[i];
        mlx_array p = m_linear(y, c->is_linear[i] ? L->in_qkv : L->q_proj);
        y = m_linear(m_slice_axis(p, -1, 0, 4096), c->is_linear[i] ? L->out_proj : L->o_proj);
        mlx_array q = m_linear(y, L->sh_gate);
        y = m_linear(q, L->sh_down);
      }
      m_eval(y);
      double t1 = now_seconds();
      y = m_reshape(x, SH(1, 1, 1, c->hidden));
      for (int i = 0; i < c->n_layers; i++) {
        BackboneLayer *L = &m->layers[i];
        mlx_array gu = m_gather_qmm(y, L->gu_w, L->gu_s, L->gu_b, idx, c->expert_group,
                                    c->expert_bits, false); /* [1, 8, 1, 2 * inter] */
        mlx_array d = m_gather_qmm(m_slice_axis(gu, -1, 0, c->moe_inter), L->down_w, L->down_s,
                                   L->down_b, idx, c->expert_group, c->expert_bits, false);
        y = m_slice_axis(d, 1, 0, 1); /* [1, 1, 1, H] */
      }
      m_eval(y);
      double t2 = now_seconds();
      if (s >= 4) {
        dense_chain.run += t1 - t0; dense_chain.n++;
        gather_chain.run += t2 - t1; gather_chain.n++;
      }
      arena_release(mark);
    }
    /* routing + experts + shared expert (the whole MoE block) vs experts alone */
    Timer moe = {0}, mixer_gdn = {0}, mixer_attn = {0}, sampler = {0};
    model_reset_caches(m, 4096);
    for (int s = 0; s < 24; s++) {
      size_t mark = arena_mark();
      mlx_array x = m_astype(m_uniform(SH(1, 1, c->hidden), MLX_FLOAT32), MLX_BFLOAT16);
      mlx_array lg = m_uniform(SH(1, 4097), MLX_FLOAT32);
      m_eval(x);
      m_eval(lg);
      mlx_array outs[4 * MAX_LAYERS + 4 * MAX_CODEBOOKS + 16];
      double t[5];
      t[0] = now_seconds();
      for (int i = 0; i < c->n_layers; i++) outs[i] = backbone_moe_forward(m, i, x);
      m_eval_many(outs, c->n_layers);
      t[1] = now_seconds();
      size_t n = 0;
      for (int i = 0; i < c->n_layers; i++)
        if (c->is_linear[i]) outs[n++] = backbone_mixer_forward(m, i, x, 100 + s);
      n += model_cache_arrays(m, outs + n, 64);
      m_eval_many(outs, n);
      t[2] = now_seconds();
      n = 0;
      for (int i = 0; i < c->n_layers; i++)
        if (!c->is_linear[i]) outs[n++] = backbone_mixer_forward(m, i, x, 1000 + s);
      n += model_cache_arrays(m, outs + n, 64);
      m_eval_many(outs, n);
      t[3] = now_seconds();
      for (int i = 0; i < 11; i++) outs[i] = sample_top_k_top_p(lg, 1.0f, 30, 0.9f);
      m_eval_many(outs, 11);
      t[4] = now_seconds();
      if (s >= 4) {
        moe.run += t[1] - t[0]; moe.n++;
        mixer_gdn.run += t[2] - t[1]; mixer_gdn.n++;
        mixer_attn.run += t[3] - t[2]; mixer_attn.n++;
        sampler.run += t[4] - t[3]; sampler.n++;
      }
      arena_release(mark);
    }
    printf("blocks alone (eager, single row):\n");
    report("MoE blocks (40)", &moe);
    report("delta-net mixers (30)", &mixer_gdn);
    report("attention mixers (10)", &mixer_attn);
    report("samplers (11)", &sampler);
    printf("backbone components alone (one graph each, mean of 20):\n");
    report("dense matmuls (160)", &dense);
    report("dense matmuls chained", &dense_chain);
    report("expert gathers (80)", &gather);
    report("expert gathers chained", &gather_chain);
    report("zc norms (80)", &norms);
  }
  printf("decode step (mean of %d):\n", steps);
  printf("  full step: encode + commit %.2f ms of the run (thread cpu %.2f ms)\n",
         1e3 * full.encode / full.n, 1e3 * full.encode_cpu / full.n);
  report("full step", &full);
  report("backbone only", &bb);
  report("tower + sampling only", &tw);
  double floor_ms = 1e3 * (backbone + experts + tower) / 400e9;
  printf("full step: %.1f frames/s; bandwidth floor at 400 GB/s: %.1f ms (%.0f frames/s)\n",
         full.n / (full.build + full.run), floor_ms, 1e3 / floor_ms);
  if (steps_only) {
    model_free(m);
    return 0;
  }
  release(&hidden);
  rows_free(&prompt);

  /* the real decode loop on a long sentence */
  {
    Rows long_prompt;
    const char *text =
        "The storm arrived just after midnight. At first it was only a low rumble far out "
        "over the water, the kind of sound you could mistake for a passing truck. Then the "
        "wind picked up, rattling the shutters and bending the old pines along the cliff.";
    if (build_prompt(m, text, &ref, n_refs, NULL, 0, &long_prompt, err, sizeof(err)) != PROMPT_OK)
      die("%s", err);
    for (int run = 0; run < 2; run++) {
      m_seed(1234);
      FinishReason reason;
      double a = now_seconds();
      Codes codes = generate_frames(m, &long_prompt, 600, &sp, false, &reason);
      double e = now_seconds() - a;
      printf("generate_frames run %d: %d frames in %.2f s = %.1f frames/s (incl. prefill of %d "
             "rows)\n", run, codes.frames, e, codes.frames / e, long_prompt.rows);
      codes_free(&codes);
    }
    rows_free(&long_prompt);
  }
  model_free(m);
  return 0;
}
