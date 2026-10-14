/* Codebook-MTP audio decoder tower, following fish_speech fish_omni tower.py. */
#include <math.h>

#include "model.h"

/* Repetition-aware sampling of the main token, fixed by the training recipe. */
#define RAS_TEMPERATURE 1.0f
#define RAS_TOP_K 30
#define RAS_TOP_P 0.95f

static mlx_array zc_norm(mlx_array x, mlx_array w1, float eps) {
  mlx_dtype dtype = m_dtype(x);
  return m_astype(m_rms_norm(m_astype(x, MLX_FLOAT32), w1, eps), dtype);
}

/* Gated attention over frame rows pos..pos+L-1; x [L, dim]. */
static mlx_array tower_attention(Model *m, TowerLayer *t, mlx_array x, int pos) {
  const Config *c = &m->cfg;
  int L = m_dim(x, 0);
  int hd = c->t_head_dim;
  int q_width = c->t_heads * hd, kv_width = c->t_kv_heads * hd;
  mlx_array parts[4];
  m_split(m_linear(x, t->wqkv), (int[]){q_width, 2 * q_width, 2 * q_width + kv_width}, 3, -1,
          parts);
  mlx_array q = zc_norm(m_reshape(parts[0], SH(L, c->t_heads, hd)), t->q_norm, c->t_eps);
  mlx_array k = zc_norm(m_reshape(parts[2], SH(L, c->t_kv_heads, hd)), t->k_norm, c->t_eps);
  mlx_array v = m_reshape(parts[3], SH(L, c->t_kv_heads, hd));
  /* [1, heads, L, hd]; interleaved partial RoPE over absolute row positions */
  q = m_rope(m_expand_dims(m_transpose(q, SH(1, 0, 2)), 0), c->t_rotary_dim, true,
             c->t_rope_base, pos);
  k = m_rope(m_expand_dims(m_transpose(k, SH(1, 0, 2)), 0), c->t_rotary_dim, true,
             c->t_rope_base, pos);
  v = m_expand_dims(m_transpose(v, SH(1, 0, 2)), 0);

  int end = pos + L;
  if (end > m_dim(t->k_cache, 2)) die("tower cache capacity exceeded");
  assign(&t->k_cache, m_slice_update(t->k_cache, k, (int[]){0, 0, pos, 0},
                                     (int[]){1, c->t_kv_heads, end, hd}, 4));
  assign(&t->v_cache, m_slice_update(t->v_cache, v, (int[]){0, 0, pos, 0},
                                     (int[]){1, c->t_kv_heads, end, hd}, 4));
  mlx_array keys = m_slice_axis(t->k_cache, 2, 0, end);
  mlx_array values = m_slice_axis(t->v_cache, 2, 0, end);
  mlx_array out = m_sdpa(q, keys, values, 1.0f / sqrtf((float)hd), L > 1 ? "causal" : "",
                         null_array());
  out = m_reshape(m_transpose(m_squeeze(out, 0), SH(1, 0, 2)), SH(L, q_width));
  return m_linear(m_mul(out, m_sigmoid(parts[1])), t->wo);
}

static mlx_array tower_block(Model *m, TowerLayer *t, mlx_array x, int pos) {
  const Config *c = &m->cfg;
  x = m_add(x, tower_attention(m, t, zc_norm(x, t->attn_norm, c->t_eps), pos));
  mlx_array h = zc_norm(x, t->ffn_norm, c->t_eps);
  mlx_array ff = m_linear(m_mul(m_silu_f32(m_linear(h, t->w1)), m_linear(h, t->w3)), t->w2);
  return m_add(x, ff);
}

/* TowerLayer: mix a code embedding into the hidden state, then one block.
 * hidden [L, dim], codes [L] int32, semantic [L] bool. */
static mlx_array tower_layer(Model *m, TowerLayer *t, mlx_array hidden, mlx_array codes,
                             mlx_array semantic, int pos, mlx_array shared_input) {
  const Config *c = &m->cfg;
  mlx_array clamped = m_clip(codes, m_int(0), m_int(c->code_vocab - 1));
  mlx_array e = m_take(t->embedding, clamped, 0);
  e = m_where(m_expand_dims(semantic, -1), e, m_scalar(0.0f, m_dtype(e)));
  if (!is_null(shared_input)) e = m_add(e, shared_input);
  mlx_array cat = m_concat2(zc_norm(e, t->enorm, c->t_eps),
                            zc_norm(hidden, t->hnorm, c->t_eps), -1);
  mlx_array mixed = m_linear(cat, t->eh_proj);
  return zc_norm(tower_block(m, t, mixed, pos), t->final_norm, c->t_eps);
}

static mlx_array is_semantic(Model *m, mlx_array tokens) {
  return m_logical_and(m_greater_equal(tokens, m_int(m->cfg.semantic_start)),
                       m_less_equal(tokens, m_int(m->cfg.semantic_end)));
}

/* Summed embeddings of codes first..end-1 of each frame row, [L, dim] bf16. */
static mlx_array embedding_sum(Model *m, mlx_array frames, int first, int end) {
  const Config *c = &m->cfg;
  int L = m_dim(frames, 0);
  if (first == end) return m_zeros(SH(L, c->t_dim), MLX_BFLOAT16);
  mlx_array acc = null_array();
  for (int j = first; j < end; j++) {
    mlx_array codes = m_reshape(m_slice_axis(frames, 1, 1 + j, 2 + j), SH(L));
    codes = m_add(codes, m_int(j * c->code_vocab));
    mlx_array e = m_astype(m_take(m->codebook_embeddings, codes, 0), MLX_FLOAT32);
    acc = is_null(acc) ? e : m_add(acc, e);
  }
  return m_astype(acc, MLX_BFLOAT16);
}

static mlx_array run_shared(Model *m, mlx_array hidden, mlx_array c0, mlx_array semantic,
                            int pos, mlx_array frames) {
  const Config *c = &m->cfg;
  int L = m_dim(frames, 0);
  mlx_array unseen = embedding_sum(m, frames, 1 + c->vq_backbone_codebooks, c->n_codebooks);
  mlx_array frame_semantic =
      is_semantic(m, m_reshape(m_slice_axis(frames, 1, 0, 1), SH(L)));
  unseen = m_where(m_expand_dims(frame_semantic, -1), unseen, m_scalar(0.0f, MLX_BFLOAT16));
  for (int i = 0; i < c->n_shared; i++)
    hidden = tower_layer(m, &m->shared[i], hidden, c0, semantic, pos,
                         i == 0 ? unseen : null_array());
  return hidden;
}

static mlx_array column(mlx_array frames, int col) {
  return m_reshape(m_slice_axis(frames, 1, col, col + 1), SH(m_dim(frames, 0)));
}

void tower_prefill(Model *m, mlx_array hidden, int pos, mlx_array frames,
                   mlx_array next_frames) {
  const Config *c = &m->cfg;
  mlx_array semantic = is_semantic(m, column(next_frames, 0));
  mlx_array h = run_shared(m, hidden, column(next_frames, 1), semantic, pos, frames);
  for (int j = 0; j < c->n_codebooks - 1; j++)
    h = tower_layer(m, &m->residual[j], h, column(next_frames, j + 1), semantic, pos,
                    null_array());
}

/* Candidate index -> token id: 0 is eos, i >= 1 is semantic_start + i - 1. */
static mlx_array candidate_token(Model *m, mlx_array idx) {
  return m_where(m_equal(idx, m_int(0)), m_int(m->cfg.eos_id),
                 m_add(idx, m_int(m->cfg.semantic_start - 1)));
}

mlx_array tower_decode_step(Model *m, mlx_array logits, mlx_array hidden, int pos,
                            mlx_array frame, mlx_array history,
                            const SamplingParams *p) {
  const Config *c = &m->cfg;
  mlx_array first = candidate_token(
      m, m_astype(sample_top_k_top_p(logits, p->temperature, p->top_k, p->top_p), MLX_INT32));
  mlx_array retry = candidate_token(
      m, m_astype(sample_top_k_top_p(logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P),
                  MLX_INT32));
  /* Only audio tokens are resampled; a stop must not be redrawn. */
  mlx_array repeated =
      m_any(m_equal(history, m_reshape(first, SH(1))));
  repeated = m_logical_and(repeated, m_reshape(is_semantic(m, first), SH(-1)));
  mlx_array main_tok = m_where(repeated, retry, first); /* [1] */
  mlx_array c0 = m_clip(m_sub(main_tok, m_int(c->semantic_start)), m_int(0),
                        m_int(c->code_vocab - 1));
  mlx_array semantic = is_semantic(m, main_tok);
  mlx_array h = run_shared(m, hidden, c0, semantic, pos, frame);
  mlx_array codes[MAX_CODEBOOKS + 1];
  codes[0] = main_tok;
  codes[1] = c0;
  for (int j = 0; j < c->n_codebooks - 1; j++) {
    h = tower_layer(m, &m->residual[j], h, codes[j + 1], semantic, pos, null_array());
    mlx_array head_logits = m_linear(h, m->heads[j]);
    codes[j + 2] =
        m_astype(sample_top_k_top_p(head_logits, p->temperature, p->top_k, p->top_p),
                 MLX_INT32);
  }
  return m_stack(codes, c->n_codebooks + 1, -1); /* [1, 1 + N] */
}

mlx_array tower_forced_logits(Model *m, mlx_array hidden, int pos, mlx_array frame,
                              mlx_array next_frame) {
  const Config *c = &m->cfg;
  mlx_array semantic = is_semantic(m, column(next_frame, 0));
  mlx_array h = run_shared(m, hidden, column(next_frame, 1), semantic, pos, frame);
  mlx_array logits[MAX_CODEBOOKS];
  for (int j = 0; j < c->n_codebooks - 1; j++) {
    h = tower_layer(m, &m->residual[j], h, column(next_frame, j + 1), semantic, pos,
                    null_array());
    logits[j] = m_linear(h, m->heads[j]);
  }
  return m_concat(logits, c->n_codebooks - 1, 0);
}
