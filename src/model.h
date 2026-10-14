/* Drama 3 (FishOmniModel): Qwen3.5-MoE hybrid backbone + codebook-MTP tower. */
#ifndef DRAMA3_MODEL_H
#define DRAMA3_MODEL_H

#include <stdbool.h>

#include "mlxu.h"
#include "tokenizer.h"
#include "weights.h"

#define MAX_LAYERS 64
#define MAX_CODEBOOKS 16

typedef struct {
  /* backbone (text_config) */
  int vocab_size, hidden, n_layers;
  int n_heads, n_kv_heads, head_dim, rotary_dim;
  float rope_theta, rms_eps;
  int conv_kernel, lin_key_dim, lin_value_dim, lin_n_key, lin_n_value;
  int n_experts, top_k, moe_inter, shared_inter;
  bool norm_topk;
  bool is_linear[MAX_LAYERS];
  int max_positions;
  /* tower (audio_decoder_config) */
  int t_dim, t_inter, t_heads, t_kv_heads, t_head_dim, t_rotary_dim;
  float t_rope_base, t_eps;
  int code_vocab, n_codebooks, n_shared, vq_backbone_codebooks;
  /* token ids */
  int semantic_start, semantic_end, eos_id;
  /* quantized experts */
  int expert_bits, expert_group;
} Config;

typedef struct {
  /* norms: fp32 (1 + w) */
  mlx_array input_norm, post_norm;
  /* Projections that read the same input are fused into one matrix at load
   * (rows stacked in the order listed), so each runs as a single matmul. */
  /* gated delta net */
  mlx_array in_qkv, in_z, in_b, in_a, conv_w, dt_bias, a_log, gdn_norm, out_proj;
  /* gated full attention (q_proj rows: each head's q, then its gate) */
  mlx_array q_proj, k_proj, v_proj, o_proj, q_norm, k_norm;
  /* sparse MoE: router, gated shared expert */
  mlx_array router, sh_gate, sh_up, sh_expert_gate, sh_down;
  mlx_array gu_w, gu_s, gu_b, down_w, down_s, down_b;
} BackboneLayer;

typedef struct {
  /* KV cache of a full-attention layer: [1, kv_heads, capacity, head_dim] */
  mlx_array k, v;
  /* gated delta net state: conv [1, kernel-1, conv_dim], recurrent [1, Hv, Dv, Dk] fp32 */
  mlx_array conv, rec;
} LayerCache;

typedef struct {
  mlx_array embedding;                /* [code_vocab, dim] */
  mlx_array enorm, hnorm, final_norm; /* fp32 (1 + w) */
  mlx_array eh_proj;                  /* [dim, 2 dim] */
  mlx_array attn_norm, ffn_norm, q_norm, k_norm;
  mlx_array wqkv, wo, w1, w2, w3;
  /* KV cache over frame rows: [1, kv_heads, capacity, head_dim] */
  mlx_array k_cache, v_cache;
} TowerLayer;

typedef struct {
  Config cfg;
  WeightStore *weights;
  Tokenizer *tok;
  mlx_array embed_tokens;  /* [vocab, hidden] */
  mlx_array final_norm;    /* fp32 (1 + w) */
  mlx_array lm_head_audio; /* [1 + code_vocab, hidden]: eos then the semantic band */
  BackboneLayer layers[MAX_LAYERS];
  LayerCache cache[MAX_LAYERS];
  mlx_array codebook_embeddings; /* [code_vocab * n_codebooks, dim] */
  TowerLayer shared[MAX_CODEBOOKS];
  TowerLayer residual[MAX_CODEBOOKS];
  mlx_array heads[MAX_CODEBOOKS];
  int capacity; /* rows the caches hold */
  /* gather_qmm lhs indices of a single-row MoE: zeros / arange, both [1, top_k] */
  mlx_array moe_lhs_up, moe_lhs_down;
  /* compiled single-row decode steps */
  struct {
    Compiled gdn, attn_pre, attn_post, attn;
  } backbone_compiled;
  struct {
    Compiled pre, post, post_head, layer, layer_head;
  } tower_compiled;
} Model;

Model *model_load(const char *dir);
void model_free(Model *m);
/* Ensures caches for capacity rows and clears all recurrent state. */
void model_reset_caches(Model *m, int capacity);
/* All cache arrays, for evaluation after a step (returns count). */
size_t model_cache_arrays(Model *m, mlx_array *out, size_t cap);

/* ---- backbone ---- */
/* Input embeddings of frame rows [T, 1 + n_codebooks] (int32 on device). */
mlx_array backbone_embed_frames(Model *m, mlx_array frames);
/* Final-norm hidden states [1, T, hidden] (bf16) for rows at pos..pos+T-1. */
mlx_array backbone_forward(Model *m, mlx_array embeds, int pos);
/* backbone_forward of the embeddings of frame rows [T, 1 + n_codebooks]. */
mlx_array backbone_forward_frames(Model *m, mlx_array frames, int pos);
/* One decoder layer (and its parts) on x [1, T, hidden]; for tests. */
mlx_array backbone_layer_forward(Model *m, int layer, mlx_array x, int pos);
mlx_array backbone_mixer_forward(Model *m, int layer, mlx_array x, int pos);
mlx_array backbone_moe_forward(Model *m, int layer, mlx_array x);
/* Logits over [eos, semantic 0..V-1] for hidden [N, hidden]: [N, 1 + V]. */
mlx_array backbone_audio_logits(Model *m, mlx_array hidden);

/* ---- fused single-row attention (attention.c) ---- */
typedef struct {
  int heads, kv_heads, head_dim, rotary_dim;
  bool traditional; /* interleaved RoPE pairs */
  float rope_base, eps;
  int q_stride;  /* projection offset from one head's q to the next */
  int gate_base; /* offset of a head's gate from its q */
} AttnShape;
/* How a row attending over `rows` cached rows (its own included) runs fused,
 * following MLX's choice of attention kernel: -1 not fused (D3_NO_FUSED_ATTN, or
 * a kernel the fused path does not reproduce), 0 single pass, else the two-pass
 * kernel's block count. */
int attn_route(const AttnShape *s, int rows);
/* One row's attention projections: qg [1, 1, *] with each head's q (and its
 * gate at gate_base from it), k and v [1, 1, kv_heads * head_dim]; caches
 * [1, kv_heads, cap, head_dim] hold rows 0..pos-1 (pos a scalar int32 array);
 * blocks from attn_route. out [1, 1, heads * head_dim] is the gated attention
 * output and k_new / v_new [1, kv_heads, 1, head_dim] the row for the caches. */
void attn_decode_fused(const AttnShape *s, int blocks, mlx_array qg, mlx_array k, mlx_array v,
                       mlx_array q_norm, mlx_array k_norm, mlx_array k_cache, mlx_array v_cache,
                       mlx_array pos, mlx_array *out, mlx_array *k_new, mlx_array *v_new);
/* A constant array whose shape encodes blocks, as an input of a compiled closure
 * that calls attn_decode_fused: each variant gets its own trace. */
mlx_array attn_variant_tag(int blocks);
/* Blocks back from a tag. */
#define ATTN_TAG_BLOCKS(tag) (m_dim((tag), 0) - 1)
/* Writes row [1, kv, 1, hd] into cache [1, kv, cap, hd] at pos. */
void cache_write_row(mlx_array *cache, mlx_array row, int pos);

/* ---- tower ---- */
/* Teacher-forced tower pass over rows pos..pos+T-1, filling its caches.
 * hidden [T, dim], frames / next_frames [T, 1 + n_codebooks] int32. */
void tower_prefill(Model *m, mlx_array hidden, int pos, mlx_array frames,
                   mlx_array next_frames);

typedef struct {
  float temperature, top_p;
  int top_k;
} SamplingParams;

/* Samples the frame after row pos: returns [1, 1 + n_codebooks] int32.
 * logits [1, 1 + V] from backbone_audio_logits, hidden [1, dim], frame [1, 1+N]
 * the row's own frame, history [W] the main tokens of the last W rows.
 * backbone_input (optional) receives the leading columns of the frame that the
 * backbone's embedding reads ([1, 2 + vq_backbone_codebooks]): they are ready
 * before the remaining codebooks, so the next backbone step need not wait. */
mlx_array tower_decode_step(Model *m, mlx_array logits, mlx_array hidden, int pos,
                            mlx_array frame, mlx_array history,
                            const SamplingParams *params, mlx_array *backbone_input);

/* Teacher-forced tower pass for one row: per-codebook head logits [N-1, V] for
 * codes 1..N-1 of next_frame, given the row's hidden [1, dim] and frame. */
mlx_array tower_forced_logits(Model *m, mlx_array hidden, int pos, mlx_array frame,
                              mlx_array next_frame);

/* Config + tokenizer only (no weights), for prompt building. */
Model *model_load_light(const char *dir);

/* ---- sampling ---- */
/* One id per row of logits [N, V] (positions into the last axis). */
mlx_array sample_top_k_top_p(mlx_array logits, float temperature, int top_k, float top_p);
/* The uniform draws sample_top_k_top_p makes for logits [rows, V], taken from the
 * global random stream on the CPU (the same values the GPU would draw), lazy. */
mlx_array sample_noise(int rows, int V, int top_k);
/* sample_top_k_top_p with its uniform draws u [rows, k] given. */
mlx_array sample_with_noise(mlx_array logits, mlx_array u, float temperature, float top_p);

#endif
