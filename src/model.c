#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

static int req_int(const JNode *obj, const char *key) {
  JNode *n = json_get(obj, key);
  if (!n || n->type != J_NUM) die("config: missing integer %s", key);
  return (int)n->number;
}

static float req_float(const JNode *obj, const char *key) {
  JNode *n = json_get(obj, key);
  if (!n || n->type != J_NUM) die("config: missing number %s", key);
  return (float)n->number;
}

static void require_flag(const JNode *obj, const char *key, bool fallback, bool want) {
  if (json_bool(obj, key, fallback) != want)
    die("config: %s must be %s", key, want ? "true" : "false");
}

static void parse_config(const char *dir, Config *c) {
  char path[4096], err[256];
  snprintf(path, sizeof(path), "%s/config.json", dir);
  JDoc *doc = json_parse_file(path, err, sizeof(err));
  if (!doc) die("%s", err);
  JNode *root = json_root(doc);
  const char *type = json_str(root, "model_type");
  if (!type || strcmp(type, "fish_omni")) die("config: model_type must be fish_omni");
  const char *tmpl = json_str(root, "speech_template_version");
  if (!tmpl || strcmp(tmpl, "reference-bracket-v1"))
    die("config: only speech_template_version reference-bracket-v1 is supported");
  JNode *conv = json_get(root, "mlx_conversion");
  if (!conv) die("config: %s was not produced by tools/convert.py", path);

  JNode *t = json_get(root, "text_config");
  c->vocab_size = req_int(t, "vocab_size");
  c->hidden = req_int(t, "hidden_size");
  c->n_layers = req_int(t, "num_hidden_layers");
  if (c->n_layers > MAX_LAYERS) die("config: too many layers");
  c->n_heads = req_int(t, "num_attention_heads");
  c->n_kv_heads = req_int(t, "num_key_value_heads");
  c->head_dim = req_int(t, "head_dim");
  JNode *rope = json_get(t, "rope_parameters");
  if (!rope) rope = t;
  const char *rope_type = json_str(rope, "rope_type");
  if (rope_type && strcmp(rope_type, "default")) die("config: only default rope");
  c->rope_theta = req_float(rope, "rope_theta");
  c->rotary_dim = (int)(c->head_dim * json_num(rope, "partial_rotary_factor", 0.25));
  c->rms_eps = req_float(t, "rms_norm_eps");
  c->conv_kernel = req_int(t, "linear_conv_kernel_dim");
  c->lin_key_dim = req_int(t, "linear_key_head_dim");
  c->lin_value_dim = req_int(t, "linear_value_head_dim");
  c->lin_n_key = req_int(t, "linear_num_key_heads");
  c->lin_n_value = req_int(t, "linear_num_value_heads");
  c->n_experts = req_int(t, "num_experts");
  c->top_k = req_int(t, "num_experts_per_tok");
  c->moe_inter = req_int(t, "moe_intermediate_size");
  c->shared_inter = req_int(t, "shared_expert_intermediate_size");
  c->norm_topk = json_bool(t, "norm_topk_prob", true);
  c->max_positions = req_int(t, "max_position_embeddings");
  require_flag(t, "rotary_interleaved", false, false);
  require_flag(t, "use_qk_norm", true, true);
  require_flag(t, "attn_output_gate", true, true);
  require_flag(t, "layernorm_zero_centered_gamma", true, true);
  require_flag(t, "shared_expert_gate", true, true);
  JNode *types = json_get(t, "layer_types");
  if (!types || types->type != J_ARR || (int)types->count != c->n_layers)
    die("config: layer_types must list every layer");
  for (int i = 0; i < c->n_layers; i++) {
    const char *lt = types->items[i]->str;
    if (!lt) die("config: bad layer type");
    if (!strcmp(lt, "linear_attention"))
      c->is_linear[i] = true;
    else if (!strcmp(lt, "full_attention"))
      c->is_linear[i] = false;
    else
      die("config: unknown layer type %s", lt);
  }
  JNode *mlp = json_get(t, "mlp_layer_types");
  for (size_t i = 0; mlp && i < mlp->count; i++)
    if (strcmp(mlp->items[i]->str, "moe")) die("config: every mlp layer must be moe");

  JNode *a = json_get(root, "audio_decoder_config");
  const char *dtype = json_str(a, "decoder_type");
  if (!dtype || strcmp(dtype, "codebook_mtp_shared"))
    die("config: decoder_type must be codebook_mtp_shared");
  require_flag(a, "rotary_interleaved", true, true);
  require_flag(a, "attention_output_gate", true, true);
  require_flag(a, "layernorm_zero_centered_gamma", true, true);
  require_flag(a, "use_moe", false, false);
  require_flag(a, "mtp_use_swa", false, false);
  require_flag(a, "attention_qkv_bias", false, false);
  require_flag(a, "attention_o_bias", false, false);
  const char *placement = json_str(a, "vq_feature_placement");
  if (!placement || strcmp(placement, "backbone_partial"))
    die("config: vq_feature_placement must be backbone_partial");
  c->t_dim = req_int(a, "dim");
  c->t_inter = req_int(a, "intermediate_size");
  c->t_heads = req_int(a, "n_head");
  c->t_kv_heads = req_int(a, "n_local_heads");
  c->t_head_dim = req_int(a, "head_dim");
  c->t_rotary_dim = (int)(c->t_head_dim * req_float(a, "rotary_percent"));
  c->t_rope_base = req_float(a, "rope_base");
  c->t_eps = req_float(a, "norm_eps");
  c->code_vocab = req_int(a, "vocab_size");
  c->n_codebooks = req_int(a, "num_codebooks");
  c->n_shared = req_int(a, "shared_enrichment_layers");
  c->vq_backbone_codebooks = req_int(a, "vq_backbone_codebooks");
  if (c->n_codebooks > MAX_CODEBOOKS || c->n_shared > MAX_CODEBOOKS)
    die("config: too many codebooks");
  if (req_int(a, "n_layer") != c->n_shared) die("config: n_layer != shared layers");
  if (c->t_dim != c->hidden) die("config: tower dim must equal hidden size");

  c->semantic_start = req_int(root, "semantic_start_token_id");
  c->semantic_end = req_int(root, "semantic_end_token_id");
  c->eos_id = req_int(root, "eos_token_id");
  if (c->semantic_end - c->semantic_start + 1 != c->code_vocab)
    die("config: semantic band must match the tower vocabulary");
  if (c->eos_id >= c->semantic_start) die("config: eos must precede the semantic band");

  JNode *experts = json_get(conv, "experts");
  c->expert_bits = req_int(experts, "bits");
  c->expert_group = req_int(experts, "group_size");
  json_free(doc);
}

static void load_tower_layer(WeightStore *ws, TowerLayer *t, const char *prefix) {
  t->embedding = weights_get(ws, "%s.embedding.weight", prefix);
  t->enorm = weights_get(ws, "%s.enorm.weight", prefix);
  t->hnorm = weights_get(ws, "%s.hnorm.weight", prefix);
  t->final_norm = weights_get(ws, "%s.final_layernorm.weight", prefix);
  t->eh_proj = weights_get(ws, "%s.eh_proj.weight", prefix);
  t->attn_norm = weights_get(ws, "%s.block.attention_norm.weight", prefix);
  t->ffn_norm = weights_get(ws, "%s.block.ffn_norm.weight", prefix);
  t->q_norm = weights_get(ws, "%s.block.attention.q_norm.weight", prefix);
  t->k_norm = weights_get(ws, "%s.block.attention.k_norm.weight", prefix);
  t->wqkv = weights_get(ws, "%s.block.attention.wqkv.weight", prefix);
  t->wo = weights_get(ws, "%s.block.attention.wo.weight", prefix);
  t->w2 = weights_get(ws, "%s.block.feed_forward.w2.weight", prefix);
  t->w1 = weights_get(ws, "%s.block.feed_forward.w1.weight", prefix);
  t->w3 = weights_get(ws, "%s.block.feed_forward.w3.weight", prefix);
}

static void push(mlx_array **list, size_t *n, size_t *cap, mlx_array a) {
  if (*n == *cap) {
    *cap = *cap ? *cap * 2 : 1024;
    *list = realloc(*list, *cap * sizeof(mlx_array));
  }
  (*list)[(*n)++] = a;
}

static void load_tokenizer(Model *m, const char *dir) {
  const Config *c = &m->cfg;
  char path[4096];
  snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
  m->tok = tok_load(path);
  if (!m->tok) die("cannot load %s", path);
  char name[64];
  snprintf(name, sizeof(name), "<|semantic:%d|>", 0);
  if (tok_token_to_id(m->tok, name) != c->semantic_start) die("tokenizer/config mismatch");
  snprintf(name, sizeof(name), "<|semantic:%d|>", c->code_vocab - 1);
  if (tok_token_to_id(m->tok, name) != c->semantic_end) die("tokenizer/config mismatch");
  if (tok_token_to_id(m->tok, "<|im_end|>") != c->eos_id) die("tokenizer/config mismatch");
}

Model *model_load_light(const char *dir) {
  mlxu_init();
  Model *m = calloc(1, sizeof(Model));
  parse_config(dir, &m->cfg);
  load_tokenizer(m, dir);
  return m;
}

Model *model_load(const char *dir) {
  mlxu_init();
  Model *m = calloc(1, sizeof(Model));
  Config *c = &m->cfg;
  parse_config(dir, c);
  load_tokenizer(m, dir);

  {
    int32_t up[64] = {0}, down[64];
    if (c->top_k > 64) die("config: top_k too large");
    for (int i = 0; i < c->top_k; i++) down[i] = i;
    size_t mark = arena_mark();
    m->moe_lhs_up = own(m_astype(m_from_data(up, SH(1, c->top_k), MLX_INT32), MLX_UINT32));
    m->moe_lhs_down = own(m_astype(m_from_data(down, SH(1, c->top_k), MLX_INT32), MLX_UINT32));
    m_eval(m->moe_lhs_up);
    m_eval(m->moe_lhs_down);
    arena_release(mark);
  }

  WeightStore *ws = m->weights = weights_open_index(dir);
  mlx_array *all = NULL;
  size_t n = 0, cap = 0;
#define GET(field, ...)                       \
  do {                                        \
    field = weights_get(ws, __VA_ARGS__);     \
    push(&all, &n, &cap, field);              \
  } while (0)

  GET(m->embed_tokens, "embed_tokens.weight");
  GET(m->final_norm, "norm.weight");
  GET(m->lm_head_audio, "lm_head_audio.weight");
  for (int i = 0; i < c->n_layers; i++) {
    BackboneLayer *L = &m->layers[i];
    GET(L->input_norm, "layers.%d.input_layernorm.weight", i);
    GET(L->post_norm, "layers.%d.post_attention_layernorm.weight", i);
    if (c->is_linear[i]) {
      GET(L->in_qkv, "layers.%d.linear_attn.in_proj_qkv.weight", i);
      GET(L->in_z, "layers.%d.linear_attn.in_proj_z.weight", i);
      GET(L->in_b, "layers.%d.linear_attn.in_proj_b.weight", i);
      GET(L->in_a, "layers.%d.linear_attn.in_proj_a.weight", i);
      GET(L->conv_w, "layers.%d.linear_attn.conv1d.weight", i);
      GET(L->dt_bias, "layers.%d.linear_attn.dt_bias", i);
      GET(L->a_log, "layers.%d.linear_attn.A_log", i);
      GET(L->gdn_norm, "layers.%d.linear_attn.norm.weight", i);
      GET(L->out_proj, "layers.%d.linear_attn.out_proj.weight", i);
    } else {
      GET(L->q_proj, "layers.%d.self_attn.q_proj.weight", i);
      GET(L->k_proj, "layers.%d.self_attn.k_proj.weight", i);
      GET(L->v_proj, "layers.%d.self_attn.v_proj.weight", i);
      GET(L->o_proj, "layers.%d.self_attn.o_proj.weight", i);
      GET(L->q_norm, "layers.%d.self_attn.q_norm.weight", i);
      GET(L->k_norm, "layers.%d.self_attn.k_norm.weight", i);
    }
    GET(L->router, "layers.%d.mlp.gate.weight", i);
    GET(L->sh_gate, "layers.%d.mlp.shared_expert.gate_proj.weight", i);
    GET(L->sh_up, "layers.%d.mlp.shared_expert.up_proj.weight", i);
    GET(L->sh_expert_gate, "layers.%d.mlp.shared_expert_gate.weight", i);
    GET(L->sh_down, "layers.%d.mlp.shared_expert.down_proj.weight", i);
    GET(L->gu_w, "layers.%d.mlp.experts.gate_up_proj.weight", i);
    GET(L->gu_s, "layers.%d.mlp.experts.gate_up_proj.scales", i);
    GET(L->gu_b, "layers.%d.mlp.experts.gate_up_proj.biases", i);
    GET(L->down_w, "layers.%d.mlp.experts.down_proj.weight", i);
    GET(L->down_s, "layers.%d.mlp.experts.down_proj.scales", i);
    GET(L->down_b, "layers.%d.mlp.experts.down_proj.biases", i);
  }
  GET(m->codebook_embeddings, "tower.codebook_embeddings.weight");
  for (int i = 0; i < c->n_shared; i++) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "tower.shared.%d", i);
    load_tower_layer(ws, &m->shared[i], prefix);
  }
  for (int i = 0; i < c->n_codebooks - 1; i++) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "tower.residual.%d", i);
    load_tower_layer(ws, &m->residual[i], prefix);
    GET(m->heads[i], "tower.heads.%d.weight", i);
  }
#undef GET
  /* Materialize every weight up front so the first request does not pay for it. */
  for (int i = 0; i < c->n_shared; i++) {
    TowerLayer *t = &m->shared[i];
    mlx_array ws_[] = {t->embedding, t->enorm, t->hnorm, t->final_norm, t->eh_proj,
                       t->attn_norm, t->ffn_norm, t->q_norm, t->k_norm, t->wqkv,
                       t->wo, t->w1, t->w2, t->w3};
    for (size_t k = 0; k < sizeof(ws_) / sizeof(ws_[0]); k++) push(&all, &n, &cap, ws_[k]);
  }
  for (int i = 0; i < c->n_codebooks - 1; i++) {
    TowerLayer *t = &m->residual[i];
    mlx_array ws_[] = {t->embedding, t->enorm, t->hnorm, t->final_norm, t->eh_proj,
                       t->attn_norm, t->ffn_norm, t->q_norm, t->k_norm, t->wqkv,
                       t->wo, t->w1, t->w2, t->w3};
    for (size_t k = 0; k < sizeof(ws_) / sizeof(ws_[0]); k++) push(&all, &n, &cap, ws_[k]);
  }
  double t0 = now_seconds();
  /* one evaluation of every weight reads the files faster than batches */
  m_eval_many(all, n);
  log_info("loaded %zu tensors in %.1fs", n, now_seconds() - t0);
  free(all);
  /* every weight has its own handle; the file maps are no longer needed */
  weights_free(m->weights);
  m->weights = NULL;
  mlx_clear_cache();
  return m;
}

static void free_tower_layer(TowerLayer *t) {
  mlx_array *fields[] = {&t->embedding, &t->enorm, &t->hnorm, &t->final_norm,
                         &t->eh_proj, &t->attn_norm, &t->ffn_norm, &t->q_norm,
                         &t->k_norm, &t->wqkv, &t->wo, &t->w1, &t->w2, &t->w3,
                         &t->k_cache, &t->v_cache};
  for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++) release(fields[i]);
}

void model_free(Model *m) {
  if (!m) return;
  for (int i = 0; i < m->cfg.n_layers; i++) {
    BackboneLayer *L = &m->layers[i];
    mlx_array *fields[] = {&L->input_norm, &L->post_norm, &L->in_qkv, &L->in_z, &L->in_b,
                           &L->in_a, &L->conv_w, &L->dt_bias, &L->a_log, &L->gdn_norm,
                           &L->out_proj, &L->q_proj, &L->k_proj, &L->v_proj, &L->o_proj,
                           &L->q_norm, &L->k_norm, &L->router, &L->sh_gate, &L->sh_up,
                           &L->sh_expert_gate, &L->sh_down, &L->gu_w, &L->gu_s, &L->gu_b,
                           &L->down_w, &L->down_s, &L->down_b};
    for (size_t k = 0; k < sizeof(fields) / sizeof(fields[0]); k++) release(fields[k]);
    release(&m->cache[i].k);
    release(&m->cache[i].v);
    release(&m->cache[i].conv);
    release(&m->cache[i].rec);
  }
  for (int i = 0; i < m->cfg.n_shared; i++) free_tower_layer(&m->shared[i]);
  for (int i = 0; i < m->cfg.n_codebooks - 1; i++) {
    free_tower_layer(&m->residual[i]);
    release(&m->heads[i]);
  }
  release(&m->embed_tokens);
  release(&m->final_norm);
  release(&m->lm_head_audio);
  release(&m->codebook_embeddings);
  release(&m->moe_lhs_up);
  release(&m->moe_lhs_down);
  Compiled *compiled[] = {&m->backbone_compiled.gdn,     &m->backbone_compiled.attn_pre,
                          &m->backbone_compiled.attn_post, &m->backbone_compiled.attn,
                          &m->tower_compiled.pre,          &m->tower_compiled.post,
                          &m->tower_compiled.post_head,    &m->tower_compiled.layer,
                          &m->tower_compiled.layer_head};
  for (size_t i = 0; i < sizeof(compiled) / sizeof(compiled[0]); i++)
    if (compiled[i]->ready) mlx_closure_free(compiled[i]->closure);
  weights_free(m->weights);
  tok_free(m->tok);
  free(m);
  mlx_clear_cache(); /* hand the freed weight buffers back to the system */
}

void model_reset_caches(Model *m, int capacity) {
  const Config *c = &m->cfg;
  size_t mark = arena_mark();
  if (capacity > m->capacity) {
    /* Grow in steps so a run of requests does not reallocate every time. */
    int rounded = ((capacity + 1023) / 1024) * 1024;
    for (int i = 0; i < c->n_layers; i++) {
      if (c->is_linear[i]) continue;
      assign(&m->cache[i].k,
             m_zeros(SH(1, c->n_kv_heads, rounded, c->head_dim), MLX_BFLOAT16));
      assign(&m->cache[i].v,
             m_zeros(SH(1, c->n_kv_heads, rounded, c->head_dim), MLX_BFLOAT16));
    }
    TowerLayer *towers[2 * MAX_CODEBOOKS];
    int count = 0;
    for (int i = 0; i < c->n_shared; i++) towers[count++] = &m->shared[i];
    for (int i = 0; i < c->n_codebooks - 1; i++) towers[count++] = &m->residual[i];
    for (int i = 0; i < count; i++) {
      assign(&towers[i]->k_cache,
             m_zeros(SH(1, c->t_kv_heads, rounded, c->t_head_dim), MLX_BFLOAT16));
      assign(&towers[i]->v_cache,
             m_zeros(SH(1, c->t_kv_heads, rounded, c->t_head_dim), MLX_BFLOAT16));
    }
    m->capacity = rounded;
  }
  int conv_dim = 2 * c->lin_n_key * c->lin_key_dim + c->lin_n_value * c->lin_value_dim;
  for (int i = 0; i < c->n_layers; i++) {
    if (!c->is_linear[i]) continue;
    assign(&m->cache[i].conv,
           m_zeros(SH(1, c->conv_kernel - 1, conv_dim), MLX_BFLOAT16));
    assign(&m->cache[i].rec,
           m_zeros(SH(1, c->lin_n_value, c->lin_value_dim, c->lin_key_dim), MLX_FLOAT32));
  }
  mlx_array arrays[4 * MAX_LAYERS + 4 * MAX_CODEBOOKS];
  size_t count = model_cache_arrays(m, arrays, sizeof(arrays) / sizeof(arrays[0]));
  m_eval_many(arrays, count);
  arena_release(mark);
}

size_t model_cache_arrays(Model *m, mlx_array *out, size_t cap) {
  const Config *c = &m->cfg;
  size_t n = 0;
  for (int i = 0; i < c->n_layers && n + 2 <= cap; i++) {
    if (c->is_linear[i]) {
      out[n++] = m->cache[i].conv;
      out[n++] = m->cache[i].rec;
    } else {
      out[n++] = m->cache[i].k;
      out[n++] = m->cache[i].v;
    }
  }
  for (int i = 0; i < c->n_shared && n + 2 <= cap; i++) {
    out[n++] = m->shared[i].k_cache;
    out[n++] = m->shared[i].v_cache;
  }
  for (int i = 0; i < c->n_codebooks - 1 && n + 2 <= cap; i++) {
    out[n++] = m->residual[i].k_cache;
    out[n++] = m->residual[i].v_cache;
  }
  return n;
}
