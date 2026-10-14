/* Top-k / top-p sampling with the semantics of tower.py sample_top_k_top_p:
 * the token whose cumulative probability crosses top_p is dropped (except the
 * highest-ranked one), and the draw is argmax(p / Exp(1) noise). */
#include <math.h>

#include "model.h"

#define MIN_TEMPERATURE 1e-5f

mlx_array sample_top_k_top_p(mlx_array logits, float temperature, int top_k, float top_p) {
  if (temperature < MIN_TEMPERATURE) temperature = MIN_TEMPERATURE;
  int V = m_dim(logits, -1);
  int N = m_dim(logits, 0);
  mlx_array x = m_div(m_astype(logits, MLX_FLOAT32), m_float(temperature));
  mlx_array ids, values;
  /* Only the top_k ranks can survive, so sort just those; the softmax for the
   * cumulative mass still normalizes over the whole row. */
  int k = top_k < V ? top_k : V;
  if (k < V) {
    mlx_array part = m_slice_axis(m_argpartition(m_neg(x), k - 1, -1), -1, 0, k);
    mlx_array part_vals = m_take_along_axis(x, part, -1);
    mlx_array order = m_argsort(m_neg(part_vals), -1);
    ids = m_take_along_axis(part, order, -1);
    values = m_take_along_axis(part_vals, order, -1);
  } else {
    ids = m_argsort(m_neg(x), -1);
    values = m_take_along_axis(x, ids, -1);
  }
  mlx_array lse = m_max(x, -1, true);
  lse = m_add(lse, m_log(m_sum(m_exp(m_sub(x, lse)), -1, true)));
  mlx_array cumulative = m_cumsum(m_exp(m_sub(values, lse)), -1);
  mlx_array ranks = m_arange(0, k, 1, MLX_INT32);
  mlx_array dropped = m_greater(cumulative, m_float(top_p));
  dropped = m_logical_and(dropped, m_greater(ranks, m_int(0)));
  mlx_array masked = m_where(dropped, m_float(-INFINITY), values);
  mlx_array probs = m_softmax(masked, -1);
  /* Exp(1) noise: -log(u), u in (0, 1] */
  mlx_array u = m_sub(m_float(1.0f), m_uniform(SH(N, k), MLX_FLOAT32));
  mlx_array noise = m_neg(m_log(u));
  mlx_array choice = m_argmax(m_div(probs, noise), -1, true);
  return m_reshape(m_take_along_axis(ids, choice, -1), SH(N));
}
