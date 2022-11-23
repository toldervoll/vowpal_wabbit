// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "vw/core/reductions/ftrl.h"

#include "vw/core/debug_log.h"
#include "vw/core/correctedMath.h"
#include "vw/core/crossplat_compat.h"
#include "vw/core/label_parser.h"
#include "vw/core/learner.h"
#include "vw/core/loss_functions.h"
#include "vw/core/parse_regressor.h"
#include "vw/core/parser.h"
#include "vw/core/reductions/gd.h"
#include "vw/core/setup_base.h"
#include "vw/core/shared_data.h"
#include "vw/io/logger.h"

#include <cfloat>
#include <cmath>
#include <string>

#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::ftrl

using namespace VW::LEARNER;
using namespace VW::config;
using namespace VW::math;

#define W_XT 0  // current parameter
#define W_ZT 1  // in proximal is "accumulated z(t) = z(t-1) + g(t) + sigma*w(t)", in general is the dual weight vector
#define W_G2 2  // accumulated gradient information
#define W_MX 3  // maximum absolute value
#define W_WE 4  // Wealth
#define W_MG 5  // maximum gradient

namespace
{
struct uncertainty
{
  float pred;
  float score;
  ftrl& b;
  uncertainty(ftrl& ftrlb) : b(ftrlb)
  {
    pred = 0;
    score = 0;
  }
};

inline void predict_with_confidence(uncertainty& d, const float fx, float& fw)
{
  float* w = &fw;
  d.pred += w[W_XT] * fx;
  float sqrtf_ng2 = sqrtf(w[W_G2]);
  float uncertain = ((d.b.data.ftrl_beta + sqrtf_ng2) / d.b.data.ftrl_alpha + d.b.data.l2_lambda);
  d.score += (1 / uncertain) * sign(fx);
}
float sensitivity(ftrl& b, base_learner& /* base */, VW::example& ec)
{
  uncertainty uncetain(b);
  GD::foreach_feature<uncertainty, predict_with_confidence>(*(b.all), ec, uncetain);
  return uncetain.score;
}

template <bool audit>
void predict(ftrl& b, base_learner&, VW::example& ec)
{
  size_t num_features_from_interactions = 0;
  ec.partial_prediction = GD::inline_predict(*b.all, ec, num_features_from_interactions);
  ec.num_features_from_interactions = num_features_from_interactions;
  ec.pred.scalar = GD::finalize_prediction(b.all->sd, b.all->logger, ec.partial_prediction);
  if (audit) { GD::print_audit_features(*(b.all), ec); }
}

template <bool audit>
void multipredict(ftrl& b, base_learner&, VW::example& ec, size_t count, size_t step, VW::polyprediction* pred,
    bool finalize_predictions)
{
  VW::workspace& all = *b.all;
  for (size_t c = 0; c < count; c++)
  {
    const auto& simple_red_features = ec._reduction_features.template get<simple_label_reduction_features>();
    pred[c].scalar = simple_red_features.initial;
  }
  size_t num_features_from_interactions = 0;
  if (b.all->weights.sparse)
  {
    GD::multipredict_info<sparse_parameters> mp = {
        count, step, pred, all.weights.sparse_weights, static_cast<float>(all.sd->gravity)};
    GD::foreach_feature<GD::multipredict_info<sparse_parameters>, uint64_t, GD::vec_add_multipredict>(
        all, ec, mp, num_features_from_interactions);
  }
  else
  {
    GD::multipredict_info<dense_parameters> mp = {
        count, step, pred, all.weights.dense_weights, static_cast<float>(all.sd->gravity)};
    GD::foreach_feature<GD::multipredict_info<dense_parameters>, uint64_t, GD::vec_add_multipredict>(
        all, ec, mp, num_features_from_interactions);
  }
  ec.num_features_from_interactions = num_features_from_interactions;
  if (all.sd->contraction != 1.)
  {
    for (size_t c = 0; c < count; c++) { pred[c].scalar *= static_cast<float>(all.sd->contraction); }
  }
  if (finalize_predictions)
  {
    for (size_t c = 0; c < count; c++) { pred[c].scalar = GD::finalize_prediction(all.sd, all.logger, pred[c].scalar); }
  }
  if (audit)
  {
    for (size_t c = 0; c < count; c++)
    {
      ec.pred.scalar = pred[c].scalar;
      GD::print_audit_features(all, ec);
      ec.ft_offset += static_cast<uint64_t>(step);
    }
    ec.ft_offset -= static_cast<uint64_t>(step * count);
  }
}

void inner_update_proximal(ftrl_update_data& d, float x, float& wref)
{
  float* w = &wref;
  float gradient = d.update * x;
  float ng2 = w[W_G2] + gradient * gradient;
  float sqrt_ng2 = sqrtf(ng2);
  float sqrt_wW_G2 = sqrtf(w[W_G2]);
  float sigma = (sqrt_ng2 - sqrt_wW_G2) / d.ftrl_alpha;
  w[W_ZT] += gradient - sigma * w[W_XT];
  w[W_G2] = ng2;
  sqrt_wW_G2 = sqrt_ng2;
  float flag = sign(w[W_ZT]);
  float fabs_zt = w[W_ZT] * flag;
  if (fabs_zt <= d.l1_lambda) { w[W_XT] = 0.; }
  else
  {
    float step = 1 / (d.l2_lambda + (d.ftrl_beta + sqrt_wW_G2) / d.ftrl_alpha);
    w[W_XT] = step * flag * (d.l1_lambda - fabs_zt);
  }
}

void inner_update_pistol_state_and_predict(ftrl_update_data& d, float x, float& wref)
{
  float* w = &wref;

  float fabs_x = std::fabs(x);
  if (fabs_x > w[W_MX]) { w[W_MX] = fabs_x; }

  float squared_theta = w[W_ZT] * w[W_ZT];
  float tmp = 1.f / (d.ftrl_alpha * w[W_MX] * (w[W_G2] + w[W_MX]));
  w[W_XT] = std::sqrt(w[W_G2]) * d.ftrl_beta * w[W_ZT] * correctedExp(squared_theta / 2.f * tmp) * tmp;

  d.predict += w[W_XT] * x;
}

void inner_update_pistol_post(ftrl_update_data& d, float x, float& wref)
{
  float* w = &wref;
  float gradient = d.update * x;

  w[W_ZT] += -gradient;
  w[W_G2] += std::fabs(gradient);
}

// Coin betting vectors
// W_XT 0  current parameter
// W_ZT 1  sum negative gradients
// W_G2 2  sum of absolute value of gradients
// W_MX 3  maximum absolute value
// W_WE 4  Wealth
// W_MG 5  Maximum Lipschitz constant
void inner_coin_betting_predict(ftrl_update_data& d, float x, float& wref)
{
  float* w = &wref;
  float w_mx = w[W_MX];
  float w_xt = 0.0;

  float fabs_x = std::fabs(x);
  if (fabs_x > w_mx) { w_mx = fabs_x; }

  // COCOB update without sigmoid
  if (w[W_MG] * w_mx > 0)
  { w_xt = ((d.ftrl_alpha + w[W_WE]) / (w[W_MG] * w_mx * (w[W_MG] * w_mx + w[W_G2]))) * w[W_ZT]; }

  std::cout << std::hexfloat << "==predict: " << d.predict << "==xt: " << w_xt << ", x: " << x <<std::endl;
  d.predict += w_xt * x;
  if (w_mx > 0)
  {
    const float x_normalized = x / w_mx;
    d.normalized_squared_norm_x += x_normalized * x_normalized;
  }
}

void print_ftrl(ftrl& b) {
  std::cout << "[ftrl] ftrl alpha: " << b.ftrl_alpha
    << "ftrl beta: " << b.ftrl_beta
    << "no win counter: " << b.no_win_counter
    << "early stop thres: " << b.early_stop_thres
    << "ftrl size: " << b.ftrl_size
    << "total weight: " << b.total_weight
    << "normalized sum norm x: " << b.normalized_sum_norm_x
    << std::endl;

  std::cout << "[ftrl] all->sd: queries: " << b.all->sd->queries
    << "\n  "<< "example_number: " << b.all->sd->example_number
    << "\n  "<< "total_features: " << b.all->sd->total_features
    << "\n  "<< "t" << b.all->sd->t
    << "\n  "<< "weighted_labeled_examples: " << b.all->sd->weighted_labeled_examples
    << "\n  "<< "old_weighted_labeled_examples: " << b.all->sd->old_weighted_labeled_examples
    << "\n  "<< "weighted_unlabeled_examples: " << b.all->sd->weighted_unlabeled_examples
    << "\n  "<< "weighted_labels: " << b.all->sd->weighted_labels
    << "\n  "<< "sum_loss: " << b.all->sd->sum_loss
    << "\n  "<< "sum_loss_since_last_dump: " << b.all->sd->sum_loss_since_last_dump
    << "\n  "<< "dump_interval: " << b.all->sd->dump_interval
    << "\n  "<< "gravity: " << b.all->sd->gravity
    << "\n  "<< "contraction: " <<b.all->sd->contraction
    << "\n  "<< "min_label: " << b.all->sd->min_label
    << "\n  "<< "max_label: " << b.all->sd->max_label
    << "\n  "<< "weighted_holdout_examples: " << b.all->sd->weighted_holdout_examples
    << "\n  "<< "weighted_holdout_examples_since_last_dump: " << b.all->sd->weighted_holdout_examples_since_last_dump
    << "\n  "<< "holdout_sum_loss_since_last_dump: " << b.all->sd->holdout_sum_loss_since_last_dump
    << "\n  "<< "holdout_sum_loss: " << b.all->sd->holdout_sum_loss
    << "\n  "<< "holdout_best_loss: " << b.all->sd->holdout_best_loss
    << "\n  "<< "weighted_holdout_examples_since_last_pass: " << b.all->sd->weighted_holdout_examples_since_last_pass
    << "\n  "<< "holdout_sum_loss_since_last_pass: " << b.all->sd->holdout_sum_loss_since_last_pass
    << "\n  "<< "holdout_best_pass: " << b.all->sd->holdout_best_pass
    << "\n  "<< "report_multiclass_log_loss: " << b.all->sd->report_multiclass_log_loss
    << "\n  "<< "multiclass_log_loss: " << b.all->sd->multiclass_log_loss
    << "\n  "<< "holdout_multiclass_log_loss: " << b.all->sd->holdout_multiclass_log_loss
    << "\n  "<< "is_more_than_two_labels_observed: " << b.all->sd->is_more_than_two_labels_observed
    << "\n  "<< "first_observed_label: " << b.all->sd->first_observed_label
    << "\n  "<< "second_observed_label: " << b.all->sd->second_observed_label
    << std::endl;
}
using separate_weights_vector = std::vector<std::tuple<size_t, float, float, float, float, float, float>>;

void print_separate_weights(separate_weights_vector weights_vector) {
  std::cout << "[ftrl] weights: " << std::endl;
  std::cout << std::hexfloat;
  for (auto& weights:weights_vector) {
    std::cout << std::get<0>(weights) << " "
      << std::get<1>(weights) << " "
      << std::get<2>(weights) << " "
      << std::get<3>(weights) << " "
      << std::get<4>(weights) << " "
      << std::get<5>(weights) << " "
      << std::get<6>(weights) << " "
    << std::endl;
  }
}

separate_weights_vector get_separate_weights(VW::workspace* vw) {
  auto& weights = vw->weights.dense_weights;
  auto iter = weights.begin();
  auto end = weights.end();

  separate_weights_vector weights_vector;

  while(iter < end) {
    bool non_zero = false;
    for (int i=0; i < 6; i++) {
      if (*iter[i] != 0.f) {
        non_zero = true;
      }
    }

    if (non_zero) {
      weights_vector.emplace_back(iter.index_without_stride(), *iter[0], *iter[1], *iter[2], *iter[3], *iter[4], *iter[5]);
    }
    ++iter;
  }

  return weights_vector;
}


void print_ftrl_update_data(ftrl_update_data& d) {
  std::cout << "[ftrl update data] d.update: " << d.update 
    << ", ftrl_alpha: " << d.ftrl_alpha
    << ", ftrl_beta: " << d.ftrl_beta
    << ", l1_lambda: " << d.l1_lambda
    << ", l2 lambda: " << d.l2_lambda
    << ", predict: " << d.predict
    << ", normalized squared norm x: " << d.normalized_squared_norm_x
    << ", average squared norm x: " << d.average_squared_norm_x
    << std::endl;
}

void print_w(float* w) {
  std::cout << "w[0]: " << w[0] << ", "
    << "w[1]: " << w[1] << ", "
    << "w[2]: " << w[2] << ", "
    << "w[3]: " << w[3] << ", "
    << "w[4]: " << w[4] << ", "
    << "w[5]: " << w[5] << std::endl; 
}

void inner_coin_betting_update_after_prediction(ftrl_update_data& d, float x, float& wref)
{
  // print_ftrl_update_data(d);
  // std::cout << "[ftrl] x: " << x << std::endl;

  float* w = &wref;
  // std::cout << "[ftrl] before: " << std::endl;
  // print_w(w);

  float fabs_x = std::fabs(x);
  float gradient = d.update * x;

  if (fabs_x > w[W_MX]) { w[W_MX] = fabs_x; }

  float fabs_gradient = std::fabs(d.update);
  if (fabs_gradient > w[W_MG]) { w[W_MG] = fabs_gradient > d.ftrl_beta ? fabs_gradient : d.ftrl_beta; }

  // COCOB update without sigmoid.
  // If a new Lipschitz constant and/or magnitude of x is found, the w is
  // recalculated and used in the update of the wealth below.
  if (w[W_MG] * w[W_MX] > 0)
  { w[W_XT] = ((d.ftrl_alpha + w[W_WE]) / (w[W_MG] * w[W_MX] * (w[W_MG] * w[W_MX] + w[W_G2]))) * w[W_ZT]; }
  else
  {
    w[W_XT] = 0;
  }

  w[W_ZT] += -gradient;
  w[W_G2] += std::fabs(gradient);
  w[W_WE] += (-gradient * w[W_XT]);


  std::cout << std::hexfloat << "\t [ftrl] w[4]: " << w[W_WE] << ", update: " << d.update << ", x: " << x << ", gradient: " << gradient << std::endl;

  w[W_XT] /= d.average_squared_norm_x;

  std::cout << "[ftrl] after: " << std::endl;
  print_w(w);
}

void coin_betting_predict(ftrl& b, base_learner&, VW::example& ec)
{
  VW_DBG(ec) << "coin_betting_predict.predict() ex#=" << ec.example_counter << ", offset=" << ec.ft_offset << std::endl;
  b.data.predict = 0;
  b.data.normalized_squared_norm_x = 0;

  // std::cout << "[ftrl] coin betting predict before: " << std::endl;
  // print_ftrl(b);
  // print_ftrl_update_data(b.data);
  size_t num_features_from_interactions = 0;
  separate_weights_vector sw =  get_separate_weights(b.all);
  print_separate_weights(sw);
  GD::foreach_feature<ftrl_update_data, inner_coin_betting_predict>(*b.all, ec, b.data, num_features_from_interactions);
  ec.num_features_from_interactions = num_features_from_interactions;
  // std::cout << "[ftrl] coin betting predict after: " << std::endl;
  // print_ftrl(b);
  // print_ftrl_update_data(b.data);

  b.normalized_sum_norm_x += (static_cast<double>(ec.weight)) * b.data.normalized_squared_norm_x;
  b.total_weight += ec.weight;
  b.data.average_squared_norm_x = (static_cast<float>((b.normalized_sum_norm_x + 1e-6) / b.total_weight));

  ec.partial_prediction = b.data.predict / b.data.average_squared_norm_x;
  std::cout << std::hexfloat << "[ftrl] partial_p: " << ec.partial_prediction << ", predict: " << b.data.predict << ", avg norm x: " << b.data.average_squared_norm_x << std::endl;
  ec.pred.scalar = GD::finalize_prediction(b.all->sd, b.all->logger, ec.partial_prediction);

  VW_DBG(ec) << "coin_betting_predict.predict() " << VW::debug::scalar_pred_to_string(ec) << VW::debug::features_to_string(ec)
             << std::endl;

  // std::cout << "[ftrl] coin_betting_predict.predict() " << VW::debug::features_to_string(ec)
  //            << std::endl;

  VW_DBG(ec)  << "coin_betting_predict.predict() " << b.data.predict << ", "
    << "normalized_sum_norm_x: " << b.normalized_sum_norm_x << ", "
    << "total weight: " <<  b.total_weight << ", "
    << "avg squared norm: " << b.data.average_squared_norm_x << ", "
    << "partial pred: " << ec.partial_prediction << ", "
    << "pred.scalar: " << ec.pred.scalar << std::endl;
}

void update_state_and_predict_pistol(ftrl& b, base_learner&, VW::example& ec)
{
  b.data.predict = 0;

  size_t num_features_from_interactions = 0;
  GD::foreach_feature<ftrl_update_data, inner_update_pistol_state_and_predict>(
      *b.all, ec, b.data, num_features_from_interactions);
  ec.num_features_from_interactions = num_features_from_interactions;

  ec.partial_prediction = b.data.predict;
  ec.pred.scalar = GD::finalize_prediction(b.all->sd, b.all->logger, ec.partial_prediction);
}

void update_after_prediction_proximal(ftrl& b, VW::example& ec)
{
  b.data.update = b.all->loss->first_derivative(b.all->sd, ec.pred.scalar, ec.l.simple.label) * ec.weight;
  GD::foreach_feature<ftrl_update_data, inner_update_proximal>(*b.all, ec, b.data);
}

void update_after_prediction_pistol(ftrl& b, VW::example& ec)
{
  b.data.update = b.all->loss->first_derivative(b.all->sd, ec.pred.scalar, ec.l.simple.label) * ec.weight;
  GD::foreach_feature<ftrl_update_data, inner_update_pistol_post>(*b.all, ec, b.data);
}

void coin_betting_update_after_prediction(ftrl& b, VW::example& ec)
{
  // print_ftrl(b);
  // std::cout << "[coin betting] b.data before: " << std::endl; 
  // print_ftrl_update_data(b.data);
  b.data.update = b.all->loss->first_derivative(b.all->sd, ec.pred.scalar, ec.l.simple.label) * ec.weight;
  // std::cout << "[coin betting] b.data after: " << std::endl;
  // print_ftrl_update_data(b.data);
  GD::foreach_feature<ftrl_update_data, inner_coin_betting_update_after_prediction>(*b.all, ec, b.data);
  // std::cout << "[coin betting] b.data after foreach: " << std::endl;
  // print_ftrl_update_data(b.data);
}

// NO_SANITIZE_UNDEFINED needed in learn functions because
// base_learner& base might be a reference created from nullptr
template <bool audit>
void NO_SANITIZE_UNDEFINED learn_proximal(ftrl& a, base_learner& base, VW::example& ec)
{
  // predict with confidence
  predict<audit>(a, base, ec);

  // update state based on the prediction
  update_after_prediction_proximal(a, ec);
}

template <bool audit>
void NO_SANITIZE_UNDEFINED learn_pistol(ftrl& a, base_learner& base, VW::example& ec)
{
  // update state based on the example and predict
  update_state_and_predict_pistol(a, base, ec);
  if (audit) { GD::print_audit_features(*(a.all), ec); }
  // update state based on the prediction
  update_after_prediction_pistol(a, ec);
}

template <bool audit>
void NO_SANITIZE_UNDEFINED learn_coin_betting(ftrl& a, base_learner& base, VW::example& ec)
{
  // assert(a.all->sd->multiclass_log_loss == 7);
  // update state based on the example and predict
  coin_betting_predict(a, base, ec);
  if (audit) { GD::print_audit_features(*(a.all), ec); }
  // update state based on the prediction
  coin_betting_update_after_prediction(a, ec);
}

void save_load(ftrl& b, io_buf& model_file, bool read, bool text)
{
  VW::workspace* all = b.all;
  if (read) { initialize_regressor(*all); }

  if (model_file.num_files() != 0)
  {
    bool resume = all->save_resume;
    std::stringstream msg;
    msg << ":" << resume << "\n";
    bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&resume), sizeof(resume), read, msg, text);

    if (resume)
    {
      GD::save_load_online_state(
          *all, model_file, read, text, b.total_weight, b.normalized_sum_norm_x, nullptr, b.ftrl_size);
    }
    else
    {
      GD::save_load_regressor(*all, model_file, read, text);
    }
  }
}

void end_pass(ftrl& g)
{
  VW::workspace& all = *g.all;

  if (!all.holdout_set_off)
  {
    if (summarize_holdout_set(all, g.no_win_counter)) { finalize_regressor(all, all.final_regressor_name); }
    if ((g.early_stop_thres == g.no_win_counter) &&
        ((all.check_holdout_every_n_passes <= 1) || ((all.current_pass % all.check_holdout_every_n_passes) == 0)))
    { set_done(all); }
  }
}
}  // namespace

base_learner* VW::reductions::ftrl_setup(VW::setup_base_i& stack_builder)
{
  options_i& options = *stack_builder.get_options();
  VW::workspace& all = *stack_builder.get_all_pointer();
  auto b = VW::make_unique<ftrl>();

  bool ftrl_option_no_not_use = false;
  bool pistol_no_not_use = false;
  bool coin_no_not_use = false;

  option_group_definition ftrl_options("[Reduction] Follow the Regularized Leader - FTRL");
  ftrl_options
      .add(make_option("ftrl", ftrl_option_no_not_use)
               .necessary()
               .keep()
               .help("FTRL: Follow the Proximal Regularized Leader"))
      .add(make_option("ftrl_alpha", b->ftrl_alpha).help("Learning rate for FTRL optimization"))
      .add(make_option("ftrl_beta", b->ftrl_beta).help("Learning rate for FTRL optimization"));

  option_group_definition pistol_options("[Reduction] Follow the Regularized Leader - Pistol");
  pistol_options
      .add(make_option("pistol", pistol_no_not_use)
               .necessary()
               .keep()
               .help("PiSTOL: Parameter-free STOchastic Learning"))
      .add(make_option("ftrl_alpha", b->ftrl_alpha).help("Learning rate for FTRL optimization"))
      .add(make_option("ftrl_beta", b->ftrl_beta).help("Learning rate for FTRL optimization"));

  option_group_definition coin_options("[Reduction] Follow the Regularized Leader - Coin");
  coin_options.add(make_option("coin", coin_no_not_use).necessary().keep().help("Coin betting optimizer"))
      .add(make_option("ftrl_alpha", b->ftrl_alpha).help("Learning rate for FTRL optimization"))
      .add(make_option("ftrl_beta", b->ftrl_beta).help("Learning rate for FTRL optimization"));

  const auto ftrl_enabled = options.add_parse_and_check_necessary(ftrl_options);
  const auto pistol_enabled = options.add_parse_and_check_necessary(pistol_options);
  const auto coin_enabled = options.add_parse_and_check_necessary(coin_options);

  if (!ftrl_enabled && !pistol_enabled && !coin_enabled) { return nullptr; }
  size_t count = 0;
  count += ftrl_enabled ? 1 : 0;
  count += pistol_enabled ? 1 : 0;
  count += coin_enabled ? 1 : 0;

  if (count != 1) { THROW("You can only use one of 'ftrl', 'pistol', or 'coin' at a time."); }

  b->all = &all;
  b->no_win_counter = 0;
  b->normalized_sum_norm_x = 0;
  b->total_weight = 0;

  std::string algorithm_name;
  void (*learn_ptr)(ftrl&, base_learner&, VW::example&) = nullptr;
  bool learn_returns_prediction = false;

  // Defaults that are specific to the mode that was chosen.
  if (ftrl_enabled)
  {
    b->ftrl_alpha = options.was_supplied("ftrl_alpha") ? b->ftrl_alpha : 0.005f;
    b->ftrl_beta = options.was_supplied("ftrl_beta") ? b->ftrl_beta : 0.1f;
    algorithm_name = "Proximal-FTRL";
    learn_ptr = all.audit || all.hash_inv ? learn_proximal<true> : learn_proximal<false>;
    all.weights.stride_shift(2);  // NOTE: for more parameter storage
    b->ftrl_size = 3;
  }
  else if (pistol_enabled)
  {
    b->ftrl_alpha = options.was_supplied("ftrl_alpha") ? b->ftrl_alpha : 1.0f;
    b->ftrl_beta = options.was_supplied("ftrl_beta") ? b->ftrl_beta : 0.5f;
    algorithm_name = "PiSTOL";
    learn_ptr = all.audit || all.hash_inv ? learn_pistol<true> : learn_pistol<false>;
    all.weights.stride_shift(2);  // NOTE: for more parameter storage
    b->ftrl_size = 4;
    learn_returns_prediction = true;
  }
  else if (coin_enabled)
  {
    b->ftrl_alpha = options.was_supplied("ftrl_alpha") ? b->ftrl_alpha : 4.0f;
    b->ftrl_beta = options.was_supplied("ftrl_beta") ? b->ftrl_beta : 1.0f;
    algorithm_name = "Coin Betting";
    learn_ptr = all.audit || all.hash_inv ? learn_coin_betting<true> : learn_coin_betting<false>;
    all.weights.stride_shift(3);  // NOTE: for more parameter storage
    b->ftrl_size = 6;
    learn_returns_prediction = true;
  }

  b->data.ftrl_alpha = b->ftrl_alpha;
  b->data.ftrl_beta = b->ftrl_beta;
  b->data.l1_lambda = b->all->l1_lambda;
  b->data.l2_lambda = b->all->l2_lambda;

  if (!all.quiet)
  {
    *(all.trace_message) << "Enabling FTRL based optimization" << std::endl;
    *(all.trace_message) << "Algorithm used: " << algorithm_name << std::endl;
    *(all.trace_message) << "ftrl_alpha = " << b->ftrl_alpha << std::endl;
    *(all.trace_message) << "ftrl_beta = " << b->ftrl_beta << std::endl;
  }

  if (!all.holdout_set_off)
  {
    all.sd->holdout_best_loss = FLT_MAX;
    b->early_stop_thres = options.get_typed_option<uint64_t>("early_terminate").value();
  }

  auto predict_ptr = (all.audit || all.hash_inv) ? predict<true> : predict<false>;
  auto multipredict_ptr = (all.audit || all.hash_inv) ? multipredict<true> : multipredict<false>;
  std::string name_addition = (all.audit || all.hash_inv) ? "-audit" : "";

  auto l = VW::LEARNER::make_base_learner(std::move(b), learn_ptr, predict_ptr,
      stack_builder.get_setupfn_name(ftrl_setup) + "-" + algorithm_name + name_addition, VW::prediction_type_t::scalar,
      VW::label_type_t::simple)
               .set_learn_returns_prediction(learn_returns_prediction)
               .set_params_per_weight(UINT64_ONE << all.weights.stride_shift())
               .set_sensitivity(sensitivity)
               .set_multipredict(multipredict_ptr)
               .set_save_load(save_load)
               .set_end_pass(end_pass)
               .build();
  return make_base(*l);
}