// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "vw/core/reductions/scorer.h"

#include "vw/common/vw_exception.h"
#include "vw/config/options.h"
#include "vw/core/correctedMath.h"
#include "vw/core/global_data.h"
#include "vw/core/learner.h"
#include "vw/core/loss_functions.h"
#include "vw/core/setup_base.h"
#include "vw/core/shared_data.h"
#include <cfloat>

#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::scorer

using namespace VW::config;

namespace
{
struct scorer
{
  scorer(VW::workspace* all) : all(all) {}
  VW::workspace* all;
};  // for set_minmax, loss

void print_sd(shared_data* sd) {
  std::cout << "[scorer] all->sd: queries: " << sd->queries
    << "\n  "<< "example_number: " << sd->example_number
    << "\n  "<< "total_features: " << sd->total_features
    << "\n  "<< "t" << sd->t
    << "\n  "<< "weighted_labeled_examples: " << sd->weighted_labeled_examples
    << "\n  "<< "old_weighted_labeled_examples: " << sd->old_weighted_labeled_examples
    << "\n  "<< "weighted_unlabeled_examples: " << sd->weighted_unlabeled_examples
    << "\n  "<< "weighted_labels: " << sd->weighted_labels
    << "\n  "<< "sum_loss: " << sd->sum_loss
    << "\n  "<< "sum_loss_since_last_dump: " << sd->sum_loss_since_last_dump
    << "\n  "<< "dump_interval: " << sd->dump_interval
    << "\n  "<< "gravity: " << sd->gravity
    << "\n  "<< "contraction: " << sd->contraction
    << "\n  "<< "min_label: " << sd->min_label
    << "\n  "<< "max_label: " << sd->max_label
    << "\n  "<< "weighted_holdout_examples: " << sd->weighted_holdout_examples
    << "\n  "<< "weighted_holdout_examples_since_last_dump: " << sd->weighted_holdout_examples_since_last_dump
    << "\n  "<< "holdout_sum_loss_since_last_dump: " << sd->holdout_sum_loss_since_last_dump
    << "\n  "<< "holdout_sum_loss: " << sd->holdout_sum_loss
    << "\n  "<< "holdout_best_loss: " << sd->holdout_best_loss
    << "\n  "<< "weighted_holdout_examples_since_last_pass: " << sd->weighted_holdout_examples_since_last_pass
    << "\n  "<< "holdout_sum_loss_since_last_pass: " << sd->holdout_sum_loss_since_last_pass
    << "\n  "<< "holdout_best_pass: " << sd->holdout_best_pass
    << "\n  "<< "report_multiclass_log_loss: " << sd->report_multiclass_log_loss
    << "\n  "<< "multiclass_log_loss: " << sd->multiclass_log_loss
    << "\n  "<< "holdout_multiclass_log_loss: " << sd->holdout_multiclass_log_loss
    << "\n  "<< "is_more_than_two_labels_observed: " << sd->is_more_than_two_labels_observed
    << "\n  "<< "first_observed_label: " << sd->first_observed_label
    << "\n  "<< "second_observed_label: " << sd->second_observed_label
    << std::endl;
}

template <bool is_learn, float (*link)(float in)>
void predict_or_learn(scorer& s, VW::LEARNER::single_learner& base, VW::example& ec)
{
  // Predict does not need set_minmax
  if (is_learn) { s.all->set_minmax(s.all->sd, ec.l.simple.label); }

  bool learn = is_learn && ec.l.simple.label != FLT_MAX && ec.weight > 0;
  if (learn) { base.learn(ec); }
  else
  {
    base.predict(ec);
  }

  if (ec.weight > 0 && ec.l.simple.label != FLT_MAX)
  { 
    // std::cout << "[scorer]: ec.pred.scalar" << ec.pred.scalar
    //   << "ec.l.simple.label: " << ec.l.simple.label
    //   << "ec.weight: " << ec.weight << std::endl;
    print_sd(s.all->sd);
    ec.loss = s.all->loss->get_loss(s.all->sd, ec.pred.scalar, ec.l.simple.label) * ec.weight; 
    std::cout << "[scorer loss]: " << ec.loss << std::endl;
  }

  ec.pred.scalar = link(ec.pred.scalar);
  VW_DBG(ec) << "ex#= " << ec.example_counter << ", offset=" << ec.ft_offset << ", lbl=" << ec.l.simple.label
             << ", pred= " << ec.pred.scalar << ", wt=" << ec.weight << ", gd.raw=" << ec.partial_prediction
             << ", loss=" << ec.loss << std::endl;
}

template <float (*link)(float in)>
inline void multipredict(scorer& /*unused*/, VW::LEARNER::single_learner& base, VW::example& ec, size_t count,
    size_t /*unused*/, VW::polyprediction* pred, bool finalize_predictions)
{
  base.multipredict(ec, 0, count, pred, finalize_predictions);  // TODO: need to thread step through???
  for (size_t c = 0; c < count; c++) { pred[c].scalar = link(pred[c].scalar); }
}

void update(scorer& s, VW::LEARNER::single_learner& base, VW::example& ec)
{
  s.all->set_minmax(s.all->sd, ec.l.simple.label);
  base.update(ec);
  VW_DBG(ec) << "ex#= " << ec.example_counter << ", offset=" << ec.ft_offset << ", lbl=" << ec.l.simple.label
             << ", pred= " << ec.pred.scalar << ", wt=" << ec.weight << ", gd.raw=" << ec.partial_prediction
             << ", loss=" << ec.loss << std::endl;
}

// y = f(x) -> [0, 1]
inline float logistic(float in) {
  return 1.f / (1.f + correctedExp(-in));
}

// http://en.wikipedia.org/wiki/Generalized_logistic_curve
// where the lower & upper asymptotes are -1 & 1 respectively
// 'glf1' stands for 'Generalized Logistic Function with [-1,1] range'
//    y = f(x) -> [-1, 1]
inline float glf1(float in) { return 2.f / (1.f + correctedExp(-in)) - 1.f; }

inline float id(float in) { return in; }
}  // namespace

VW::LEARNER::base_learner* VW::reductions::scorer_setup(VW::setup_base_i& stack_builder)
{
  options_i& options = *stack_builder.get_options();
  VW::workspace& all = *stack_builder.get_all_pointer();
  std::string link;
  option_group_definition new_options("[Reduction] Scorer");
  new_options.add(make_option("link", link)
                      .default_value("identity")
                      .keep()
                      .one_of({"identity", "logistic", "glf1", "poisson"})
                      .help("Specify the link function"));
  options.add_and_parse(new_options);

  using predict_or_learn_fn_t = void (*)(scorer&, VW::LEARNER::single_learner&, VW::example&);
  using multipredict_fn_t =
      void (*)(scorer&, VW::LEARNER::single_learner&, VW::example&, size_t, size_t, VW::polyprediction*, bool);
  multipredict_fn_t multipredict_f = multipredict<id>;
  predict_or_learn_fn_t learn_fn;
  predict_or_learn_fn_t predict_fn;
  std::string name = stack_builder.get_setupfn_name(scorer_setup);

  if (link == "identity")
  {
    learn_fn = predict_or_learn<true, id>;
    predict_fn = predict_or_learn<false, id>;
    name += "-identity";
  }
  else if (link == "logistic")
  {
    learn_fn = predict_or_learn<true, logistic>;
    predict_fn = predict_or_learn<false, logistic>;
    name += "-logistic";
    multipredict_f = multipredict<logistic>;
  }
  else if (link == "glf1")
  {
    learn_fn = predict_or_learn<true, glf1>;
    predict_fn = predict_or_learn<false, glf1>;
    name += "-glf1";
    multipredict_f = multipredict<glf1>;
  }
  else if (link == "poisson")
  {
    learn_fn = predict_or_learn<true, expf>;
    predict_fn = predict_or_learn<false, expf>;
    name += "-poisson";
    multipredict_f = multipredict<expf>;
  }
  else
  {
    THROW("Unknown link function: " << link);
  }

  auto s = VW::make_unique<scorer>(&all);
  // This always returns a base_learner.
  auto* base = as_singleline(stack_builder.setup_base_learner());
  auto* l = VW::LEARNER::make_reduction_learner(std::move(s), base, learn_fn, predict_fn, name)
                .set_learn_returns_prediction(base->learn_returns_prediction)
                .set_input_label_type(VW::label_type_t::simple)
                .set_output_prediction_type(VW::prediction_type_t::scalar)
                .set_multipredict(multipredict_f)
                .set_update(update)
                // .set_params_per_weight(2) // REMOVEEEEEE
                .build();

  return make_base(*l);
}
