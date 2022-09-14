// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "vw/core/reductions/interaction_ground.h"
#include "vw/config/options_cli.h"
#include "vw/core/reduction_stack.h"

#include "vw/common/vw_exception.h"
#include "vw/config/options.h"
#include "vw/core/label_dictionary.h"
#include "vw/core/label_parser.h"
#include "vw/core/prediction_type.h"
#include "vw/core/reductions/cb/cb_adf.h"
#include "vw/core/reductions/cb/cb_algs.h"
#include "vw/core/setup_base.h"
#include "vw/core/shared_data.h"
#include "vw/core/vw.h"

using namespace VW::LEARNER;
using namespace VW::config;
using namespace CB_ALGS;

namespace
{
struct interaction_ground
{
  // the accumulated importance weighted reward of a policy which optimizes the given value
  double total_importance_weighted_reward = 0.;
  double total_uniform_reward = 0.;
  // the accumulated importance weighted loss of the policy which optimizes the negative of the given value
  double total_importance_weighted_cost = 0.;
  double total_uniform_cost = 0.;

  VW::LEARNER::base_learner* decoder_learner = nullptr;
};

void negate_cost(VW::multi_ex& ec_seq)
{
  for (auto* example_ptr : ec_seq)
  {
    for (auto& label : example_ptr->l.cb.costs)
    {
      if (label.has_observed_cost()) { label.cost = -label.cost; }
    }
  }
}

void learn(interaction_ground& ig, multi_learner& base, VW::multi_ex& ec_seq)
{
  // find reward of sequence
  CB::cb_class label = CB_ADF::get_observed_cost_or_default_cb_adf(ec_seq);
  ig.total_uniform_cost += label.cost / label.probability / ec_seq.size();  //=p(uniform) * IPS estimate
  ig.total_uniform_reward += -label.cost / label.probability / ec_seq.size();

  // find prediction & update for cost
  base.predict(ec_seq);
  ig.total_importance_weighted_cost += get_cost_estimate(label, ec_seq[0]->pred.a_s[0].action);
  base.learn(ec_seq);

  // find prediction & update for reward
  label.cost = -label.cost;
  base.predict(ec_seq, 1);
  ig.total_importance_weighted_reward += get_cost_estimate(label, ec_seq[0]->pred.a_s[0].action);

  // change cost to reward
  negate_cost(ec_seq);
  base.learn(ec_seq, 1);
  negate_cost(ec_seq);
}

void predict(interaction_ground& ig, multi_learner& base, VW::multi_ex& ec_seq)
{
  // figure out which is better by our current estimate.
  if (ig.total_uniform_cost - ig.total_importance_weighted_cost >
      ig.total_uniform_reward - ig.total_importance_weighted_reward)
  { base.predict(ec_seq); }
  else
  {
    base.predict(ec_seq, 1);
  }
}
}  // namespace

// this impl only needs (re-use instantiated coin instead of ftrl func, scorer setup func, count_label 
// setup func) -> remove all other reduction setups
struct custom_builder : VW::default_reduction_stack_setup
{
  custom_builder(VW::LEARNER::base_learner* ftrl_coin):psi_base(ftrl_coin)
  {
    std::set<std::string> keep = {"scorer", "count_label"};

    for (int i=reduction_stack.size() - 1; i >= 0; i--) {
      if (keep.count(std::get<0>(reduction_stack.at(i))) == 0) {
        reduction_stack.erase(reduction_stack.begin() + i);
      }
    }
  }

  VW::LEARNER::base_learner* setup_base_learner() {
    // auto base_label_type = all->example_parser->lbl_parser.label_type;
    if (reduction_stack.size() == 0) {
      return psi_base;
    }
    auto *psi_base_learner = VW::default_reduction_stack_setup::setup_base_learner();
    return psi_base_learner;
  }

  private:
  VW::LEARNER::base_learner* psi_base;
};

base_learner* VW::reductions::interaction_ground_setup(VW::setup_base_i& stack_builder)
{
  // rename this var to options_first_stack
  options_i& options = *stack_builder.get_options();
  VW::workspace *all = stack_builder.get_all_pointer();
  bool igl_option = false;

  option_group_definition new_options("[Reduction] Interaction Grounded Learning");
  new_options.add(make_option("experimental_igl", igl_option)
                      .keep()
                      .necessary()
                      .help("Do Interaction Grounding with multiline action dependent features")
                      .experimental());

  if (!options.add_parse_and_check_necessary(new_options)) { return nullptr; }

  // number of weight vectors needed
  size_t problem_multiplier = 2;  // One for pi(training model) and one for psi(decoder model)
  auto ld = VW::make_unique<interaction_ground>();

  // Ensure cb_explore_adf so we are reducing to something useful.
  // Question: are cb_adf and cb_explore_adf using same stack? Can we support both?
  if (!options.was_supplied("cb_explore_adf")) { options.insert("cb_explore_adf", ""); }

  auto* pi = as_multiline(stack_builder.setup_base_learner());

  // 1. fetch already allocated coin reduction
  std::vector<std::string> enabled_reductions;
  pi->get_enabled_reductions(enabled_reductions);
  const char* const delim = ", ";
  std::ostringstream imploded;
  std::copy(
      enabled_reductions.begin(), enabled_reductions.end() - 1, std::ostream_iterator<std::string>(imploded, delim));
  std::cerr << "(inside IGL): " << imploded.str() << enabled_reductions.back() << std::endl;

  auto* ftrl_coin = pi->get_learner_by_name_prefix("ftrl-Coin");

  // 2. prepare args for second stack
  // Question: how to handle interactions?
  std::string psi_args = "--quiet --link=logistic --loss_function=logistic --cubic cva --coin";

  // Question: what is argc and argv?
  int argc = 0;
  char** argv = to_argv_escaped(psi_args, argc);

  std::unique_ptr<options_i, options_deleter_type> psi_options(
    new config::options_cli(std::vector<std::string>(argv + 1, argv + argc)),
    [](VW::config::options_i* ptr) { delete ptr; });

  assert(psi_options->was_supplied("cb_adf") == false);
  assert(psi_options->was_supplied("loss_function") == true);

  std::unique_ptr<VW::setup_base_i> psi_builder= VW::make_unique<custom_builder>(ftrl_coin);
  VW::workspace temp(VW::io::create_null_logger());
  temp.example_parser = all->example_parser;
  psi_builder->delayed_state_attach(temp, *psi_options);

  ld->decoder_learner = psi_builder->setup_base_learner();
  // TODO: free argc and argv
  // for (int i = 0; i < argc; i++) { free(argv[i]); }
  // free(argv);

  /*
      stack1 (synonym of base line 103): ftrl-Coin Betting, scorer-identity, csoaa_ldf-rank, cb_adf, shared_feature_merger
      stack2: ftrl-Coin Betting, scorer-logistic, count_label

        in another file i.e. custom_igl_stack_builder : implements interface setup_base_i or inherit from default_reduction_stack_setup
        this impl only needs (re-use instantiated coin instead of ftrl func, scorer setup func, count_label setup func) -> remove all other reduction setups
        (this is done in custom_reduction_test line 122)

        my_second_builder = custom_igl_stack_builder(instantiated_coin, my_second_options)
        second_stack = my_second_builder.setup_base()


      ld.second_stack = second_stack

      inside learn we should be able to reference ld.second_stack to callinto second_stack.learn /second_stack.preedict
      build single_ex
      second_stack.learn(single_ex, OFFSET=1)

      */




  auto* pi_learner = make_reduction_learner(
      std::move(ld), pi, learn, predict, stack_builder.get_setupfn_name(interaction_ground_setup))
                .set_params_per_weight(problem_multiplier)
                .set_input_label_type(label_type_t::cb)
                .set_output_label_type(label_type_t::cb)
                .set_output_prediction_type(prediction_type_t::action_scores)
                .set_input_prediction_type(prediction_type_t::action_scores)
                .build();

  return make_base(*pi_learner);
}
