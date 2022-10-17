// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "vw/core/reductions/interaction_ground.h"
#include "vw/config/options_cli.h"

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
#include "vw/core/loss_functions.h"

using namespace VW::LEARNER;
using namespace VW::config;
using namespace CB_ALGS;

namespace
{
struct interaction_ground
{
  float p_unlabeled_prior = 0.5f;
  VW::LEARNER::single_learner* decoder_learner = nullptr;
  VW::example* buffer_sl = nullptr; // TODO: rename this. This is buffer single line example

  std::vector<std::vector<namespace_index>> psi_interactions;
  std::vector<std::vector<extent_term>>* extent_interactions;

  std::unique_ptr<VW::workspace> temp; // TODO: rename temp
  ~interaction_ground() {
    VW::dealloc_examples(buffer_sl, 1);
  }
};

// TODO: we copied this from parser.cc, but we don't need workspace
void empty_example(example& ec)
{
  for (features& fs : ec) { fs.clear(); }

  ec.indices.clear();
  ec.tag.clear();
  ec.sorted = false;
  ec.end_pass = false;
  ec.is_newline = false;
  ec._reduction_features.clear();
  ec.num_features_from_interactions = 0;
}

void learn(interaction_ground& ig, multi_learner& base, VW::multi_ex& ec_seq)
{
  for (auto& ex_action: ec_seq) {
    empty_example(*ig.buffer_sl);
    // TODO: Do we need constant feature here? If so, VW::add_constant_feature
    LabelDict::add_example_namespaces_from_example(*ig.buffer_sl, *ex_action);
    ig.buffer_sl->indices.push_back(feedback_namespace);
    ig.buffer_sl->feature_space[feedback_namespace].push_back(1, 777); // feature name 777 value 1
    std::cout << "==features: " << VW::debug::features_to_string(*ig.buffer_sl) << std::endl;
    std::cout << ex_action->l.cb.costs.empty() << std::endl;

    // 1. learning psi
    float label = -1.f;
    // TODO: update the importance for each example
    // Need to pass in the pa distribution
    float importance = 0.5;
    if (!ex_action->l.cb.costs.empty()) {
      label = 1.f;
    }

    ig.buffer_sl->l.simple.label = label;
    ig.buffer_sl->weight = importance;

    // TODO(high p): change the interactions to loop through the ig.interactions. similiar to https://github.com/VowpalWabbit/vowpal_wabbit/blob/14b59e8249edaeb74251e9ac0d2d25966767ed0f/vowpalwabbit/core/src/reductions/details/automl/automl_oracle.cc#L87
    ig.buffer_sl->interactions = &ig.psi_interactions;
    std::cout << "--interactions--" << std::endl;
    std::cout << ig.buffer_sl->interactions << std::endl;
    ig.buffer_sl->extent_interactions = ig.extent_interactions; // TODO(low pri): not reuse ig.extent_interactions, need to add in feedback

    // 2. psi learn
    ig.decoder_learner->learn(*ig.buffer_sl, 0);
  }

  // 3. Psi predict - (single line ex) feedback + context + chosen action
  auto chosen_action_it = std::find_if(ec_seq.begin(), ec_seq.end(), [](const example* action_ex) {
    return !action_ex->l.cb.costs.empty();
  });

  if (chosen_action_it != ec_seq.end()) {
    empty_example(*ig.buffer_sl);

    size_t chosen_action_idx = std::distance(ec_seq.begin(), chosen_action_it);
    std::cout << "==chosen action idx: " << chosen_action_idx << std::endl;
    // TODO: Do we need constant feature here? If so, VW::add_constant_feature
    LabelDict::add_example_namespaces_from_example(*ig.buffer_sl, **chosen_action_it);

    // TODO: pop out eventID, actionTaken, definitelyBad features
    ig.buffer_sl->indices.push_back(70); // TODO: change to feedback_namespace
    for (auto& a : ig.buffer_sl->indices) {
      std::cout << a << std::endl;
    }
    ig.buffer_sl->feature_space[70].push_back(1, 777); // feature name 777 value 1 // TODO: change 70 to feedback_namespace
    std::cout << "[IGL] psi predict features: " << VW::debug::features_to_string(*ig.buffer_sl) << std::endl;

    ig.decoder_learner->predict(*ig.buffer_sl, 0);
    std::cout << "[IGL] psi pred:" << ig.buffer_sl->pred.scalar << std::endl;

    // 4. Update the label in es
    auto psi_pred = ig.buffer_sl->pred.scalar;
    float fake_cost = 0.f;

    if (psi_pred * 2 > 1) {
      // extreme state
      // TODO: get Definitely Bad from feedback example
      bool is_neg = 1;
      fake_cost = -ig.p_unlabeled_prior + is_neg * (1 + ig.p_unlabeled_prior); // TODO: update to latest version
    }

    // 5. Train pi policy
    std::cout << "[IGL] chosen_action_idx: " << chosen_action_idx << std::endl;
    ec_seq[chosen_action_idx]->l.cb.costs[0].cost = fake_cost;
    base.learn(ec_seq, 1);
  } else {
    // TODO: throw error if example has no chosen action?
  }
}

void predict(interaction_ground& ig, multi_learner& base, VW::multi_ex& ec_seq)
{
  base.predict(ec_seq, 1);
}
}  // namespace

// this impl only needs (re-use instantiated coin instead of ftrl func, scorer setup func, count_label 
// setup func) -> remove all other reduction setups

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

  ld->buffer_sl = VW::alloc_examples(1);


  // Ensure cb_explore_adf so we are reducing to something useful.
  // Question: are cb_adf and cb_explore_adf using same stack? Can we support both?
  if (!options.was_supplied("cb_explore_adf")) { options.insert("cb_explore_adf", ""); }

  auto* pi = as_multiline(stack_builder.setup_base_learner());

  std::cout << "pi policy is multiline: " << pi->is_multiline() << std::endl;

  // 1. fetch already allocated coin reduction
  std::vector<std::string> enabled_reductions;
  pi->get_enabled_reductions(enabled_reductions);
  const char* const delim = ", ";
  std::ostringstream imploded;
  std::copy(
      enabled_reductions.begin(),
      enabled_reductions.end() - 1,
      std::ostream_iterator<std::string>(imploded, delim));
  std::cerr << "(inside IGL): " << imploded.str() << enabled_reductions.back() << std::endl;

  auto* ftrl_coin = pi->get_learner_by_name_prefix("ftrl-Coin");

  // 2. prepare args for second stack
  // TODO: construct psi args from pi args
  std::string psi_args = "--quiet --link=logistic --loss_function=logistic --cubic UAF --coin";
  int argc = 0;
  char** argv = to_argv_escaped(psi_args, argc);

  std::unique_ptr<options_i, options_deleter_type> psi_options(
    new config::options_cli(std::vector<std::string>(argv + 1, argv + argc)),
    [](VW::config::options_i* ptr) { delete ptr; });

  assert(psi_options->was_supplied("cb_adf") == false);
  assert(psi_options->was_supplied("loss_function") == true);

  std::unique_ptr<custom_builder> psi_builder = VW::make_unique<custom_builder>(ftrl_coin);
  //VW::workspace temp(VW::io::create_null_logger());
  ld->temp = VW::make_unique<VW::workspace>(VW::io::create_null_logger());
  // assuming parser gets destroyed by workspace
  ld->temp->example_parser = new parser{all->example_parser->example_queue_limit, all->example_parser->strict_parse};
  ld->temp->sd = new shared_data(); //TODO: separate sd or shared?
  ld->temp->loss = get_loss_function(*ld->temp, "logistic", 0.f, 1.f); // TODO: min is -1 or 0?
  psi_builder->delayed_state_attach(*ld->temp, *psi_options);
  ld->decoder_learner = as_singleline(psi_builder->setup_base_learner());

  for (auto& interaction : all->interactions) {
    interaction.push_back(feedback_namespace);
    ld->psi_interactions.push_back(interaction);
  }

  ld->extent_interactions = &all->extent_interactions; // TODO: do we care about full ns interaction?

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

  // TODO: assert ftrl is the base, fail otherwise
  // VW::reductions::util::fail_if_enabled

  return make_base(*pi_learner);
}
