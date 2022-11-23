// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "vw/core/reductions/interaction_ground.h"
#include "vw/config/options_cli.h"

#include "vw/core/debug_log.h"
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
#include "vw/core/simple_label.h"
#include "details/automl_impl.h"

using namespace VW::LEARNER;
using namespace VW::config;

#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::igl

namespace
{
struct interaction_ground
{
  float p_unlabeled_prior = 0.5f;
  VW::LEARNER::single_learner* decoder_learner = nullptr;
  VW::example* buffer_sl = nullptr; // TODO: rename this. This is buffer single line example

  std::vector<std::vector<namespace_index>> psi_interactions;
  std::vector<std::vector<extent_term>>* extent_interactions;

  std::unique_ptr<VW::workspace> sl_all;
  ftrl* ftrl_base; // this one get automatically save resume
  std::shared_ptr<ftrl> ftrl2; // TODO: we have save resume this thing separately
  
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
  ec.loss = 0.f;
  ec.num_features_from_interactions = 0;
  ec.num_features = 0;
}

void learn(interaction_ground& ig, multi_learner& base, VW::multi_ex& ec_seq)
{
  // std::swap(ig.ftrl_base->all->loss, ig.sl_all->loss);
  shared_data* original_sd = ig.ftrl_base->all->sd;
  // std::swap(ig.ftrl_base->all->sd, ig.sl_all->sd);
  ig.ftrl_base->all->sd = ig.sl_all->sd;
  ig.ftrl_base->all->loss = get_loss_function(*ig.ftrl_base->all, "logistic", -1.f, 1.f);

  float psi_pred = 0.f;
  int chosen_action_idx = 0;

  const auto it = std::find_if(ec_seq.begin(), ec_seq.end(), [](VW::example* item) { return !item->l.cb.costs.empty(); });

  if (it != ec_seq.end())
  {
    chosen_action_idx = std::distance(ec_seq.begin(), it);
  }

  // auto feature_hash = VW::hash_feature(*ig.ftrl_base->all, "click", VW::hash_space(*ig.ftrl_base->all, "Feedback"));
  // auto fh2 = 1328936; // hash for "|Feedback click"

  auto feedback_ex = ec_seq.back(); // TODO: refactor this. Currently assuming last example is feedback example
  ec_seq.pop_back();

  auto ns_iter = feedback_ex->indices.begin();
  auto fh = feedback_ex->feature_space.at(*ns_iter).indices.data();
  // auto fh2 = *fh / 2; //TODO: verify if this is right. 2 is problem multiplier. 
  auto fh2 = *fh;
  // std::string feedback_feature = feedback_ex->feature_space.at(*ns_iter).space_names[0].name;
  // std::cout << "[IGL] hash value: " << fh2 << std::endl;

  for (auto& ex_action : ec_seq) {
    empty_example(*ig.buffer_sl);
    // TODO: Do we need constant feature here? If so, VW::add_constant_feature
    LabelDict::add_example_namespaces_from_example(*ig.buffer_sl, *ex_action);
    ig.buffer_sl->indices.push_back(feedback_namespace);
    ig.buffer_sl->feature_space[feedback_namespace].push_back(1, fh2); // TODO: remove hardcode fh2
    // ig.buffer_sl->_debug_current_reduction_depth = ex_action->_debug_current_reduction_depth;
    ig.buffer_sl->num_features++;
    // std::cout << "[IGL] psi learn features: " << VW::debug::features_to_string(*ig.buffer_sl) << std::endl;

    // 1. learning psi
    float label = -1.f;
    // TODO: update the importance for each example
    float importance = 0.6f;
    if (!ex_action->l.cb.costs.empty()) {
      label = 1.f;
    }

    ig.buffer_sl->l.simple.label = label;
    ig.buffer_sl->weight = importance;

    ig.buffer_sl->interactions = &ig.psi_interactions;
    ig.buffer_sl->extent_interactions = ig.extent_interactions; // TODO(low pri): not reuse ig.extent_interactions, need to add in feedback

    // 2. psi learn
    ig.decoder_learner->learn(*ig.buffer_sl, 0);

    if (!ex_action->l.cb.costs.empty()) {
      psi_pred = ig.buffer_sl->pred.scalar;
    }
    output_and_account_example(*ig.ftrl_base->all, *ig.buffer_sl);
  }
  
  // std::swap(ig.ftrl_base->all->loss, ig.sl_all->loss);
  // std::swap(ig.ftrl_base->all->sd, ig.sl_all->sd);
  ig.ftrl_base->all->sd = original_sd;
  ig.ftrl_base->all->loss = get_loss_function(*ig.ftrl_base->all, "squared", -1.f, 1.f); //VW::make_unique<squaredloss>();
  
  float fake_cost = 0.f;
  // 4. update multi line ex label
  if (psi_pred * 2 > 1) {
    // extreme state
    // TODO: get Definitely Bad from feedback example
    bool is_neg = 1;
    fake_cost = -ig.p_unlabeled_prior + is_neg * (1 + ig.p_unlabeled_prior); // TODO: update to latest version
  }

  // 5. Train pi policy
  // ec_seq[chosen_action_idx]->l.cb.costs[0].cost = fake_cost;
  // std::cout << "[IGL] psi pred: " << psi_pred << ","
  //           << "chosen action prob: " << ec_seq[chosen_action_idx]->l.cb.costs[0].probability << ", "
  //           << "fake cost: " << fake_cost
  //           << std::endl;

  // VW::reductions::swap_ftrl(ig.ftrl2.get(), ig.ftrl_base);
  // base.learn(ec_seq, 1);
  // VW::reductions::swap_ftrl(ig.ftrl2.get(), ig.ftrl_base);
  ec_seq.push_back(feedback_ex);
}

void predict(interaction_ground& ig, multi_learner& base, VW::multi_ex& ec_seq)
{
  // Is loss func used in predict?
  // matches what we do for learn
  // VW::reductions::swap_ftrl(ig.ftrl2.get(), ig.ftrl_base);
  // base.predict(ec_seq, 1);
  // VW::reductions::swap_ftrl(ig.ftrl2.get(), ig.ftrl_base);
}
} // namespace

// this impl only needs (re-use instantiated coin instead of ftrl func, scorer setup func, count_label 
// setup func) -> remove all other reduction setups

void copy_ftrl(ftrl* source, ftrl* destination) { // convert this to swap and try
  destination->all = source->all;
  destination->ftrl_alpha = source->ftrl_alpha;
  destination->ftrl_beta = source->ftrl_beta;
  destination->data.update = source->data.update;
  destination->data.ftrl_alpha = source->data.ftrl_alpha;
  destination->data.ftrl_beta = source->data.ftrl_beta;
  destination->data.l1_lambda = source->data.l1_lambda;
  destination->data.l2_lambda = source->data.l2_lambda;
  destination->data.predict = source->data.predict;
  destination->data.normalized_squared_norm_x = source->data.normalized_squared_norm_x;
  destination->data.average_squared_norm_x = source->data.average_squared_norm_x;
  destination->no_win_counter = source->no_win_counter;
  destination->early_stop_thres = source->early_stop_thres;
  destination->ftrl_size = source->ftrl_size;
  destination->total_weight = source->total_weight;
  destination->normalized_sum_norm_x = source->normalized_sum_norm_x;
}

void VW::reductions::swap_ftrl(ftrl* source, ftrl* destination) { // convert this to swap and try
  std::swap(destination->all, source->all);
  std::swap(destination->ftrl_alpha , source->ftrl_alpha);
  std::swap(destination->ftrl_beta , source->ftrl_beta);
  std::swap(destination->data.update , source->data.update);
  std::swap(destination->data.ftrl_alpha , source->data.ftrl_alpha);
  std::swap(destination->data.ftrl_beta , source->data.ftrl_beta);
  std::swap(destination->data.l1_lambda , source->data.l1_lambda);
  std::swap(destination->data.l2_lambda , source->data.l2_lambda);
  std::swap(destination->data.predict , source->data.predict);
  std::swap(destination->data.normalized_squared_norm_x , source->data.normalized_squared_norm_x);
  std::swap(destination->data.average_squared_norm_x , source->data.average_squared_norm_x);
  std::swap(destination->no_win_counter , source->no_win_counter);
  std::swap(destination->early_stop_thres , source->early_stop_thres);
  std::swap(destination->ftrl_size , source->ftrl_size);
  std::swap(destination->total_weight , source->total_weight);
  std::swap(destination->normalized_sum_norm_x , source->normalized_sum_norm_x);
}

void finish_igl_examples(VW::workspace& all, interaction_ground& ig, VW::multi_ex& ec_seq)
{
  // VW::LEARNER::as_singleline(data._base)->finish_example(all, ec);
  // output_and_account_example(all, ec);
}

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

  // std::cout << "[IGL] pi policy is multiline: " << pi->is_multiline() << std::endl;

  // 1. fetch already allocated coin reduction
  std::vector<std::string> enabled_reductions;
  pi->get_enabled_reductions(enabled_reductions);
  const char* const delim = ", ";
  std::ostringstream imploded;
  std::copy(
      enabled_reductions.begin(),
      enabled_reductions.end() - 1,
      std::ostream_iterator<std::string>(imploded, delim));
  std::cerr << "[IGL]: " << imploded.str() << enabled_reductions.back() << std::endl;

  auto* ftrl_coin = pi->get_learner_by_name_prefix("ftrl-Coin");

  // 2. prepare args for second stack
  // TODO: construct psi args from pi args
  std::string psi_args = "--quiet --link=logistic --loss_function=logistic --coin"; //TODO: Add back --cubic

  int argc = 0;
  char** argv = to_argv_escaped(psi_args, argc);

  std::unique_ptr<options_i, options_deleter_type> psi_options(
    new config::options_cli(std::vector<std::string>(argv + 1, argv + argc)),
    [](VW::config::options_i* ptr) { delete ptr; });

  assert(psi_options->was_supplied("cb_adf") == false);
  assert(psi_options->was_supplied("loss_function") == true);

  std::unique_ptr<custom_builder> psi_builder = VW::make_unique<custom_builder>(ftrl_coin);
  //VW::workspace sl_all(VW::io::create_null_logger());
  ld->sl_all = VW::make_unique<VW::workspace>(VW::io::create_null_logger());
  // assuming parser gets destroyed by workspace
  ld->sl_all->example_parser = new parser{all->example_parser->example_queue_limit, all->example_parser->strict_parse};
  ld->sl_all->sd = new shared_data(); //TODO: separate sd or shared?
  ld->sl_all->loss = get_loss_function(*ld->sl_all, "logistic", -1.f, 1.f); // TODO: min is -1 or 0? // for scorer
  // ld->sl_all->weights.dense_weights = std::move(all->weights.dense_weights);

  psi_builder->delayed_state_attach(*ld->sl_all, *psi_options);
  ld->decoder_learner = as_singleline(psi_builder->setup_base_learner());
  ld->ftrl_base = static_cast<ftrl*>(ftrl_coin->get_internal_type_erased_data_pointer_test_use_only());
  // auto other_ftrl = VW::make_unique<ftrl>();
  ld->ftrl2 = std::make_shared<ftrl>(); //other_ftrl.release()
  copy_ftrl(ld->ftrl_base, ld->ftrl2.get());

  for (auto& interaction : all->interactions) {
    interaction.push_back(feedback_namespace);
    ld->psi_interactions.push_back(interaction);
    interaction.pop_back();
  }

  // std::cout << "[IGL] interations:" << VW::reductions::util::interaction_vec_t_to_string(all->interactions, "quadratic") <<std::endl;

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
                // .set_finish_example(finish_igl_examples)
                .build();
  // TODO: assert ftrl is the base, fail otherwise
  // VW::reductions::util::fail_if_enabled

  return make_base(*pi_learner);
}
