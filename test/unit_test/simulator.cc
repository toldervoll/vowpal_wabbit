// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "simulator.h"

#include "vw/core/vw.h"

#include <fmt/format.h>

#include <numeric>

namespace simulator
{
cb_sim::cb_sim(uint64_t seed)
    : users({"Tom", "Anna"})
    , times_of_day({"morning", "afternoon"})
    , actions({"politics", "sports", "music", "food", "finance", "health", "camping"})
    // , actions({"politics", "sports", "music"})
    , user_ns("User")
    , action_ns("Action")
{
  random_state.set_random_state(seed);
  callback_count = 0;
}

float cb_sim::get_reaction(
    const std::map<std::string, std::string>& context, const std::string& action, bool add_noise, bool swap_reward)
{
  float like_reward = USER_LIKED_ARTICLE;
  float dislike_reward = USER_DISLIKED_ARTICLE;
  if (add_noise)
  {
    like_reward += random_state.get_and_update_random();
    dislike_reward += random_state.get_and_update_random();
  }

  float reward = dislike_reward;
  if (context.at("user") == "Tom")
  {
    if (context.at("time_of_day") == "morning" && action == "politics") { reward = like_reward; }
    else if (context.at("time_of_day") == "afternoon" && action == "music")
    {
      reward = like_reward;
    }
  }
  else if (context.at("user") == "Anna")
  {
    if (context.at("time_of_day") == "morning" && action == "sports") { reward = like_reward; }
    else if (context.at("time_of_day") == "afternoon" && action == "politics")
    {
      reward = like_reward;
    }
  }

  if (swap_reward) { return (reward == like_reward) ? dislike_reward : like_reward; }
  return reward;
}

// todo: skip text format and create vw example directly
std::vector<std::string> cb_sim::to_vw_example_format(
    const std::map<std::string, std::string>& context, const std::string& chosen_action, float cost, float prob)
{
  std::vector<std::string> multi_ex_str;
  multi_ex_str.push_back(
      fmt::format("shared |{} user={} time_of_day={}", user_ns, context.at("user"), context.at("time_of_day")));
  for (const auto& action : actions)
  {
    std::ostringstream ex;
    if (action == chosen_action) { ex << fmt::format("0:{}:{} ", cost, prob); }
    ex << fmt::format("|{} article={}", action_ns, action);
    multi_ex_str.push_back(ex.str());
  }
  return multi_ex_str;
}

std::pair<int, float> cb_sim::sample_custom_pmf(std::vector<float>& pmf)
{
  float total = std::accumulate(pmf.begin(), pmf.end(), 0.f);
  float scale = 1.f / total;
  for (float& val : pmf) { val *= scale; }
  float draw = random_state.get_and_update_random();
  float sum_prob = 0.f;
  for (int index = 0; index < pmf.size(); ++index)
  {
    sum_prob += pmf[index];
    if (sum_prob > draw) { return std::make_pair(index, pmf[index]); }
  }
  THROW("Error: No prob selected");
}

std::vector<float> cb_sim::get_pmf(VW::workspace* vw, const std::map<std::string, std::string>& context)
{
  std::vector<std::string> multi_ex_str = cb_sim::to_vw_example_format(context, "");
  VW::multi_ex examples;
  for (const std::string& ex : multi_ex_str) { examples.push_back(VW::read_example(*vw, ex)); }
  vw->predict(examples);

  std::vector<float> pmf;
  auto const& scores = examples[0]->pred.a_s;
  std::vector<float> ordered_scores(scores.size());
  for (auto const& action_score : scores) { ordered_scores[action_score.action] = action_score.score; }
  for (auto action_score : ordered_scores) { pmf.push_back(action_score); }
  vw->finish_example(examples);

  return pmf;
}

const std::string& cb_sim::choose_user()
{
  int rand_ind = static_cast<int>(random_state.get_and_update_random() * users.size());
  return users[rand_ind];
}

const std::string& cb_sim::choose_time_of_day()
{
  int rand_ind = static_cast<int>(random_state.get_and_update_random() * times_of_day.size());
  return times_of_day[rand_ind];
}

void cb_sim::call_if_exists(VW::workspace& vw, VW::multi_ex& ex, const callback_map& callbacks, const size_t event)
{
  auto iter = callbacks.find(event);
  if (iter != callbacks.end())
  {
    callback_count++;
    bool callback_result = iter->second(*this, vw, ex);
    BOOST_CHECK_EQUAL(callback_result, true);
  }
}

std::vector<float> cb_sim::run_simulation_hook(VW::workspace* vw, size_t num_iterations, callback_map& callbacks,
    bool do_learn, size_t shift, bool add_noise, uint64_t num_useless_features, const std::vector<uint64_t>& swap_after)
{
  // check if there's a callback for the first possible element,
  // in this case most likely 0th event
  // i.e. right before sending any event to VW
  VW::multi_ex dummy;
  call_if_exists(*vw, dummy, callbacks, shift - 1);

  bool swap_reward = false;
  auto swap_after_iter = swap_after.begin();

  for (size_t i = shift; i < shift + num_iterations; ++i)
  {
    if (swap_after_iter != swap_after.end())
    {
      if (i > *swap_after_iter)
      {
        ++swap_after_iter;
        swap_reward = !swap_reward;
      }
    }
    // 1. In each simulation choose a user
    auto user = choose_user();

    // 2. Choose time of day for a given user
    auto time_of_day = choose_time_of_day();

    // 3. Pass context to vw to get an action
    std::map<std::string, std::string> context{{"user", user}, {"time_of_day", time_of_day}};
    // Add useless features if specified
    for (uint64_t j = 0; j < num_useless_features; ++j)
    { context.insert(std::pair<std::string, std::string>(std::to_string(j), std::to_string(j))); }
    // auto action_prob = get_action(vw, context);
    auto pmf = get_pmf(vw, context);
    std::pair<int, float> pmf_sample = sample_custom_pmf(pmf);
    std::pair<std::string, float> action_prob =  std::make_pair(actions[pmf_sample.first], pmf_sample.second);
    auto chosen_action = action_prob.first;
    auto prob = action_prob.second;

    // 4. Get cost of the action we chose
    // Check for reward swap
    float cost = get_reaction(context, chosen_action, add_noise, swap_reward);
    cost_sum += cost;

    if (do_learn)
    {
      // 5. Inform VW of what happened so we can learn from it
      std::vector<std::string> multi_ex_str = to_vw_example_format(context, chosen_action, cost, prob);
      VW::multi_ex examples;
      for (const std::string& ex : multi_ex_str) { examples.push_back(VW::read_example(*vw, ex)); }

      // 6. Learn
      vw->learn(examples);

      call_if_exists(*vw, examples, callbacks, i);

      // 7. Let VW know you're done with these objects
      vw->finish_example(examples);
    }

    // We negate this so that on the plot instead of minimizing cost, we are maximizing reward
    ctr.push_back(-1 * cost_sum / static_cast<float>(i));
  }

  // avoid silently failing: ensure that all callbacks
  // got called and then cleanup
  BOOST_CHECK_EQUAL(callbacks.size(), callback_count);
  callbacks.clear();
  callback_count = 0;
  BOOST_CHECK_EQUAL(callbacks.size(), callback_count);

  return ctr;
}

std::vector<float> cb_sim::run_simulation(
    VW::workspace* vw, size_t num_iterations, bool do_learn, size_t shift, const std::vector<uint64_t>& swap_after)
{
  callback_map callbacks;
  return cb_sim::run_simulation_hook(vw, num_iterations, callbacks, do_learn, shift, false, 0, swap_after);
}

std::vector<float> _test_helper(const std::string& vw_arg, size_t num_iterations, int seed)
{
  auto vw = VW::initialize(vw_arg);
  simulator::cb_sim sim(seed);
  auto ctr = sim.run_simulation(vw, num_iterations);
  VW::finish(*vw);
  return ctr;
}

std::vector<float> _test_helper_save_load(const std::string& vw_arg, size_t num_iterations, int seed,
    const std::vector<uint64_t>& swap_after, const size_t split)
{
  BOOST_CHECK_GT(num_iterations, split);
  size_t before_save = num_iterations - split;

  auto first_vw = VW::initialize(vw_arg);
  simulator::cb_sim sim(seed);
  // first chunk
  auto ctr = sim.run_simulation(first_vw, before_save, true, 1, swap_after);
  // save
  std::string model_file = "test_save_load.vw";
  VW::save_predictor(*first_vw, model_file);
  VW::finish(*first_vw);
  // reload in another instance
  auto other_vw = VW::initialize(vw_arg + " --quiet -i " + model_file);
  // continue
  ctr = sim.run_simulation(other_vw, split, true, before_save + 1, swap_after);
  VW::finish(*other_vw);
  return ctr;
}

std::vector<float> _test_helper_hook(const std::string& vw_arg, callback_map& hooks, size_t num_iterations, int seed,
    const std::vector<uint64_t>& swap_after)
{
  BOOST_CHECK(true);
  auto* vw = VW::initialize(vw_arg);
  simulator::cb_sim sim(seed);
  auto ctr = sim.run_simulation_hook(vw, num_iterations, hooks, true, 1, false, 0, swap_after);
  VW::finish(*vw);
  return ctr;
}

std::string igl_sim::sample_feedback(const std::map<std::string, float>& probs) {
  float draw = random_state.get_and_update_random();
  float sum_prob = 0.f;
  for (auto& feedback_prob : probs) {
    sum_prob += feedback_prob.second;
    if (draw < sum_prob) {
      return feedback_prob.first;
    }
  }
  THROW("Error: No feedback selected");
}

std::string igl_sim::get_feedback(const std::string& user, const std::string& chosen_action) {
  if (ground_truth_enjoy.at(user) == chosen_action) {
    return sample_feedback(enjoy_prob);
  } else if (ground_truth_hate.at(user) == chosen_action) {
    return sample_feedback(hate_prob);
  }

  return sample_feedback(neutral_prob);
}

igl_sim::igl_sim(uint64_t seed) 
: cb_sim(seed)
// , actions({"politics", "sports", "music", "food", "finance", "health", "camping"})
, feedback_ns("Feedback")
{}

std::string igl_sim::to_psi_predict_ex(const std::map<std::string, std::string>& context,
  const std::string& chosen_action, const std::string& feedback)
{
  return fmt::format(" |{} user={} time_of_day={} |{} article={} |{} feedback={}", 
      user_ns, context.at("user"), context.at("time_of_day"),
      action_ns, chosen_action,
      feedback_ns, feedback);
}

std::vector<std::string> igl_sim::to_psi_learn_ex(const std::map<std::string, std::string>& context,
  const std::string& chosen_action, const std::string& feedback, const std::vector<float>& pmf)
{
  std::vector<std::string> examples;
  for (size_t i = 0; i < actions.size(); i++)
  {
    std::ostringstream ex;
    int label = actions[i] == chosen_action ? 1 : -1;
    float importance = actions[i] == chosen_action ? (1.f/4) / pmf[i] : (3.f/4) / (1 - pmf[i]);

    ex << fmt::format("{} {} |{} user={} time_of_day={} |{} article={} |{} feedback={}", 
        label, importance,
        user_ns, context.at("user"), context.at("time_of_day"),
        action_ns, actions[i],
        feedback_ns, feedback);
    examples.push_back(ex.str());
  }
  return examples;
}

float igl_sim::true_reward(const std::string& user, const std::string& action) {
  return (ground_truth_enjoy.at(user) == action) - (ground_truth_hate.at(user) == action);
}

std::vector<float> igl_sim::run_simulation_hook(VW::workspace* pi, VW::workspace* psi, size_t num_iterations,
    callback_map& callbacks, bool do_learn)
{
  for (size_t i = 0; i < num_iterations; i++) { // TODO: figure out if we need to add shift, noise
    // 1. In each simulation choose a user
    std::string user = choose_user();

    // 2. Choose time of day for a given user
    std::string time_of_day = choose_time_of_day();

    // 3. Pass context to vw to get an action
    std::map<std::string, std::string> context{{"user", user}, {"time_of_day", time_of_day}};

    std::vector<float> pmf = get_pmf(pi, context);
    std::pair<int, float> pmf_sample = sample_custom_pmf(pmf);
    std::string chosen_action = actions[pmf_sample.first];
    float label_prob = pmf_sample.second;

    // 4. Simulate feedback for chosen action
    std::string feedback = get_feedback(user, chosen_action);
    true_reward_sum += true_reward(user, chosen_action);

    if (do_learn) {
      // 5 - create psi vw example using feedback, context and action
      std::string psi_pred_ex_str = to_psi_predict_ex(context, chosen_action, feedback);
      VW::example* psi_pred_ex = VW::read_example(*psi, psi_pred_ex_str);
      // 6 - psi predict with psi example
      psi->predict(*psi_pred_ex);
      float psi_pred = psi_pred_ex->pred.scalar;
      psi->finish_example(*psi_pred_ex);

      // 7 - psi learn with psi example
      std::vector<std::string> psi_learn_examples = to_psi_learn_ex(context, chosen_action, feedback, pmf);
      for (const auto& ex : psi_learn_examples) {
        VW::example* psi_learn_ex = VW::read_example(*psi, ex);
        psi->learn(*psi_learn_ex);
        psi->finish_example(*psi_learn_ex);
      }

      // 8 - detect extreme state and get different label cost value
      float cost = 0;
      if (psi_pred * 2 > 1) {
        bool is_negative = feedback == "dislike" ? true : false;
        cost = -1 + is_negative * (1 + 1 / p_unlabeled_prior);
      }

      // 9 - get the cost and construct pi vw example
      std::vector<std::string> pi_ex_str = to_vw_example_format(context, chosen_action, cost, label_prob);
      VW::multi_ex pi_examples;
      for (const std::string& ex : pi_ex_str) { pi_examples.push_back(VW::read_example(*pi, ex)); }

      // 10 - pi learn
      pi->learn(pi_examples);
      pi->finish_example(pi_examples);
    }

    // 11 - calculate ctr
    ctr.push_back(true_reward_sum / static_cast<float>(i));

    // TODO: Remove this
    if ( (i != 0) && ((i & (i - 1)) == 0)) {
      std::cout << i << "   " << ctr[i] << std::endl;
    }
  }
  std::cout << ctr.size() - 1 << "   " << ctr.back() << std::endl; //TODO: Remove
  return ctr;
}

std::vector<float> igl_sim::run_simulation_hook(VW::workspace* igl_vw, size_t num_iterations,
    callback_map& callbacks, bool do_learn)
{
    for (size_t i = 0; i < num_iterations; i++) { // TODO: figure out if we need to add shift, noise
    // 1. In each simulation choose a user
    std::string user = choose_user();

    // 2. Choose time of day for a given user
    std::string time_of_day = choose_time_of_day();

    // 3. Pass context to vw to get an action
    std::map<std::string, std::string> context{{"user", user}, {"time_of_day", time_of_day}};

    std::vector<float> pmf = get_pmf(igl_vw, context);
    std::pair<int, float> pmf_sample = sample_custom_pmf(pmf);
    auto chosen_action = actions[pmf_sample.first];
    float label_prob = pmf_sample.second;
    
    // TODO:
    // call igl_vw.predict to get pmf
    // then call sample_custom_pmf to get chosen_action

    // 4. Simulate feedback for chosen action
    std::string feedback = get_feedback(user, chosen_action);
    true_reward_sum += true_reward(user, chosen_action);

    if (do_learn) {
      // TODO: 5 - create IGL example including feedback
      std::vector<std::string> igl_ex_str = to_vw_example_format(context, chosen_action, 1.5f, 0.8f);
      std::cout << igl_ex_str << std::endl;
      VW::multi_ex igl_examples;
      for (const std::string& ex : igl_ex_str) { igl_examples.push_back(VW::read_example(*igl_vw, ex)); }
      
    //   // TODO 6 - igl_vw.learn()
      igl_vw->learn(igl_examples);
      igl_vw->finish_example(igl_examples);
    }

    // 7 - calculate ctr
    ctr.push_back(true_reward_sum / static_cast<float>(i));

    // // TODO: Remove this
    // if ( (i != 0) && ((i & (i - 1)) == 0)) {
    //   std::cout << i << "   " << ctr[i] << std::endl;
    // }
  }
  // std::cout << ctr.size() - 1 << "   " << ctr.back() << std::endl; //TODO: Remove
  return ctr;
}
}  // namespace simulator
