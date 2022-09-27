// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include "test_common.h"
#include "vw/core/rand_state.h"

#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test.hpp>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace VW
{
struct workspace;
}

namespace simulator
{
class cb_sim;
// maps an int: # learned examples
// with a function to 'test' at that point in time in the simulator
using callback_map = typename std::map<size_t, std::function<bool(cb_sim&, VW::workspace&, VW::multi_ex&)>>;

class cb_sim
{
  const float USER_LIKED_ARTICLE = -1.f;
  const float USER_DISLIKED_ARTICLE = 0.f;
  const std::vector<std::string> users;
  const std::vector<std::string> times_of_day;
  float cost_sum = 0.f;
  std::vector<float> ctr;
  size_t callback_count;

public:
  std::string user_ns;
  std::string action_ns;
  VW::rand_state random_state;
  const std::vector<std::string> actions;

  cb_sim(uint64_t seed = 0);
  float get_reaction(const std::map<std::string, std::string>& context, const std::string& action,
      bool add_noise = false, bool swap_reward = false);
  std::vector<std::string> to_vw_example_format(const std::map<std::string, std::string>& context,
      const std::string& chosen_action, float cost = 0.f, float prob = 0.f);
  std::pair<int, float> sample_custom_pmf(std::vector<float>& pmf);
  std::vector<float> get_pmf(VW::workspace* vw, const std::map<std::string, std::string>& context);
  const std::string& choose_user();
  const std::string& choose_time_of_day();
  std::vector<float> run_simulation(VW::workspace* vw, size_t num_iterations, bool do_learn = true, size_t shift = 1,
      const std::vector<uint64_t>& swap_after = std::vector<uint64_t>());
  std::vector<float> run_simulation_hook(VW::workspace* vw, size_t num_iterations, callback_map& callbacks,
      bool do_learn = true, size_t shift = 1, bool add_noise = false, uint64_t num_useless_features = 0,
      const std::vector<uint64_t>& swap_after = std::vector<uint64_t>());

private:
  void call_if_exists(VW::workspace& vw, VW::multi_ex& ex, const callback_map& callbacks, const size_t event);
};

class igl_sim : public cb_sim {
  float true_reward_sum = 0.f;
  std::vector<float> ctr;
  const std::map<std::string, std::string> ground_truth_enjoy = {
    {"Tom", "politics"},
    {"Anna", "music"}
  };

  const std::map<std::string, std::string> ground_truth_hate = {
    {"Tom", "music"},
    {"Anna", "sports"}
  };

  // assume two users communicate their satisfaction in the same way
  const std::map<std::string, float> enjoy_prob = {
    {"dislike", 0}, {"skip", 0}, {"click", 0.5}, {"like", 0.5}, {"none", 0}
  };
  const std::map<std::string, float> hate_prob = {
    {"dislike", 0.1}, {"skip", 0.9}, {"click", 0}, {"like", 0}, {"none", 0}
  };
  const std::map<std::string, float> neutral_prob = {
    {"dislike", 0}, {"skip", 0}, {"click", 0}, {"like", 0}, {"none", 1}
  };

  std::string feedback_ns;
  const float p_unlabeled_prior = 0.5;

public:
  igl_sim(uint64_t seed = 0);
  std::vector<float> run_simulation_hook(VW::workspace* pi, VW::workspace* psi, size_t num_iterations,
      callback_map& callbacks, bool do_learn = true);
  std::string sample_feedback(const std::map<std::string, float>& probs);
  std::string get_feedback(const std::string& pref, const std::string& chosen_action);
  std::string to_psi_predict_ex(const std::map<std::string, std::string>& context,
      const std::string& chosen_action, const std::string& feedback);
  std::vector<std::string> to_psi_learn_ex(const std::map<std::string, std::string>& context,
      const std::string& chosen_action, const std::string& feedback, const std::vector<float>& pmf);
  float true_reward(const std::string& user, const std::string& action);
};

std::vector<float> _test_helper(const std::string& vw_arg, size_t num_iterations = 3000, int seed = 10);
std::vector<float> _test_helper_save_load(const std::string& vw_arg, size_t num_iterations = 3000, int seed = 10,
    const std::vector<uint64_t>& swap_after = std::vector<uint64_t>(), const size_t split = 1500);
std::vector<float> _test_helper_hook(const std::string& vw_arg, callback_map& hooks, size_t num_iterations = 3000,
    int seed = 10, const std::vector<uint64_t>& swap_after = std::vector<uint64_t>());
}  // namespace simulator
