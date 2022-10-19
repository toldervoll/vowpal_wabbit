// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "simulator.h"
#include "test_common.h"
#include "vw/core/array_parameters_dense.h"
#include "vw/core/constant.h"  // FNV_prime
#include "vw/core/vw_math.h"

#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test.hpp>
#include <functional>
#include <map>

using simulator::callback_map;
using simulator::cb_sim;

BOOST_AUTO_TEST_CASE(igl_weights_equals_to_separate_vw_instances)
{
  callback_map test_hooks;
  // training policy
  std::string pi_arg =
    "--quiet --cb_explore_adf --coin -q UA "; // --epsilon 0.2

  // decoder policy
  std::string psi_arg =
    "--quiet --link=logistic --loss_function=logistic --coin --cubic UFA ";

  // --readable_model igl.readable
  std::string igl_arg =
    "-f igl.model --invert_hash igl.invert --readable_model igl.readable --quiet --cb_explore_adf -q UA --coin --experimental_igl --noconstant"; // TODO: try -q::, what about noconstant?

  size_t seed = 782391;
  // size_t num_iterations = 800000;
  size_t num_iterations = 1;

  // auto* vw_pi = VW::initialize(pi_arg + "--invert_hash pi.vw");
  // auto* vw_psi = VW::initialize(psi_arg);
  auto* vw_igl = VW::initialize(igl_arg); // + "--invert_hash igl.vw"

  simulator::igl_sim sim1(seed);
  simulator::igl_sim sim2(seed);

  // auto ctr1 = sim1.run_simulation_hook(vw_pi, vw_psi, num_iterations, test_hooks);
  auto ctr2 = sim2.run_simulation_hook(vw_igl, num_iterations, test_hooks);
  VW::finish(*vw_igl);
}

BOOST_AUTO_TEST_CASE(debug_igl_stack)
{
  std::vector<std::string> igl_multi = {
    "0:0.5:0.8 |User user=Tom time_of_day=morning |Action article=sports",
    " |User user=Tom time_of_day=morning |Action article=politics",
    " |User user=Tom time_of_day=morning |Action article=music"
  };

  std::vector<std::string> sl_examples = {
    "1 0.6 |User user=Tom time_of_day=morning |Action article=sports |Feedback click:1",
    "-1 0.6 |User user=Tom time_of_day=morning |Action article=politics |Feedback click:1",
    "-1 0.6 |User user=Tom time_of_day=morning |Action article=music |Feedback click:1"
  };

  // std::vector<std::string> sl_examples = {
  //   "1 0.6 |Feedback click:1",
  //   "-1 0.6 |Feedback click:1",
  //   "-1 0.6 |Feedback click:1"
  // };

//-q UA
  std::string multi_args =
    "-q UA --cb_explore_adf --coin --experimental_igl --noconstant -f igl.model --invert_hash igl.invert --readable_model igl.readable"; // --epsilon 0.2

  std::string sl_args = 
    "--cubic UAF --link=logistic --loss_function=logistic --coin --noconstant -f ik.vw --readable_model ik.readable --invert_hash ik.invert";

  VW::multi_ex igl_example;
  auto* vw_igl = VW::initialize(multi_args);
  for (const std::string& ex : igl_multi) { igl_example.push_back(VW::read_example(*vw_igl, ex)); }
  vw_igl->learn(igl_example);
  vw_igl->finish_example(igl_example);
  VW::finish(*vw_igl);

  auto* vw_sl = VW::initialize(sl_args);

  for (auto& ex_str:sl_examples) {
    VW::example* ex = VW::read_example(*vw_sl, ex_str);
    vw_sl->learn(*ex);
    vw_sl->finish_example(*ex);
  }
  VW::finish(*vw_sl);
}