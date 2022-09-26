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
    "--quiet --cb_explore_adf --coin -q UA ";

  // decoder policy
  std::string psi_arg =
    "--quiet --link=logistic --loss_function=logistic --coin --cubic UAF ";

  std::string igl_arg =
    "--quiet --cb_explore_adf --coin "; //TODO: add -q

  int seed = 789343;
  size_t num_iterations = 10000;

  auto* vw_pi = VW::initialize(pi_arg + "--invert_hash pi.vw");
  auto* vw_psi = VW::initialize(psi_arg);
  auto* vw_igl = VW::initialize(igl_arg + "--invert_hash igl.vw");

  simulator::igl_sim sim1(seed);
  simulator::igl_sim sim2(seed);

  auto ctr1 = sim1.run_simulation_hook(vw_pi, vw_psi, num_iterations, test_hooks);

  for (size_t i = 0; i < ctr1.size(); i++) {
    if (i % 100 == 0) {
      std::cout << i << "   " << ctr1[i] << std::endl;
    }
  }
  // auto ctr2 = sim2.run_simulation_hook(vw_igl, num_iterations, test_hooks);
  
}