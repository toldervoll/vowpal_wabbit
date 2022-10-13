// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include "vw/core/vw_fwd.h"
#include "vw/core/reduction_stack.h"
#include <set>
#include <vector>
#include "vw/core/constant.h"

namespace VW
{
namespace reductions
{
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

  VW::LEARNER::base_learner* setup_base_learner() override {
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
/**
 * Setup interaction grounded learning reduction. Wiki page:
 * https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Interaction-Grounded-Learning
 *
 * @param stack_builder Stack builder to use for setup.
 * @return VW::LEARNER::base_learner* learner if this reduction is active, nullptr otherwise
 */
VW::LEARNER::base_learner* interaction_ground_setup(VW::setup_base_i& stack_builder);
}  // namespace reductions
}  // namespace VW