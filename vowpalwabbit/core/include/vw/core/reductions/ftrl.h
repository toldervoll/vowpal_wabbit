// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include "vw/core/vw_fwd.h"

struct ftrl_update_data
{
  float update = 0.f;
  float ftrl_alpha = 0.f;
  float ftrl_beta = 0.f;
  float l1_lambda = 0.f;
  float l2_lambda = 0.f;
  float predict = 0.f;
  float normalized_squared_norm_x = 0.f;
  float average_squared_norm_x = 0.f;
};

struct ftrl
{
  VW::workspace* all = nullptr;  // features, finalize, l1, l2,
  float ftrl_alpha = 0.f;
  float ftrl_beta = 0.f;
  ftrl_update_data data;
  size_t no_win_counter = 0;
  size_t early_stop_thres = 0;
  uint32_t ftrl_size = 0;
  double total_weight = 0.0;
  double normalized_sum_norm_x = 0.0;
};

namespace VW
{
namespace reductions
{
VW::LEARNER::base_learner* ftrl_setup(VW::setup_base_i& stack_builder);
}
}  // namespace VW
