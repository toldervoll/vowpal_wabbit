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
#include <iostream>

using simulator::callback_map;
using simulator::cb_sim;

using example_vector = std::vector<std::vector<std::string>>;
using ftrl_weights_vector = std::vector<std::tuple<float, float, float, float, float, float>>;

example_vector get_multiline_examples(size_t num) {
  example_vector multi_ex_vector = {
    {
      "shared |User user=Tom time_of_day=morning",
      "0:0.5:0.8 |Action article=sports",
      " |Action article=politics",
      " |Action article=music"
    },
    {
      "shared |User user=Anna time_of_day=afternoon",
      " |Action article=sports",
      "0:-1:0.1 |Action article=politics",
      " |Action article=music"
    }
  };
  if (multi_ex_vector.begin() + num > multi_ex_vector.end()) {
    THROW("number of examples is not valid");
  }
  example_vector result = {multi_ex_vector.begin(), multi_ex_vector.begin() + num};
  return result;
}

example_vector get_sl_examples(size_t num) {
  example_vector sl_ex_vector = {
    {
      "1 0.6 |User user=Tom time_of_day=morning |Action article=sports |Feedback click:1",
      "-1 0.6 |User user=Tom time_of_day=morning |Action article=politics |Feedback click:1",
      "-1 0.6 |User user=Tom time_of_day=morning |Action article=music |Feedback click:1"
    },
    {
      "-1 0.6 |User user=Anna time_of_day=afternoon |Action article=sports |Feedback click:1",
      "1 0.6 |User user=Anna time_of_day=afternoon |Action article=politics |Feedback click:1",
      "-1 0.6 |User user=Anna time_of_day=afternoon |Action article=music |Feedback click:1",
    }
  };

  if (sl_ex_vector.begin() + num > sl_ex_vector.end()) {
    THROW("number of examples is not valid");
  }
  example_vector result = {sl_ex_vector.begin(), sl_ex_vector.begin() + num};
  return result;
}

std::vector<std::string> get_dsjson_examples(size_t num) {
  std::vector<std::string> dsjson_ex_vector = {
    R"({
      "_label_cost": 0.5,
      "_label_probability": 0.8,
      "_label_Action": 1,
      "_labelIndex": 0,
      "o": [
        {
          "v": 1,
          "EventId": "4b49f8f0-92fc-401f-ad08-13fddd99a5cf",
          "ActionTaken": false
        }
      ],
      "Timestamp": "2022-03-09T00:31:34.0000000Z",
      "Version": "1",
      "EventId": "4b49f8f0-92fc-401f-ad08-13fddd99a5cf",
      "a": [
        0,
        1,
        2
      ],
      "c": {
        "User": {
          "user=Tom": "",
          "time_of_day=morning": ""
        },
        "_multi": [
          {
            "Action": {
              "article=sports": ""
            }
          },
          {
            "Action": {
              "article=politics": ""
            }
          },
          {
            "Action": {
              "article=music": ""
            }
          }
        ]
      },
      "p": [
        0.8,
        0.1,
        0.1
      ],
      "VWState": {
        "m": "N/A"
      }
    })",
    R"({
      "_label_cost": 0,
      "_label_probability": 0.1,
      "_label_Action": 2,
      "_labelIndex": 1,
      "o": [
        {
          "v": 1,
          "EventId": "4b49f8f0-92fc-401f-ad08-13fddd99a5cf",
          "ActionTaken": false
        }
      ],
      "Timestamp": "2022-03-11T00:31:34.0000000Z",
      "Version": "1",
      "EventId": "5678f8f0-92fc-401f-ad08-13fddd99a5cf",
      "a": [
        0,
        1,
        2
      ],
      "c": {
        "User": {
          "user=Anna": "",
          "time_of_day=afternoon": ""
        },
        "_multi": [
          {
            "Action": {
              "article=sports": ""
            }
          },
          {
            "Action": {
              "article=politics": ""
            }
          },
          {
            "Action": {
              "article=music": ""
            }
          }
        ]
      },
      "p": [
        0.8,
        0.1,
        0.1
      ],
      "VWState": {
        "m": "N/A"
      }
    })"
  };
  if (dsjson_ex_vector.begin() + num > dsjson_ex_vector.end()) {
    THROW("number of examples is not valid");
  }
  std::vector<std::string> result = {dsjson_ex_vector.begin(), dsjson_ex_vector.begin() + num};
  return result;
}

void print_weights(ftrl_weights_vector weights_vector) {
  for (auto& weights:weights_vector) {
    std::cout << std::get<0>(weights) << " "
      << std::get<1>(weights) << " "
      << std::get<2>(weights) << " "
      << std::get<3>(weights) << " "
      << std::get<4>(weights) << " "
      << std::get<5>(weights) << " "
    << std::endl;
  }
}

ftrl_weights_vector get_weights(VW::workspace* vw) {
  auto& weights = vw->weights.dense_weights;
  auto iter = weights.begin();

  ftrl_weights_vector weights_vector;
  if (*iter != 0.0f) {
    weights_vector.emplace_back(*iter[0], *iter[1], *iter[2], *iter[3], *iter[4], *iter[5]);
  }

  auto end = weights.end();
  // TODO: next_non_zero will skip the entire row of weights if the first element is 0
  // need to fix that
  while (iter.next_non_zero(end) < end) {
    weights_vector.emplace_back(*iter[0], *iter[1], *iter[2], *iter[3], *iter[4], *iter[5]);
  }
  std::sort(weights_vector.begin(), weights_vector.end());
  VW::finish(*vw);
  return weights_vector;
}

ftrl_weights_vector train_multiline_igl(example_vector examples) {
  auto* vw = VW::initialize(
    "--cb_explore_adf --coin --experimental_igl -q UA --noconstant" // --epsilon 0.2
  );

  for (auto& igl_multi_ex_str : examples) {
    VW::multi_ex igl_vw_example;
    for (const std::string& ex : igl_multi_ex_str) {
      igl_vw_example.push_back(VW::read_example(*vw, ex));
    }
    vw->learn(igl_vw_example);
    vw->finish_example(igl_vw_example);
  }

  return get_weights(vw);
}

ftrl_weights_vector train_sl_igl(example_vector sl_examples) {
  auto* vw = VW::initialize(
    "--link=logistic --loss_function=logistic --coin --noconstant --cubic UAF"
  );

  for (auto& sl_ex_str : sl_examples) {
    for (auto& ex_str : sl_ex_str) {
      VW::example* ex = VW::read_example(*vw, ex_str);
      vw->learn(*ex);
      vw->finish_example(*ex);
    }
  }

  return get_weights(vw);
}

ftrl_weights_vector train_dsjson_igl(std::vector<std::string> json_examples) {
  auto* vw = VW::initialize("--cb_explore_adf --coin --experimental_igl -q UA --noconstant --dsjson");
  for (auto& json_text : json_examples) {
    auto examples = parse_dsjson(*vw, json_text);

    vw->learn(examples);
    vw->finish_example(examples);
  }

  return get_weights(vw);
}

BOOST_AUTO_TEST_CASE(igl_learning_converges)
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

BOOST_AUTO_TEST_CASE(verify_igl_model_has_same_weights_as_two_separate_vw_instances)
{
  example_vector igl_multi = get_multiline_examples(1);
  example_vector sl_examples = get_sl_examples(1);

  ftrl_weights_vector sl_weights_vector = train_sl_igl(sl_examples);
  ftrl_weights_vector igl_weights_vector = train_multiline_igl(igl_multi);

  BOOST_CHECK_EQUAL(sl_weights_vector.size(), 3); // TODO: 9 rows starts with 0
  BOOST_CHECK(sl_weights_vector == igl_weights_vector);
  print_weights(igl_weights_vector);
}

BOOST_AUTO_TEST_CASE(verify_igl_weights_with_two_examples)
{
  example_vector igl_examples = get_multiline_examples(2);
  example_vector sl_examples = get_sl_examples(2);

  ftrl_weights_vector igl_weights_vector = train_multiline_igl(igl_examples);
  ftrl_weights_vector sl_weights_vector = train_sl_igl(sl_examples);

  BOOST_CHECK_EQUAL(sl_weights_vector.size(), 8); // all rows are compared
  BOOST_CHECK(sl_weights_vector == igl_weights_vector);
}

BOOST_AUTO_TEST_CASE(one_dsjson_example_equals_to_multiline_example)
{
  std::vector<std::string> dsjson_examples = get_dsjson_examples(1);
  ftrl_weights_vector dsjson_weights_vector = train_dsjson_igl(dsjson_examples);
  example_vector multi_examples = get_multiline_examples(1);
  ftrl_weights_vector multi_weights_vector = train_multiline_igl(multi_examples);

  BOOST_CHECK_EQUAL(multi_weights_vector.size(), 3);
  BOOST_CHECK(dsjson_weights_vector == multi_weights_vector);
}

BOOST_AUTO_TEST_CASE(two_dsjson_examples_equal_to_multiline_examples)
{
  std::vector<std::string> dsjson_examples = get_dsjson_examples(2);
  ftrl_weights_vector dsjson_weights_vector = train_dsjson_igl(dsjson_examples);
  example_vector multi_examples = get_multiline_examples(2);
  ftrl_weights_vector multi_weights_vector = train_multiline_igl(multi_examples);

  BOOST_CHECK_EQUAL(multi_weights_vector.size(), 8);
  BOOST_CHECK(dsjson_weights_vector == multi_weights_vector);
  print_weights(dsjson_weights_vector);
}