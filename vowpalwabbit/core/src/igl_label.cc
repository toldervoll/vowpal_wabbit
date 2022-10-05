// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "vw/core/igl_label.h"

#include "vw/common/string_view.h"
#include "vw/common/text_utils.h"
#include "vw/core/cache.h"
#include "vw/core/constant.h"
#include "vw/core/model_utils.h"
#include "vw/core/parse_primitives.h"
#include "vw/core/parser.h"
#include "vw/core/vw_math.h"

#include <numeric>

namespace VW
{
namespace igl
{
void default_label(igl::label& v);

float weight(const igl::label& ld) { return ld.weight; }

void default_label(igl::label& ld) { ld.reset_to_default(); }

bool test_label(const igl::label& ld) { return ld.labeled == false; }

void parse_label(igl::label& ld, VW::label_parser_reuse_mem& reuse_mem, const std::vector<VW::string_view>& words,
    VW::io::logger& logger)
{
  ld.weight = 1;

  if (words.empty()) { THROW("IGL labels may not be empty"); }
  if (!(words[0] == IGL_LABEL)) { THROW("IGL labels require the first word to be igl"); }

  if (words.size() == 1) { THROW("IGL labels require a type. It must be one of: [shared, action, feedback]"); }

  const auto& type = words[1];
  if (type == SHARED_TYPE)
  {
    if (words.size() != 2)
    {
      THROW("IGL shared labels must be of the form: igl shared");
    }

    ld.type = example_type::shared;
  }
  else if (type == ACTION_TYPE)
  {
    if (words.size() == 3) {
      ld.prob = float_of_string(words[2], logger);
    }
    else if (words.size() != 2) {
      THROW("IGL action labels must be of the form: igl action [chosen_action_probability]");
    }
    ld.type = example_type::action;
  }

  else
  {
    THROW("Unknown IGL label type: " << type);
  }
}

label_parser igl_label_parser = {
    // default_label
    [](polylabel& label) { default_label(label.igl); },
    // parse_label
    [](polylabel& label, reduction_features& /* red_features */, VW::label_parser_reuse_mem& reuse_mem,
        const VW::named_labels* /* ldict */, const std::vector<VW::string_view>& words,
        VW::io::logger& logger) { parse_label(label.igl, reuse_mem, words, logger); },
    // cache_label
    [](const polylabel& label, const reduction_features& /* red_features */, io_buf& cache,
        const std::string& upstream_name,
        bool text) { return VW::model_utils::write_model_field(cache, label.igl, upstream_name, text); },
    // read_cached_label
    [](polylabel& label, reduction_features& /* red_features */, io_buf& cache) {
      return VW::model_utils::read_model_field(cache, label.igl);
    },
    // get_weight
    [](const polylabel& label, const reduction_features& /* red_features */) { return weight(label.igl); },
    // test_label
    [](const polylabel& label) { return test_label(label.igl); },
    // label type
    label_type_t::igl};
} // namespace igl
}  // namespace VW

VW::string_view VW::to_string(VW::igl::example_type ex_type)
{
#define CASE(type) \
  case type:       \
    return #type;

  using namespace VW::igl;
  switch (ex_type)
  {
    CASE(example_type::unset)
    CASE(example_type::shared)
    CASE(example_type::action)
    CASE(example_type::feedback)
  }

  // The above enum is exhaustive and will warn on a new label type being added due to the lack of `default`
  // The following is required by the compiler, otherwise it things control can reach the end of this function without
  // returning.
  assert(false);
  return "unknown example_type enum";

#undef CASE
}

namespace VW
{
namespace model_utils
{
size_t read_model_field(io_buf& io, VW::igl::label& igl_label)
{
  // Since read_cached_features doesn't default the label we must do it here.
  size_t bytes = 0;
  bytes += read_model_field(io, igl_label.type);
  bytes += read_model_field(io, igl_label.weight);
  bytes += read_model_field(io, igl_label.labeled);
  // bytes += read_model_field(io, slates.cost);
  // bytes += read_model_field(io, slates.slot_id);
  // bytes += read_model_field(io, slates.probabilities);
  return bytes;
}

size_t write_model_field(io_buf& io, const VW::igl::label& igl_label, const std::string& upstream_name, bool text)
{
  size_t bytes = 0;
  bytes += write_model_field(io, igl_label.type, upstream_name + "_type", text);
  bytes += write_model_field(io, igl_label.weight, upstream_name + "_weight", text);
  bytes += write_model_field(io, igl_label.labeled, upstream_name + "_labeled", text);
  // bytes += write_model_field(io, slates.cost, upstream_name + "_cost", text);
  // bytes += write_model_field(io, slates.slot_id, upstream_name + "_slot_id", text);
  // bytes += write_model_field(io, slates.probabilities, upstream_name + "_probabilities", text);
  return bytes;
}
}  // namespace model_utils
} // namespace vw