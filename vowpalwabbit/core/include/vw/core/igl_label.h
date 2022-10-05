// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include "vw/core/action_score.h"
#include "vw/core/label_parser.h"
#include "vw/core/vw_fwd.h"

#include <cstdint>

namespace VW
{
namespace igl
{
enum class example_type : uint8_t
{
  unset = 0,
  shared = 1,
  action = 2,
  feedback = 3
};

struct label
{
  example_type type;
  // indicates if the example has feedback and the example is for training
  bool labeled;
  float weight;
  label() { reset_to_default(); }

  void reset_to_default()
  {
    type = example_type::unset;
    weight = 1.f;
    labeled = false;
  }
};

void default_label(VW::igl::label& v);
void parse_label(igl::label& ld, VW::label_parser_reuse_mem& reuse_mem, const std::vector<VW::string_view>& words,
    VW::io::logger& logger);

extern VW::label_parser igl_label_parser;
}  // namespace igl

VW::string_view to_string(VW::igl::example_type);

namespace model_utils
{
size_t read_model_field(io_buf&, VW::igl::label&);
size_t write_model_field(io_buf&, const VW::igl::label&, const std::string&, bool);
}  // namespace model_utils
}  // namespace VW

namespace fmt
{
template <>
struct formatter<VW::igl::example_type> : formatter<std::string>
{
  auto format(VW::igl::example_type c, format_context& ctx) -> decltype(ctx.out())
  {
    return formatter<std::string>::format(std::string{VW::to_string(c)}, ctx);
  }
};
}  // namespace fmt