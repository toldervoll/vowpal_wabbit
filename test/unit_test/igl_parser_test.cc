// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "test_common.h"
#include "vw/common/string_view.h"
#include "vw/common/text_utils.h"
#include "vw/core/parse_primitives.h"
#include "vw/core/parser.h"
#include "vw/core/igl_label.h"
#include "vw/io/logger.h"

#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test.hpp>
#include <vector>

void parse_igl_label(VW::string_view label, VW::igl::label& l)
{
  std::vector<VW::string_view> words;
  VW::tokenize(' ', label, words);
  VW::igl::default_label(l);
  VW::reduction_features red_fts;
  VW::label_parser_reuse_mem mem;
  auto null_logger = VW::io::create_null_logger();
  VW::igl::parse_label(l, mem, words, null_logger);
}

BOOST_AUTO_TEST_CASE(igl_parse_label)
{
  {
    VW::igl::label igl_label;
    parse_igl_label("igl shared", igl_label);
    BOOST_CHECK_EQUAL(igl_label.type, VW::igl::example_type::shared);
    BOOST_CHECK_EQUAL(igl_label.labeled, false);
  }
}
