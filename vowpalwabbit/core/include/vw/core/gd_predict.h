// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once

#include "interactions_predict.h"
#include "vw/core/debug_log.h"
#include "vw/core/example_predict.h"
#include "vw/core/v_array.h"
#include <iostream>
#include <bitset>

#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::gd_predict

namespace GD
{
// iterate through one namespace (or its part), callback function FuncT(some_data_R, feature_value_x, feature_index)
template <class DataT, void (*FuncT)(DataT&, float feature_value, uint64_t feature_index), class WeightsT>
void foreach_feature(WeightsT& /*weights*/, const features& fs, DataT& dat, uint64_t offset = 0, float mult = 1.)
{
  for (const auto& f : fs) { FuncT(dat, mult * f.value(), f.index() + offset); }
}

// iterate through one namespace (or its part), callback function FuncT(some_data_R, feature_value_x, feature_weight)
template <class DataT, void (*FuncT)(DataT&, const float feature_value, float& weight_reference), class WeightsT>
inline void foreach_feature(WeightsT& weights, const features& fs, DataT& dat, uint64_t offset = 0, float mult = 1.)
{
  for (const auto& f : fs)
  {
    weight& w = weights[(f.index() + offset)];
    float* w2 = &weights[(f.index() + offset)];
    std::cout << "\t  [gd pred] WEIGHTS:" << std::bitset<32>(f.index()) << ", ";
    for (size_t i = 0; i < 5; i++) {
      std::cout << w2[i] << ",";
    }
   std::cout << std::endl;
  
    // std::cout << "[gd pred] w: " << w << ", f.indexUNORDER: " << f.index() << ", offset: " << offset << ", mult: " << mult << ", f.value(): " << f.value() << std::endl;
    FuncT(dat, mult * f.value(), w);
  }
}

// iterate through one namespace (or its part), callback function FuncT(some_data_R, feature_value_x, feature_weight)
template <class DataT, void (*FuncT)(DataT&, float, float), class WeightsT>
inline void foreach_feature(
    const WeightsT& weights, const features& fs, DataT& dat, uint64_t offset = 0, float mult = 1.)
{
  for (const auto& f : fs) { FuncT(dat, mult * f.value(), weights[static_cast<size_t>(f.index() + offset)]); }
}

template <class DataT>
inline void dummy_func(DataT&, const VW::audit_strings*)
{
}  // should never be called due to call_audit overload

template <class DataT, class WeightOrIndexT, void (*FuncT)(DataT&, float, WeightOrIndexT),
    class WeightsT>  // nullptr func can't be used as template param in old
                     // compilers

inline void generate_interactions(const std::vector<std::vector<VW::namespace_index>>& interactions,
    const std::vector<std::vector<extent_term>>& extent_interactions, bool permutations, VW::example_predict& ec,
    DataT& dat, WeightsT& weights, size_t& num_interacted_features,
    INTERACTIONS::generate_interactions_object_cache& cache)  // default value removed to eliminate
                                                              // ambiguity in old complers
{
  INTERACTIONS::generate_interactions<DataT, WeightOrIndexT, FuncT, false, dummy_func<DataT>, WeightsT>(
      interactions, extent_interactions, permutations, ec, dat, weights, num_interacted_features, cache);
}

// iterate through all namespaces and quadratic&cubic features, callback function FuncT(some_data_R, feature_value_x,
// WeightOrIndexT) where WeightOrIndexT is EITHER float& feature_weight OR uint64_t feature_index
template <class DataT, class WeightOrIndexT, void (*FuncT)(DataT&, float, WeightOrIndexT), class WeightsT>
inline void foreach_feature(WeightsT& weights, bool ignore_some_linear, std::array<bool, NUM_NAMESPACES>& ignore_linear,
    const std::vector<std::vector<VW::namespace_index>>& interactions,
    const std::vector<std::vector<extent_term>>& extent_interactions, bool permutations, VW::example_predict& ec,
    DataT& dat, size_t& num_interacted_features, INTERACTIONS::generate_interactions_object_cache& cache)
{
  uint64_t offset = ec.ft_offset;
  if (ignore_some_linear)
  {
    for (VW::example_predict::iterator i = ec.begin(); i != ec.end(); ++i)
    {
      if (!ignore_linear[i.index()])
      {
        features& f = *i;
        foreach_feature<DataT, FuncT, WeightsT>(weights, f, dat, offset);
      }
    }
  }
  else
  {
    // auto end = *weights1.end();

    // std::vector<std::tuple<size_t, float, float, float, float, float, float>> weights_vector;

    // while(iter < end) {
    //   bool non_zero = false;
    //   for (int i=0; i < 6; i++) {
    //     if (*iter[i] != 0.f) {
    //       non_zero = true;
    //     }
    //   }

    //   if (non_zero) {
    //     weights_vector.emplace_back(iter.index_without_stride(), *iter[0], *iter[1], *iter[2], *iter[3], *iter[4], *iter[5]);
    //   }
    //   ++iter;
    // }

    // for (auto& w:weights) {
    //   std::cout << w << ", ";
    //   // std::cout << std::get<0>(w) << " "
    //   //   << std::get<1>(w) << " "
    //   //   << std::get<2>(w) << " "
    //   //   << std::get<3>(w) << " "
    //   //   << std::get<4>(w) << " "
    //   //   << std::get<5>(w) << " "
    //   //   << std::get<6>(w) << " "
    //   // << std::endl;
    // }
    std::cout << "[gd predict] ec: " << VW::debug::features_to_string(ec) << std::endl;
    for (features& f : ec) {
      foreach_feature<DataT, FuncT, WeightsT>(weights, f, dat, offset); 
    }
  }

  generate_interactions<DataT, WeightOrIndexT, FuncT, WeightsT>(
      interactions, extent_interactions, permutations, ec, dat, weights, num_interacted_features, cache);
}

template <class DataT, class WeightOrIndexT, void (*FuncT)(DataT&, float, WeightOrIndexT), class WeightsT>
inline void foreach_feature(WeightsT& weights, bool ignore_some_linear, std::array<bool, NUM_NAMESPACES>& ignore_linear,
    const std::vector<std::vector<VW::namespace_index>>& interactions,
    const std::vector<std::vector<extent_term>>& extent_interactions, bool permutations, VW::example_predict& ec,
    DataT& dat, INTERACTIONS::generate_interactions_object_cache& cache)
{
  size_t num_interacted_features_ignored = 0;
  foreach_feature<DataT, WeightOrIndexT, FuncT, WeightsT>(weights, ignore_some_linear, ignore_linear, interactions,
      extent_interactions, permutations, ec, dat, num_interacted_features_ignored, cache);
}

inline void vec_add(float& p, float fx, float fw) { p += fw * fx; }

template <class WeightsT>
inline float inline_predict(WeightsT& weights, bool ignore_some_linear, std::array<bool, NUM_NAMESPACES>& ignore_linear,
    const std::vector<std::vector<VW::namespace_index>>& interactions,
    const std::vector<std::vector<extent_term>>& extent_interactions, bool permutations, VW::example_predict& ec,
    INTERACTIONS::generate_interactions_object_cache& cache, float initial = 0.f)
{
  foreach_feature<float, float, vec_add, WeightsT>(
      weights, ignore_some_linear, ignore_linear, interactions, extent_interactions, permutations, ec, initial, cache);
  return initial;
}

template <class WeightsT>
inline float inline_predict(WeightsT& weights, bool ignore_some_linear, std::array<bool, NUM_NAMESPACES>& ignore_linear,
    const std::vector<std::vector<VW::namespace_index>>& interactions,
    const std::vector<std::vector<extent_term>>& extent_interactions, bool permutations, VW::example_predict& ec,
    size_t& num_interacted_features, INTERACTIONS::generate_interactions_object_cache& cache, float initial = 0.f)
{
  foreach_feature<float, float, vec_add, WeightsT>(weights, ignore_some_linear, ignore_linear, interactions,
      extent_interactions, permutations, ec, initial, num_interacted_features, cache);
  return initial;
}
}  // namespace GD
