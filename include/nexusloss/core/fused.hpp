#pragma once

/**
 * @file fused.hpp
 * @brief Tek geciste kayip degeri ve gradyan. AVX2 yolu ayri ceviri birimindedir.
 */

#include <cstddef>

namespace nexusloss {
namespace fused {

/// reduction: 0 mean, diger sum. Gradyan Mean'de 1/N ile olceklenir.
double mse(const float* pred, const float* target, float* grad, std::size_t n, int reduction);
double mae(const float* pred, const float* target, float* grad, std::size_t n, int reduction);
double huber(const float* pred, const float* target, float* grad, std::size_t n, float delta, int reduction);
double bce_with_logits(const float* pred, const float* target, float* grad, std::size_t n, int reduction);

}  // namespace fused
}  // namespace nexusloss
