/**
 * @file fused_simd.cpp
 * @brief MSE, MAE, Huber ve logit-BCE icin tek geciste kayip ve gradyan.
 * AVX2 yoksa skaler kalan ayni formulu kullanir.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <immintrin.h>

namespace nexusloss {
namespace fused {
namespace {

bool avx2_enabled() {
    int info[4] = {};
#if defined(_MSC_VER)
    __cpuid(info, 1);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    if (!osxsave) {
        return false;
    }
    const unsigned long long xcr = _xgetbv(0);
    if ((xcr & 0x6ull) != 0x6ull) {
        return false;
    }
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#else
    return false;
#endif
}

void scale_gradient(__m256 scale, float* grad, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(grad + i, _mm256_mul_ps(_mm256_loadu_ps(grad + i), scale));
    }
    const float scalar = _mm_cvtss_f32(_mm256_castps256_ps128(scale));
    for (; i < n; ++i) {
        grad[i] *= scalar;
    }
}

float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

}  // namespace

double mse(const float* pred, const float* target, float* grad, std::size_t n, int reduction) {
    double sum = 0.0;
    std::size_t i = 0;
    if (avx2_enabled()) {
        __m256 acc = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) {
            const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(pred + i), _mm256_loadu_ps(target + i));
            acc = _mm256_fmadd_ps(diff, diff, acc);
            _mm256_storeu_ps(grad + i, _mm256_mul_ps(_mm256_set1_ps(2.0f), diff));
        }
        sum = static_cast<double>(hsum(acc));
    }
    for (; i < n; ++i) {
        const float diff = pred[i] - target[i];
        sum += static_cast<double>(diff) * static_cast<double>(diff);
        grad[i] = 2.0f * diff;
    }
    if (reduction == 0 && n > 0) {
        const float scale = 1.0f / static_cast<float>(n);
        if (avx2_enabled()) {
            scale_gradient(_mm256_set1_ps(scale), grad, n);
        } else {
            for (std::size_t k = 0; k < n; ++k) {
                grad[k] *= scale;
            }
        }
        return sum / static_cast<double>(n);
    }
    return sum;
}

double mae(const float* pred, const float* target, float* grad, std::size_t n, int reduction) {
    double sum = 0.0;
    std::size_t i = 0;
    if (avx2_enabled()) {
        const __m256 sign = _mm256_set1_ps(-0.0f);
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 minus = _mm256_set1_ps(-1.0f);
        const __m256 zero = _mm256_setzero_ps();
        __m256 acc = zero;
        for (; i + 8 <= n; i += 8) {
            const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(pred + i), _mm256_loadu_ps(target + i));
            acc = _mm256_add_ps(acc, _mm256_andnot_ps(sign, diff));
            const __m256 positive = _mm256_cmp_ps(diff, zero, _CMP_GT_OQ);
            const __m256 negative = _mm256_cmp_ps(diff, zero, _CMP_LT_OQ);
            const __m256 direction = _mm256_or_ps(_mm256_and_ps(positive, one), _mm256_and_ps(negative, minus));
            _mm256_storeu_ps(grad + i, direction);
        }
        sum = static_cast<double>(hsum(acc));
    }
    for (; i < n; ++i) {
        const float diff = pred[i] - target[i];
        sum += static_cast<double>(std::fabs(diff));
        grad[i] = diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
    }
    if (reduction == 0 && n > 0) {
        const float scale = 1.0f / static_cast<float>(n);
        if (avx2_enabled()) {
            scale_gradient(_mm256_set1_ps(scale), grad, n);
        } else {
            for (std::size_t k = 0; k < n; ++k) {
                grad[k] *= scale;
            }
        }
        return sum / static_cast<double>(n);
    }
    return sum;
}

double huber(const float* pred, const float* target, float* grad, std::size_t n, float delta, int reduction) {
    double sum = 0.0;
    std::size_t i = 0;
    if (avx2_enabled()) {
        const __m256 sign = _mm256_set1_ps(-0.0f);
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 delta_v = _mm256_set1_ps(delta);
        const __m256 half_delta = _mm256_set1_ps(0.5f * delta);
        __m256 acc = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) {
            const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(pred + i), _mm256_loadu_ps(target + i));
            const __m256 magnitude = _mm256_andnot_ps(sign, diff);
            const __m256 inside = _mm256_cmp_ps(magnitude, delta_v, _CMP_LE_OQ);
            const __m256 quadratic = _mm256_mul_ps(half, _mm256_mul_ps(diff, diff));
            const __m256 linear = _mm256_mul_ps(delta_v, _mm256_sub_ps(magnitude, half_delta));
            acc = _mm256_add_ps(acc, _mm256_blendv_ps(linear, quadratic, inside));
            const __m256 outer = _mm256_or_ps(_mm256_and_ps(diff, sign), delta_v);
            _mm256_storeu_ps(grad + i, _mm256_blendv_ps(outer, diff, inside));
        }
        sum = static_cast<double>(hsum(acc));
    }
    for (; i < n; ++i) {
        const float diff = pred[i] - target[i];
        const float magnitude = std::fabs(diff);
        if (magnitude <= delta) {
            sum += 0.5 * static_cast<double>(diff) * static_cast<double>(diff);
            grad[i] = diff;
        } else {
            sum += static_cast<double>(delta) * (static_cast<double>(magnitude) - 0.5 * static_cast<double>(delta));
            grad[i] = diff > 0.0f ? delta : -delta;
        }
    }
    if (reduction == 0 && n > 0) {
        const float scale = 1.0f / static_cast<float>(n);
        if (avx2_enabled()) {
            scale_gradient(_mm256_set1_ps(scale), grad, n);
        } else {
            for (std::size_t k = 0; k < n; ++k) {
                grad[k] *= scale;
            }
        }
        return sum / static_cast<double>(n);
    }
    return sum;
}

double bce_with_logits(const float* pred, const float* target, float* grad, std::size_t n, int reduction) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const float x = pred[i];
        const float y = target[i];
        const float loss = std::max(x, 0.0f) - x * y + std::log1p(std::exp(-std::fabs(x)));
        sum += static_cast<double>(loss);
        const float sig = x >= 0.0f ? 1.0f / (1.0f + std::exp(-x)) : std::exp(x) / (1.0f + std::exp(x));
        grad[i] = sig - y;
    }
    if (reduction == 0 && n > 0) {
        const float scale = 1.0f / static_cast<float>(n);
        for (std::size_t i = 0; i < n; ++i) {
            grad[i] *= scale;
        }
        return sum / static_cast<double>(n);
    }
    return sum;
}

}  // namespace fused
}  // namespace nexusloss
