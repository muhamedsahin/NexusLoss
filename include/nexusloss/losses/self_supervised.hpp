/**
 * @file self_supervised.hpp
 * @brief Etiketsiz temsil ogrenimi icin kontrastif ve cift-goruntu kayiplari.
 *
 * InfoNCE/NT-Xent benzerlik skorlari ve sicaklik alir; payda log-sum-exp'tir.
 * BYOL negatif ornek istemez. Barlow Twins kare capraz-korelasyon matrisi bekler.
 *
 * @see van den Oord et al., CPC / InfoNCE, 2018
 * @see Zbontar et al., Barlow Twins, 2021
 */
#ifndef NEXUSLOSS_SELF_SUPERVISED_HPP
#define NEXUSLOSS_SELF_SUPERVISED_HPP

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>
#include "../core/utils.hpp"
#include "metric.hpp"

namespace nexusloss::self_supervised {

template<typename T>
T info_nce_loss(T positive_similarity, std::span<const T> negative_similarities, T temperature = T(0.1)) {
    if (temperature <= T(0)) throw std::invalid_argument("NexusLoss: InfoNCE temperature must be positive");
    T maximum = positive_similarity / temperature;
    for (T value : negative_similarities) maximum = std::max(maximum, value / temperature);
    T denominator = std::exp(positive_similarity / temperature - maximum);
    for (T value : negative_similarities) denominator += std::exp(value / temperature - maximum);
    return maximum - positive_similarity / temperature + std::log(denominator);
}

template<typename T>
T nt_xent_loss(T positive_similarity, std::span<const T> negative_similarities, T temperature = T(0.1)) {
    return info_nce_loss(positive_similarity, negative_similarities, temperature);
}

template<typename T>
T byol_loss(std::span<const T> prediction, std::span<const T> target) {
    return T(2) - T(2) * metric::cosine_similarity(prediction, target);
}

template<typename T>
T dino_loss(std::span<const T> student_logits, std::span<const T> teacher_probabilities,
            T student_temperature = T(0.1)) {
    if (student_logits.empty() || student_logits.size() != teacher_probabilities.size()
        || student_temperature <= T(0))
        throw std::invalid_argument("NexusLoss: invalid DINO inputs or temperature");
    const T maximum = *std::max_element(student_logits.begin(), student_logits.end()) / student_temperature;
    T exp_sum = T(0), loss = T(0);
    for (T logit : student_logits) exp_sum += std::exp(logit / student_temperature - maximum);
    const T log_z = maximum + std::log(exp_sum);
    for (size_t i = 0; i < student_logits.size(); ++i) {
        if (teacher_probabilities[i] < T(0)) throw std::invalid_argument("NexusLoss: teacher probabilities must be non-negative");
        loss -= teacher_probabilities[i] * (student_logits[i] / student_temperature - log_z);
    }
    return loss;
}

template<typename T>
T barlow_twins_loss(std::span<const T> cross_correlation, size_t dimensions,
                    T lambda = T(0.005)) {
    if (!dimensions || cross_correlation.size() != dimensions * dimensions || lambda < T(0))
        throw std::invalid_argument("NexusLoss: Barlow Twins requires a square correlation matrix");
    T loss = T(0);
    for (size_t i = 0; i < dimensions; ++i)
        for (size_t j = 0; j < dimensions; ++j) {
            const T error = cross_correlation[i * dimensions + j] - (i == j ? T(1) : T(0));
            loss += (i == j ? T(1) : lambda) * error * error;
        }
    return loss;
}

template<typename T>
T vicreg_loss(std::span<const T> view_a, std::span<const T> view_b, size_t samples, size_t dimensions,
              T invariance_weight = T(25), T variance_weight = T(25), T covariance_weight = T(1),
              T gamma = T(1)) {
    if (!samples || !dimensions || view_a.size() != samples * dimensions || view_b.size() != view_a.size())
        throw std::invalid_argument("NexusLoss: VICReg inputs must be flattened sample-major matrices");
    T invariance = T(0);
    for (size_t i = 0; i < view_a.size(); ++i) { const T d = view_a[i] - view_b[i]; invariance += d * d; }
    invariance /= static_cast<T>(view_a.size());
    std::vector<T> means_a(dimensions, T(0)), means_b(dimensions, T(0));
    for (size_t n = 0; n < samples; ++n)
        for (size_t d = 0; d < dimensions; ++d) {
            means_a[d] += view_a[n * dimensions + d] / static_cast<T>(samples);
            means_b[d] += view_b[n * dimensions + d] / static_cast<T>(samples);
        }
    std::vector<T> var_a(dimensions, T(0)), var_b(dimensions, T(0));
    for (size_t n = 0; n < samples; ++n)
        for (size_t d = 0; d < dimensions; ++d) {
            const T da = view_a[n * dimensions + d] - means_a[d];
            const T db = view_b[n * dimensions + d] - means_b[d];
            var_a[d] += da * da / static_cast<T>(samples);
            var_b[d] += db * db / static_cast<T>(samples);
        }
    T variance = T(0);
    for (size_t d = 0; d < dimensions; ++d)
        variance += std::max(T(0), gamma - std::sqrt(var_a[d] + core::epsilon<T>))
                  + std::max(T(0), gamma - std::sqrt(var_b[d] + core::epsilon<T>));
    variance /= static_cast<T>(dimensions);
    T covariance = T(0);
    if (samples > 1) {
        for (size_t i = 0; i < dimensions; ++i)
            for (size_t j = 0; j < dimensions; ++j) if (i != j) {
                T ca = T(0), cb = T(0);
                for (size_t n = 0; n < samples; ++n) {
                    ca += (view_a[n * dimensions + i] - means_a[i]) * (view_a[n * dimensions + j] - means_a[j]);
                    cb += (view_b[n * dimensions + i] - means_b[i]) * (view_b[n * dimensions + j] - means_b[j]);
                }
                ca /= static_cast<T>(samples - 1);
                cb /= static_cast<T>(samples - 1);
                covariance += ca * ca + cb * cb;
            }
        covariance /= static_cast<T>(dimensions);
    }
    return invariance_weight * invariance + variance_weight * variance + covariance_weight * covariance;
}

/// @brief InfoNCE/NT-Xent turevi. Ilk eleman pozitif benzerlige, sonrakiler negatiflere aittir.
template<typename T>
std::vector<T> info_nce_gradient(T positive_similarity, std::span<const T> negative_similarities,
                                 T temperature = T(0.1)) {
    if (temperature <= T(0)) throw std::invalid_argument("NexusLoss: InfoNCE temperature must be positive");
    T maximum = positive_similarity / temperature;
    for (T value : negative_similarities) maximum = std::max(maximum, value / temperature);
    std::vector<T> weights(negative_similarities.size() + 1);
    weights[0] = std::exp(positive_similarity / temperature - maximum);
    T sum = weights[0];
    for (size_t i = 0; i < negative_similarities.size(); ++i) {
        weights[i + 1] = std::exp(negative_similarities[i] / temperature - maximum);
        sum += weights[i + 1];
    }
    std::vector<T> grad(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) grad[i] = (weights[i] / sum) / temperature;
    grad[0] -= T(1) / temperature;
    return grad;
}

template<typename T>
std::vector<T> byol_gradient(std::span<const T> prediction, std::span<const T> target) {
    const T dot = metric::dot(prediction, target);
    const T pred_norm = std::sqrt(std::max(metric::dot(prediction, prediction), core::epsilon<T>));
    const T target_norm = std::sqrt(std::max(metric::dot(target, target), core::epsilon<T>));
    const T cosine = dot / (pred_norm * target_norm);
    std::vector<T> grad(prediction.size());
    for (size_t i = 0; i < prediction.size(); ++i) {
        const T d_cosine = (target[i] / target_norm) / pred_norm
                           - cosine * prediction[i] / (pred_norm * pred_norm);
        grad[i] = T(-2) * d_cosine;
    }
    return grad;
}

template<typename T>
std::vector<T> barlow_twins_gradient(std::span<const T> cross_correlation, size_t dimensions, T lambda = T(0.005)) {
    if (!dimensions || cross_correlation.size() != dimensions * dimensions || lambda < T(0))
        throw std::invalid_argument("NexusLoss: Barlow Twins requires a square correlation matrix");
    std::vector<T> grad(cross_correlation.size());
    for (size_t i = 0; i < dimensions; ++i)
        for (size_t j = 0; j < dimensions; ++j) {
            const T target = i == j ? T(1) : T(0);
            const T weight = i == j ? T(1) : lambda;
            grad[i * dimensions + j] = T(2) * weight * (cross_correlation[i * dimensions + j] - target);
        }
    return grad;
}

/**
 * @class InfoNCELoss
 * @brief Pozitif cifti, negatifler arasinda ayirt eden kontrastif kayip (van den Oord, 2018).
 * L = -log exp(s+/tau) / (exp(s+/tau) + sum exp(s-/tau))
 * Payda log-sum-exp ile hesaplanir.
 */
template<typename T>
class InfoNCELoss {
public:
    explicit InfoNCELoss(T temperature = T(0.1)) : temperature_(temperature) {}
    T forward(T positive, std::span<const T> negatives) {
        positive_ = positive;
        negatives_.assign(negatives.begin(), negatives.end());
        ready_ = true;
        return info_nce_loss(positive, negatives, temperature_);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return info_nce_gradient(positive_, std::span<const T>(negatives_), temperature_);
    }
private:
    T temperature_, positive_ = T(0);
    std::vector<T> negatives_;
    bool ready_ = false;
};

template<typename T>
class NTXentLoss : public InfoNCELoss<T> {
public:
    using InfoNCELoss<T>::InfoNCELoss;
};

template<typename T>
class BYOLLoss {
public:
    T forward(std::span<const T> prediction, std::span<const T> target) {
        prediction_.assign(prediction.begin(), prediction.end());
        target_.assign(target.begin(), target.end());
        ready_ = true;
        return byol_loss(prediction, target);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return byol_gradient<T>(prediction_, target_);
    }
private:
    std::vector<T> prediction_, target_;
    bool ready_ = false;
};

template<typename T>
class BarlowTwinsLoss {
public:
    explicit BarlowTwinsLoss(T lambda = T(0.005)) : lambda_(lambda) {}
    T forward(std::span<const T> cross_correlation, size_t dimensions) {
        matrix_.assign(cross_correlation.begin(), cross_correlation.end());
        dimensions_ = dimensions;
        ready_ = true;
        return barlow_twins_loss(cross_correlation, dimensions, lambda_);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return barlow_twins_gradient<T>(matrix_, dimensions_, lambda_);
    }
private:
    T lambda_;
    std::vector<T> matrix_;
    size_t dimensions_ = 0;
    bool ready_ = false;
};

} // namespace nexusloss::self_supervised

#endif
