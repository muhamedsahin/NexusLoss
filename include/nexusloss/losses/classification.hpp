/**
 * @file classification.hpp
 * @brief Sınıflandırma loss aileleri: ikili, çoklu-sınıf ve dengesiz veri destekli formlar.
 *
 * Bu dosya BCE, CCE, focal, hinge ve label smoothing gibi klasik sınıflandırma loss'larını
 * içerir. Dengesizlik ve gürültülü etiketlere karşı dayanıklı formlar da burada yer alır.
 */

#ifndef NEXUSLOSS_CLASSIFICATION_HPP
#define NEXUSLOSS_CLASSIFICATION_HPP

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>
#include "loss_base.hpp"

namespace nexusloss {
namespace classification {

template<typename T>
T binary_cross_entropy(T probability, T target) {
    if (target < T(0) || target > T(1))
        throw std::invalid_argument("NexusLoss: BCE targets must be in [0, 1]");
    const T p = core::clamp(probability, core::epsilon<T>, T(1) - core::epsilon<T>);
    return -(target * std::log(p) + (T(1) - target) * std::log1p(-p));
}

template<typename T>
T binary_cross_entropy_with_logits(T logit, T target) {
    if (target < T(0) || target > T(1))
        throw std::invalid_argument("NexusLoss: BCE targets must be in [0, 1]");
    return std::max(logit, T(0)) - logit * target + std::log1p(std::exp(-std::abs(logit)));
}

template<typename T>
T categorical_cross_entropy(std::span<const T> logits, std::span<const T> target,
                            T label_smoothing = T(0)) {
    if (logits.empty() || logits.size() != target.size())
        throw std::invalid_argument("NexusLoss: categorical cross entropy requires equal non-empty vectors");
    if (label_smoothing < T(0) || label_smoothing >= T(1))
        throw std::invalid_argument("NexusLoss: label smoothing must be in [0, 1)");
    const T maximum = *std::max_element(logits.begin(), logits.end());
    T exp_sum = T(0), target_sum = T(0), weighted_logit = T(0);
    for (size_t i = 0; i < logits.size(); ++i) {
        exp_sum += std::exp(logits[i] - maximum);
        target_sum += target[i];
    }
    if (!(target_sum > T(0))) throw std::invalid_argument("NexusLoss: categorical target must have positive mass");
    const T log_normalizer = maximum + std::log(exp_sum);
    for (size_t i = 0; i < logits.size(); ++i) {
        if (target[i] < T(0)) throw std::invalid_argument("NexusLoss: categorical targets must be non-negative");
        const T normalized_target = target[i] / target_sum;
        const T smooth_target = (T(1) - label_smoothing) * normalized_target
                                + label_smoothing / static_cast<T>(logits.size());
        weighted_logit += smooth_target * (logits[i] - log_normalizer);
    }
    return -weighted_logit;
}

template<typename T>
T sparse_categorical_cross_entropy(std::span<const T> logits, size_t target,
                                   T label_smoothing = T(0)) {
    if (target >= logits.size()) throw std::out_of_range("NexusLoss: class index is outside logits");
    std::vector<T> one_hot(logits.size(), T(0));
    one_hot[target] = T(1);
    return categorical_cross_entropy<T>(logits, one_hot, label_smoothing);
}

template<typename T>
T label_smoothing_cross_entropy(std::span<const T> logits, size_t target, T smoothing = T(0.1)) {
    return sparse_categorical_cross_entropy<T>(logits, target, smoothing);
}

template<typename T>
T focal_loss(T probability, T target, T alpha = T(0.25), T gamma = T(2)) {
    if (alpha < T(0) || alpha > T(1) || gamma < T(0))
        throw std::invalid_argument("NexusLoss: focal alpha must be in [0, 1] and gamma non-negative");
    if (target < T(0) || target > T(1))
        throw std::invalid_argument("NexusLoss: focal targets must be in [0, 1]");
    const T p = core::clamp(probability, core::epsilon<T>, T(1) - core::epsilon<T>);
    const T pt = target >= T(0.5) ? p : T(1) - p;
    const T alpha_t = target >= T(0.5) ? alpha : T(1) - alpha;
    return -alpha_t * std::pow(T(1) - pt, gamma) * std::log(pt);
}

template<typename T>
T categorical_focal_loss(std::span<const T> logits, size_t target, T alpha = T(1), T gamma = T(2)) {
    if (target >= logits.size() || alpha < T(0) || gamma < T(0))
        throw std::invalid_argument("NexusLoss: invalid categorical focal loss parameters");
    const T cross_entropy = sparse_categorical_cross_entropy<T>(logits, target);
    return alpha * std::pow(T(1) - std::exp(-cross_entropy), gamma) * cross_entropy;
}

template<typename T>
T generalized_cross_entropy(T probability, T target, T q = T(0.7)) {
    if (!(q > T(0) && q <= T(1))) throw std::invalid_argument("NexusLoss: GCE q must be in (0, 1]");
    const T p = core::clamp(probability, core::epsilon<T>, T(1));
    const T pt = target >= T(0.5) ? p : T(1) - p;
    return (T(1) - std::pow(pt, q)) / q;
}

template<typename T>
T symmetric_cross_entropy(T probability, T target, T alpha = T(1), T beta = T(1)) {
    const T p = core::clamp(probability, core::epsilon<T>, T(1) - core::epsilon<T>);
    const T y = core::clamp(target, core::epsilon<T>, T(1) - core::epsilon<T>);
    return -alpha * (target * std::log(p) + (T(1) - target) * std::log1p(-p))
           -beta * (p * std::log(y) + (T(1) - p) * std::log1p(-y));
}

template<typename T>
T weighted_binary_cross_entropy(T probability, T target, T positive_weight, T negative_weight = T(1)) {
    if (positive_weight < T(0) || negative_weight < T(0))
        throw std::invalid_argument("NexusLoss: class weights must be non-negative");
    return positive_weight * target * -std::log(core::clamp(probability, core::epsilon<T>, T(1)))
           + negative_weight * (T(1) - target)
                 * -std::log(core::clamp(T(1) - probability, core::epsilon<T>, T(1)));
}

template<typename T>
T weighted_categorical_cross_entropy(std::span<const T> logits, size_t target,
                                     std::span<const T> class_weights,
                                     T label_smoothing = T(0)) {
    if (logits.empty() || logits.size() != class_weights.size() || target >= logits.size())
        throw std::invalid_argument("NexusLoss: logits and class weights must have equal sizes");
    if (label_smoothing < T(0) || label_smoothing >= T(1))
        throw std::invalid_argument("NexusLoss: label smoothing must be in [0, 1)");
    const T maximum = *std::max_element(logits.begin(), logits.end());
    T exp_sum = T(0), loss = T(0);
    for (T logit : logits) exp_sum += std::exp(logit - maximum);
    const T log_z = maximum + std::log(exp_sum);
    for (size_t i = 0; i < logits.size(); ++i) {
        if (class_weights[i] < T(0))
            throw std::invalid_argument("NexusLoss: class weights must be non-negative");
        const T desired = (i == target ? T(1) - label_smoothing : T(0))
                          + label_smoothing / static_cast<T>(logits.size());
        loss -= class_weights[i] * desired * (logits[i] - log_z);
    }
    return loss;
}

template<typename T>
std::vector<T> effective_number_class_weights(std::span<const size_t> class_counts, T beta = T(0.999)) {
    if (class_counts.empty() || beta < T(0) || beta >= T(1))
        throw std::invalid_argument("NexusLoss: invalid class-balanced loss parameters");
    std::vector<T> weights(class_counts.size());
    T sum = T(0);
    for (size_t i = 0; i < class_counts.size(); ++i) {
        if (class_counts[i] == 0) throw std::invalid_argument("NexusLoss: class counts must be positive");
        weights[i] = (T(1) - beta) / (T(1) - std::pow(beta, static_cast<T>(class_counts[i])));
        sum += weights[i];
    }
    for (T& weight : weights) weight *= static_cast<T>(weights.size()) / sum;
    return weights;
}

template<typename T>
T normalized_binary_loss(T probability, T target) {
    const T loss = binary_cross_entropy(probability, target);
    const T normalizer = std::max(binary_cross_entropy(T(1) - probability, target), core::epsilon<T>);
    return loss / normalizer;
}

/**
 * @brief Softmax cross-entropy'nin logitlere gore turevi: p - y_smooth.
 *
 * Forward ile ayni hedef normalizasyonunu kullanir (hedef kutlesi 1'e cekilir,
 * sonra label smoothing uygulanir). Bu yuzden gradyan, kaybin sayisal tureviyle
 * birebir oturur. log-sum-exp sayesinde buyuk logit'lerde p sonlu kalir.
 */
template<typename T>
std::vector<T> categorical_cross_entropy_gradient(std::span<const T> logits, std::span<const T> target,
                                                   T label_smoothing = T(0)) {
    if (logits.empty() || logits.size() != target.size())
        throw std::invalid_argument("NexusLoss: categorical cross entropy requires equal non-empty vectors");
    if (label_smoothing < T(0) || label_smoothing >= T(1))
        throw std::invalid_argument("NexusLoss: label smoothing must be in [0, 1)");
    const T log_normalizer = core::log_sum_exp<T>(logits);
    T target_sum = T(0);
    for (T value : target) {
        if (value < T(0)) throw std::invalid_argument("NexusLoss: categorical targets must be non-negative");
        target_sum += value;
    }
    if (!(target_sum > T(0))) throw std::invalid_argument("NexusLoss: categorical target must have positive mass");
    std::vector<T> grad(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        const T normalized_target = target[i] / target_sum;
        const T smooth_target = (T(1) - label_smoothing) * normalized_target
                                + label_smoothing / static_cast<T>(logits.size());
        grad[i] = std::exp(logits[i] - log_normalizer) - smooth_target;
    }
    return grad;
}

/**
 * @brief KL(target || prediction) = sum target * (log target - log prediction).
 *
 * Soft-label senaryosu: prediction OLASILIKTIR, logit degil. Sifir olasilik
 * safe_log ile kesilir. Hedef 0 ise o terim 0'dir (0*log 0 = 0 sozlesmesi).
 */
template<typename T>
T kl_divergence(std::span<const T> prediction, std::span<const T> target) {
    if (prediction.empty() || prediction.size() != target.size())
        throw std::invalid_argument("NexusLoss: KL inputs must have equal non-zero sizes");
    T loss = T(0);
    for (size_t i = 0; i < prediction.size(); ++i) {
        if (target[i] < T(0) || prediction[i] < T(0))
            throw std::invalid_argument("NexusLoss: KL inputs must be non-negative");
        if (target[i] == T(0)) continue;
        loss += target[i] * (core::safe_log(target[i]) - core::safe_log(prediction[i]));
    }
    return loss;
}

/// @brief d/d(prediction_i) = -target_i / prediction_i. Kelepce disinda clamp turevi 0.
template<typename T>
std::vector<T> kl_divergence_gradient(std::span<const T> prediction, std::span<const T> target) {
    if (prediction.empty() || prediction.size() != target.size())
        throw std::invalid_argument("NexusLoss: KL inputs must have equal non-zero sizes");
    std::vector<T> grad(prediction.size());
    for (size_t i = 0; i < prediction.size(); ++i) {
        if (target[i] < T(0) || prediction[i] < T(0))
            throw std::invalid_argument("NexusLoss: KL inputs must be non-negative");
        if (prediction[i] <= core::epsilon<T>) grad[i] = T(0);
        else grad[i] = -target[i] / prediction[i];
    }
    return grad;
}

/**
 * @class CrossEntropyLoss
 * @brief Softmax ile negatif log-olabilirligi tek adimda hesaplar.
 *
 *   L(z, c) = log(sum_k exp(z_k)) - z_c
 *
 * Log-sum-exp hilesi: logit'lerden maksimum cikarilir, boylece exp(1000)
 * yerine exp(0) hesaplanir. Analitik turev dL/dz = softmax(z) - one_hot(c).
 *
 * @warning prediction HAM LOGIT'tir. Softmax uygulanmis olasilik verilmez.
 * @see Bishop, Pattern Recognition and Machine Learning, 2006, bol. 4.3.4
 */
template<typename T>
class CrossEntropyLoss {
public:
    explicit CrossEntropyLoss(T label_smoothing = T(0)) : smoothing_(label_smoothing) {
        if (label_smoothing < T(0) || label_smoothing >= T(1))
            throw std::invalid_argument("NexusLoss: label smoothing must be in [0, 1)");
    }

    T forward(std::span<const T> logits, size_t target_class) {
        cached_logits_.assign(logits.begin(), logits.end());
        cached_class_ = target_class;
        ready_ = true;
        return sparse_categorical_cross_entropy<T>(logits, target_class, smoothing_);
    }

    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        std::vector<T> one_hot(cached_logits_.size(), T(0));
        one_hot[cached_class_] = T(1);
        return categorical_cross_entropy_gradient<T>(cached_logits_, one_hot, smoothing_);
    }

private:
    T smoothing_;
    std::vector<T> cached_logits_;
    size_t cached_class_ = 0;
    bool ready_ = false;
};

/**
 * @class LabelSmoothingCrossEntropyLoss
 * @brief Hedefi (1-eps)*one_hot + eps/C dagilimiyla yumusatilmis cross-entropy.
 *
 * Szegedy et al., Rethinking the Inception Architecture, 2016.
 * Modelin tek sinifa olasilik 1 yapmasini engeller; asiri guven cezalandirilir.
 */
template<typename T>
class LabelSmoothingCrossEntropyLoss {
public:
    explicit LabelSmoothingCrossEntropyLoss(T smoothing = T(0.1)) : inner_(smoothing) {}
    T forward(std::span<const T> logits, size_t target_class) { return inner_.forward(logits, target_class); }
    std::vector<T> backward() const { return inner_.backward(); }
private:
    CrossEntropyLoss<T> inner_;
};

/**
 * @class KLDivLoss
 * @brief Soft hedef ile tahmin olasiliklari arasindaki KL uzakligi.
 *
 * L = sum_i y_i (log y_i - log p_i)
 * dL/dp_i = -y_i / p_i
 *
 * @warning p olasiliktir. Logit icin once softmax veya distillation::kl_divergence kullan.
 */
template<typename T>
class KLDivLoss {
public:
    T forward(std::span<const T> prediction, std::span<const T> target) {
        cached_prediction_.assign(prediction.begin(), prediction.end());
        cached_target_.assign(target.begin(), target.end());
        ready_ = true;
        return kl_divergence<T>(prediction, target);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return kl_divergence_gradient<T>(cached_prediction_, cached_target_);
    }
private:
    std::vector<T> cached_prediction_;
    std::vector<T> cached_target_;
    bool ready_ = false;
};

} // namespace classification

template<typename T>
class BCELoss : public LossBase<T> {
public:
    using LossBase<T>::LossBase;
    std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
        std::vector<T> result(pred.size());
        for (size_t i = 0; i < pred.size(); ++i)
            result[i] = classification::binary_cross_entropy(pred[i], target[i]);
        return result;
    }

    // L = -[y log p + (1-y) log(1-p)]. Ic bolgede dL/dp = (p - y) / (p(1-p)).
    std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
        std::vector<T> grad(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) {
            if (target[i] < T(0) || target[i] > T(1))
                throw std::invalid_argument("NexusLoss: BCE targets must be in [0, 1]");
            if (pred[i] <= core::epsilon<T> || pred[i] >= T(1) - core::epsilon<T>) {
                grad[i] = T(0);
                continue;
            }
            const T p = pred[i];
            grad[i] = (p - target[i]) / (p * (T(1) - p));
        }
        return grad;
    }
};

template<typename T>
class BCEWithLogitsLoss : public LossBase<T> {
public:
    using LossBase<T>::LossBase;
    std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
        std::vector<T> result(pred.size());
        for (size_t i = 0; i < pred.size(); ++i)
            result[i] = classification::binary_cross_entropy_with_logits(pred[i], target[i]);
        return result;
    }

    // Kararli formun turevi, sigmoid(x) - y. Forward'daki max/log1p adimlari saklanmaz.
    std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
        std::vector<T> grad(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) {
            if (target[i] < T(0) || target[i] > T(1))
                throw std::invalid_argument("NexusLoss: BCE targets must be in [0, 1]");
            grad[i] = core::stable_sigmoid(pred[i]) - target[i];
        }
        return grad;
    }
};

template<typename T>
class HingeLoss : public LossBase<T> {
public:
    using LossBase<T>::LossBase;
    std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
        std::vector<T> result(pred.size());
        for (size_t i = 0; i < pred.size(); ++i)
            result[i] = std::max(T(0), T(1) - pred[i] * target[i]);
        return result;
    }

    // Margin'in icinde turev -y, disinda 0. Esikte 0 subgradyani kullanilir.
    std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
        std::vector<T> grad(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) {
            grad[i] = pred[i] * target[i] < T(1) ? -target[i] : T(0);
        }
        return grad;
    }
};

template<typename T>
class SquaredHingeLoss : public LossBase<T> {
public:
    using LossBase<T>::LossBase;
    std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
        std::vector<T> result(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) {
            const T margin = std::max(T(0), T(1) - pred[i] * target[i]);
            result[i] = margin * margin;
        }
        return result;
    }

    std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
        std::vector<T> grad(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) {
            const T margin = T(1) - pred[i] * target[i];
            grad[i] = margin > T(0) ? T(-2) * margin * target[i] : T(0);
        }
        return grad;
    }
};

template<typename T>
class ExponentialLoss : public LossBase<T> {
public:
    using LossBase<T>::LossBase;
    std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
        std::vector<T> result(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) result[i] = std::exp(-target[i] * pred[i]);
        return result;
    }

    std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
        std::vector<T> grad(pred.size());
        for (size_t i = 0; i < pred.size(); ++i)
            grad[i] = -target[i] * std::exp(-target[i] * pred[i]);
        return grad;
    }
};

template<typename T>
class PerceptronLoss : public LossBase<T> {
public:
    using LossBase<T>::LossBase;
    std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
        std::vector<T> result(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) result[i] = std::max(T(0), -target[i] * pred[i]);
        return result;
    }

    std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
        std::vector<T> grad(pred.size());
        for (size_t i = 0; i < pred.size(); ++i)
            grad[i] = target[i] * pred[i] < T(0) ? -target[i] : T(0);
        return grad;
    }
};

template<typename T>
class ZeroOneLoss : public LossBase<T> {
public:
    using LossBase<T>::LossBase;
    std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
        std::vector<T> result(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) result[i] = pred[i] == target[i] ? T(0) : T(1);
        return result;
    }

    // 0-1 kaybi hemen hemen her yerde turevi sifirdir. Egiten adim bunu kullanmamali.
    std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
        (void)target;
        return std::vector<T>(pred.size(), T(0));
    }
};

template<typename T>
class FocalLoss : public LossBase<T> {
    T alpha_, gamma_;
public:
    explicit FocalLoss(T alpha = T(0.25), T gamma = T(2),
                       core::ReductionType reduction = core::ReductionType::Mean)
        : LossBase<T>(reduction), alpha_(alpha), gamma_(gamma) {
        if (alpha < T(0) || alpha > T(1) || gamma < T(0))
            throw std::invalid_argument("NexusLoss: invalid focal loss parameters");
    }
    std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
        std::vector<T> result(pred.size());
        for (size_t i = 0; i < pred.size(); ++i)
            result[i] = classification::focal_loss(pred[i], target[i], alpha_, gamma_);
        return result;
    }

    /**
     * Focal loss olasilik uzerinden turetilir.
     *   L = -a_t (1-p_t)^g log(p_t)
     *   dL/dp_t = a_t * g * (1-p_t)^{g-1} * log(p_t) - a_t * (1-p_t)^g / p_t
     *   d p_t / d p = +1 hedef >= 0.5 ise, aksi halde -1
     */
    std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
        std::vector<T> grad(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) {
            if (target[i] < T(0) || target[i] > T(1))
                throw std::invalid_argument("NexusLoss: focal targets must be in [0, 1]");
            if (pred[i] <= core::epsilon<T> || pred[i] >= T(1) - core::epsilon<T>) {
                grad[i] = T(0);
                continue;
            }
            const T p = pred[i];
            const bool positive = target[i] >= T(0.5);
            const T pt = positive ? p : T(1) - p;
            const T alpha_t = positive ? alpha_ : T(1) - alpha_;
            const T one_minus = T(1) - pt;
            T d_pt;
            if (gamma_ == T(0)) {
                d_pt = -alpha_t / pt;
            } else {
                d_pt = alpha_t * gamma_ * std::pow(one_minus, gamma_ - T(1)) * std::log(pt)
                       - alpha_t * std::pow(one_minus, gamma_) / pt;
            }
            grad[i] = d_pt * (positive ? T(1) : T(-1));
        }
        return grad;
    }
};

template<typename T>
using SmoothL1Loss = HuberLoss<T>;

} // namespace nexusloss

#endif
