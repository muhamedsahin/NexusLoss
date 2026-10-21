/**
 * @file metric.hpp
 * @brief Embedding uzayinda mesafe ve acisal marjin kayiplari.
 *
 * Triplet ve contrastive, oklid mesafesiyle calisir. ArcFace ve CosFace
 * normalize edilmis kosinus logit bekler (ham ozellik degil). Gradyan,
 * embedding'lerin veya kosinuslarin birlestirilmis vektoru olarak doner.
 *
 * @see Deng et al., ArcFace, 2019
 * @see Wang et al., CosFace, 2018
 */
#ifndef NEXUSLOSS_METRIC_HPP
#define NEXUSLOSS_METRIC_HPP

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>
#include "../core/utils.hpp"

namespace nexusloss::metric {

template<typename T>
T dot(std::span<const T> a, std::span<const T> b) {
    if (a.empty() || a.size() != b.size())
        throw std::invalid_argument("NexusLoss: embeddings must have equal non-zero dimensions");
    T value = T(0);
    for (size_t i = 0; i < a.size(); ++i) value += a[i] * b[i];
    return value;
}

template<typename T>
T cosine_similarity(std::span<const T> a, std::span<const T> b) {
    T aa = dot(a, a), bb = dot(b, b);
    return dot(a, b) / std::max(std::sqrt(aa * bb), core::epsilon<T>);
}

template<typename T>
T euclidean_distance(std::span<const T> a, std::span<const T> b) {
    if (a.empty() || a.size() != b.size())
        throw std::invalid_argument("NexusLoss: embeddings must have equal non-zero dimensions");
    T sum = T(0);
    for (size_t i = 0; i < a.size(); ++i) { const T d = a[i] - b[i]; sum += d * d; }
    return std::sqrt(sum);
}

template<typename T>
T contrastive_loss(std::span<const T> a, std::span<const T> b, bool same_class, T margin = T(1)) {
    if (margin < T(0)) throw std::invalid_argument("NexusLoss: contrastive margin must be non-negative");
    const T distance = euclidean_distance(a, b);
    const T negative = std::max(T(0), margin - distance);
    return same_class ? distance * distance : negative * negative;
}

template<typename T>
T cosine_embedding_loss(std::span<const T> a, std::span<const T> b, bool same_class, T margin = T(0)) {
    const T similarity = cosine_similarity(a, b);
    return same_class ? T(1) - similarity : std::max(T(0), similarity - margin);
}

template<typename T>
T triplet_loss(std::span<const T> anchor, std::span<const T> positive,
               std::span<const T> negative, T margin = T(1)) {
    if (margin < T(0)) throw std::invalid_argument("NexusLoss: triplet margin must be non-negative");
    return std::max(T(0), euclidean_distance(anchor, positive)
                           - euclidean_distance(anchor, negative) + margin);
}

template<typename T>
T margin_ranking_loss(T x1, T x2, T target, T margin = T(0)) {
    return std::max(T(0), -target * (x1 - x2) + margin);
}

template<typename T>
T center_loss(std::span<const T> embedding, std::span<const T> class_center) {
    const T distance = euclidean_distance(embedding, class_center);
    return T(0.5) * distance * distance;
}

template<typename T>
T n_pair_loss(std::span<const T> anchor, std::span<const T> positive,
              std::span<const T> negative_embeddings, size_t negative_count) {
    if (negative_count == 0 || negative_embeddings.size() != anchor.size() * negative_count)
        throw std::invalid_argument("NexusLoss: N-pair negatives must be flattened row-major embeddings");
    T denominator = T(1);
    for (size_t n = 0; n < negative_count; ++n) {
        auto negative = negative_embeddings.subspan(n * anchor.size(), anchor.size());
        denominator += std::exp(dot(anchor, negative) - dot(anchor, positive));
    }
    return std::log(denominator);
}

template<typename T>
T angular_margin_cross_entropy(std::span<const T> cosine_logits, size_t target,
                               T margin, T scale, int kind = 0) {
    if (cosine_logits.empty() || target >= cosine_logits.size() || margin < T(0) || !(scale > T(0)))
        throw std::invalid_argument("NexusLoss: invalid angular-margin loss parameters");
    std::vector<T> logits(cosine_logits.begin(), cosine_logits.end());
    const T cosine = core::clamp(logits[target], T(-1), T(1));
    if (kind == 0) { // ArcFace
        logits[target] = std::cos(std::acos(cosine) + margin);
    } else if (kind == 1) { // CosFace
        logits[target] = cosine - margin;
    } else { // SphereFace (A-Softmax)
        logits[target] = std::cos(std::acos(cosine) * (T(1) + margin));
    }
    for (T& logit : logits) logit *= scale;
    const T maximum = *std::max_element(logits.begin(), logits.end());
    T sum = T(0);
    for (T logit : logits) sum += std::exp(logit - maximum);
    return maximum + std::log(sum) - logits[target];
}

template<typename T>
T arcface_loss(std::span<const T> cosine_logits, size_t target, T margin = T(0.5), T scale = T(64)) {
    return angular_margin_cross_entropy(cosine_logits, target, margin, scale, 0);
}
template<typename T>
T cosface_loss(std::span<const T> cosine_logits, size_t target, T margin = T(0.35), T scale = T(64)) {
    return angular_margin_cross_entropy(cosine_logits, target, margin, scale, 1);
}
template<typename T>
T sphereface_loss(std::span<const T> cosine_logits, size_t target, T margin = T(1), T scale = T(64)) {
    return angular_margin_cross_entropy(cosine_logits, target, margin, scale, 2);
}

/// @brief Kontrastif kaybin iki embedding'e gore turevi. grad_out = [a | b], ayirmasiz.
template<typename T>
void contrastive_loss_gradient_into(std::span<const T> a, std::span<const T> b, bool same_class,
                                    std::span<T> grad_out, T margin = T(1)) {
    if (grad_out.size() != a.size() * 2)
        throw std::invalid_argument("NexusLoss: contrastive grad_out must hold [a | b]");
    const T distance = std::max(euclidean_distance(a, b), core::epsilon<T>);
    T scale = T(0);
    if (same_class) scale = T(2) * distance;
    else if (distance < margin) scale = T(-2) * (margin - distance);
    const T factor = scale / distance;
    for (size_t i = 0; i < a.size(); ++i) {
        const T value = factor * (a[i] - b[i]);
        grad_out[i] = value;
        grad_out[a.size() + i] = -value;
    }
}

template<typename T>
std::vector<T> contrastive_loss_gradient(std::span<const T> a, std::span<const T> b, bool same_class,
                                         T margin = T(1)) {
    std::vector<T> grad(a.size() * 2);
    contrastive_loss_gradient_into<T>(a, b, same_class, grad, margin);
    return grad;
}

/// @brief Triplet kaybinin [anchor | positive | negative] turevi. Margin disinda sifir.
template<typename T>
void triplet_loss_gradient_into(std::span<const T> anchor, std::span<const T> positive,
                                std::span<const T> negative, std::span<T> grad_out, T margin = T(1)) {
    if (anchor.size() != positive.size() || anchor.size() != negative.size())
        throw std::invalid_argument("NexusLoss: triplet embeddings must share a dimension");
    const size_t dim = anchor.size();
    if (grad_out.size() != dim * 3)
        throw std::invalid_argument("NexusLoss: triplet grad_out must hold [anchor | positive | negative]");
    const T positive_distance = euclidean_distance(anchor, positive);
    const T negative_distance = euclidean_distance(anchor, negative);
    if (positive_distance - negative_distance + margin <= T(0)) {
        std::fill(grad_out.begin(), grad_out.end(), T(0));
        return;
    }
    const T inverse_positive = T(1) / std::max(positive_distance, core::epsilon<T>);
    const T inverse_negative = T(1) / std::max(negative_distance, core::epsilon<T>);
    for (size_t i = 0; i < dim; ++i) {
        const T to_positive = (anchor[i] - positive[i]) * inverse_positive;
        const T to_negative = (anchor[i] - negative[i]) * inverse_negative;
        grad_out[i] = to_positive - to_negative;
        grad_out[dim + i] = -to_positive;
        grad_out[dim * 2 + i] = to_negative;
    }
}

template<typename T>
std::vector<T> triplet_loss_gradient(std::span<const T> anchor, std::span<const T> positive,
                                     std::span<const T> negative, T margin = T(1)) {
    std::vector<T> grad(anchor.size() * 3);
    triplet_loss_gradient_into<T>(anchor, positive, negative, grad, margin);
    return grad;
}

/**
 * @brief ArcFace/CosFace/SphereFace logit turevi.
 * Hedef kosinüs, margin donusumunden gecer, sonra olceklenir; CE gradyani zincir kuraliyle gelir.
 */
/**
 * @brief Acisal marjin turevini `grad_out` icine yazar; bellek ayirmaz.
 *
 * Yalnizca hedef sinifin logit'i donusur; diger siniflarin d(logit)/d(cos)
 * olcegi `scale`'dir. Bu yuzden ara vektor gerekmez: once olceklenmis
 * logitler `grad_out` icine yazilir, log-sum-exp alinir, sonra yerinde turev.
 */
template<typename T>
void angular_margin_gradient_into(std::span<const T> cosine_logits, size_t target,
                                  T margin, T scale, int kind, std::span<T> grad_out) {
    if (cosine_logits.empty() || target >= cosine_logits.size() || margin < T(0) || !(scale > T(0)))
        throw std::invalid_argument("NexusLoss: invalid angular-margin loss parameters");
    if (grad_out.size() != cosine_logits.size())
        throw std::invalid_argument("NexusLoss: angular-margin grad_out must match cosine logits");
    const T cosine = core::clamp(cosine_logits[target], T(-1), T(1));
    const T sine = std::sqrt(std::max(T(0), T(1) - cosine * cosine));
    T target_logit = T(0), d_target = T(1);
    if (kind == 0) {
        const T theta = std::acos(cosine);
        target_logit = std::cos(theta + margin);
        d_target = sine < core::epsilon<T> ? T(0) : std::sin(theta + margin) / sine;
    } else if (kind == 1) {
        target_logit = cosine - margin;
    } else {
        const T theta = std::acos(cosine);
        const T multiplier = T(1) + margin;
        target_logit = std::cos(theta * multiplier);
        d_target = sine < core::epsilon<T> ? T(0) : multiplier * std::sin(theta * multiplier) / sine;
    }
    for (size_t i = 0; i < cosine_logits.size(); ++i)
        grad_out[i] = (i == target ? target_logit : cosine_logits[i]) * scale;
    const T log_z = core::log_sum_exp<T>(std::span<const T>(grad_out.data(), grad_out.size()));
    for (size_t i = 0; i < grad_out.size(); ++i) {
        const T probability = std::exp(grad_out[i] - log_z);
        const T d_logit = probability - (i == target ? T(1) : T(0));
        grad_out[i] = d_logit * scale * (i == target ? d_target : T(1));
    }
}

template<typename T>
std::vector<T> angular_margin_gradient(std::span<const T> cosine_logits, size_t target,
                                       T margin, T scale, int kind) {
    std::vector<T> grad(cosine_logits.size());
    angular_margin_gradient_into<T>(cosine_logits, target, margin, scale, kind, grad);
    return grad;
}

template<typename T>
class ContrastiveLoss {
public:
    explicit ContrastiveLoss(T margin = T(1)) : margin_(margin) {}
    T forward(std::span<const T> a, std::span<const T> b, bool same_class) {
        a_.assign(a.begin(), a.end());
        b_.assign(b.begin(), b.end());
        same_ = same_class;
        ready_ = true;
        return contrastive_loss(a, b, same_class, margin_);
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return a_.size() * 2; }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        contrastive_loss_gradient_into<T>(a_, b_, same_, grad_out, margin_);
    }
private:
    T margin_;
    std::vector<T> a_, b_;
    bool same_ = true, ready_ = false;
};

template<typename T>
class TripletLoss {
public:
    explicit TripletLoss(T margin = T(1)) : margin_(margin) {}
    T forward(std::span<const T> anchor, std::span<const T> positive, std::span<const T> negative) {
        anchor_.assign(anchor.begin(), anchor.end());
        positive_.assign(positive.begin(), positive.end());
        negative_.assign(negative.begin(), negative.end());
        ready_ = true;
        return triplet_loss(anchor, positive, negative, margin_);
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return anchor_.size() * 3; }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        triplet_loss_gradient_into<T>(anchor_, positive_, negative_, grad_out, margin_);
    }
private:
    T margin_;
    std::vector<T> anchor_, positive_, negative_;
    bool ready_ = false;
};

template<typename T>
class ArcFaceLoss {
public:
    ArcFaceLoss(T margin = T(0.5), T scale = T(64)) : margin_(margin), scale_(scale) {}
    T forward(std::span<const T> cosine_logits, size_t target) {
        cosine_.assign(cosine_logits.begin(), cosine_logits.end());
        target_ = target;
        ready_ = true;
        return arcface_loss(cosine_logits, target, margin_, scale_);
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return cosine_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        angular_margin_gradient_into<T>(cosine_, target_, margin_, scale_, 0, grad_out);
    }
private:
    T margin_, scale_;
    std::vector<T> cosine_;
    size_t target_ = 0;
    bool ready_ = false;
};

template<typename T>
class CosFaceLoss {
public:
    CosFaceLoss(T margin = T(0.35), T scale = T(64)) : margin_(margin), scale_(scale) {}
    T forward(std::span<const T> cosine_logits, size_t target) {
        cosine_.assign(cosine_logits.begin(), cosine_logits.end());
        target_ = target;
        ready_ = true;
        return cosface_loss(cosine_logits, target, margin_, scale_);
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return cosine_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        angular_margin_gradient_into<T>(cosine_, target_, margin_, scale_, 1, grad_out);
    }
private:
    T margin_, scale_;
    std::vector<T> cosine_;
    size_t target_ = 0;
    bool ready_ = false;
};

/// @brief SphereFace (A-Softmax): hedef aci (1 + margin) ile carpilir.
template<typename T>
class SphereFaceLoss {
public:
    SphereFaceLoss(T margin = T(1), T scale = T(64)) : margin_(margin), scale_(scale) {}
    T forward(std::span<const T> cosine_logits, size_t target) {
        cosine_.assign(cosine_logits.begin(), cosine_logits.end());
        target_ = target;
        ready_ = true;
        return sphereface_loss(cosine_logits, target, margin_, scale_);
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return cosine_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        angular_margin_gradient_into<T>(cosine_, target_, margin_, scale_, 2, grad_out);
    }
private:
    T margin_, scale_;
    std::vector<T> cosine_;
    size_t target_ = 0;
    bool ready_ = false;
};

} // namespace nexusloss::metric

#endif
