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

/// @brief Kontrastif kaybin iki embedding'e gore turevi. Donus [a | b] birlestirilmis.
template<typename T>
std::vector<T> contrastive_loss_gradient(std::span<const T> a, std::span<const T> b, bool same_class,
                                         T margin = T(1)) {
    const T distance = std::max(euclidean_distance(a, b), core::epsilon<T>);
    std::vector<T> grad(a.size() * 2, T(0));
    auto write = [&](T scale) {
        for (size_t i = 0; i < a.size(); ++i) {
            const T direction = (a[i] - b[i]) / distance;
            grad[i] = scale * direction;
            grad[a.size() + i] = -scale * direction;
        }
    };
    if (same_class) write(T(2) * distance);
    else if (distance < margin) write(T(-2) * (margin - distance));
    return grad;
}

/// @brief Triplet kaybinin [anchor | positive | negative] turevi. Margin disinda sifir.
template<typename T>
std::vector<T> triplet_loss_gradient(std::span<const T> anchor, std::span<const T> positive,
                                     std::span<const T> negative, T margin = T(1)) {
    if (anchor.size() != positive.size() || anchor.size() != negative.size())
        throw std::invalid_argument("NexusLoss: triplet embeddings must share a dimension");
    const size_t dim = anchor.size();
    std::vector<T> grad(dim * 3, T(0));
    const T positive_distance = euclidean_distance(anchor, positive);
    const T negative_distance = euclidean_distance(anchor, negative);
    if (positive_distance - negative_distance + margin <= T(0)) return grad;
    const T safe_positive = std::max(positive_distance, core::epsilon<T>);
    const T safe_negative = std::max(negative_distance, core::epsilon<T>);
    for (size_t i = 0; i < dim; ++i) {
        const T to_positive = (anchor[i] - positive[i]) / safe_positive;
        const T to_negative = (anchor[i] - negative[i]) / safe_negative;
        grad[i] = to_positive - to_negative;
        grad[dim + i] = -to_positive;
        grad[dim * 2 + i] = to_negative;
    }
    return grad;
}

/**
 * @brief ArcFace/CosFace/SphereFace logit turevi.
 * Hedef kosinüs, margin donusumunden gecer, sonra olceklenir; CE gradyani zincir kuraliyle gelir.
 */
template<typename T>
std::vector<T> angular_margin_gradient(std::span<const T> cosine_logits, size_t target,
                                       T margin, T scale, int kind) {
    if (cosine_logits.empty() || target >= cosine_logits.size() || margin < T(0) || !(scale > T(0)))
        throw std::invalid_argument("NexusLoss: invalid angular-margin loss parameters");
    std::vector<T> logits(cosine_logits.size());
    std::vector<T> d_modified_d_cosine(cosine_logits.size(), T(1));
    for (size_t i = 0; i < cosine_logits.size(); ++i) {
        if (i != target) {
            logits[i] = cosine_logits[i] * scale;
            continue;
        }
        const T cosine = core::clamp(cosine_logits[i], T(-1), T(1));
        if (kind == 0) {
            const T theta = std::acos(cosine);
            const T sine = std::sqrt(std::max(T(0), T(1) - cosine * cosine));
            logits[i] = std::cos(theta + margin) * scale;
            d_modified_d_cosine[i] = sine < core::epsilon<T> ? T(0) : std::sin(theta + margin) / sine;
        } else if (kind == 1) {
            logits[i] = (cosine - margin) * scale;
        } else {
            const T theta = std::acos(cosine);
            const T multiplier = T(1) + margin;
            const T sine = std::sqrt(std::max(T(0), T(1) - cosine * cosine));
            logits[i] = std::cos(theta * multiplier) * scale;
            d_modified_d_cosine[i] = sine < core::epsilon<T> ? T(0)
                : multiplier * std::sin(theta * multiplier) / sine;
        }
    }
    const T log_z = core::log_sum_exp<T>(logits);
    std::vector<T> grad(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        const T probability = std::exp(logits[i] - log_z);
        const T d_logit = probability - (i == target ? T(1) : T(0));
        grad[i] = d_logit * scale * d_modified_d_cosine[i];
    }
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
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return contrastive_loss_gradient<T>(a_, b_, same_, margin_);
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
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return triplet_loss_gradient<T>(anchor_, positive_, negative_, margin_);
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
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return angular_margin_gradient<T>(cosine_, target_, margin_, scale_, 0);
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
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return angular_margin_gradient<T>(cosine_, target_, margin_, scale_, 1);
    }
private:
    T margin_, scale_;
    std::vector<T> cosine_;
    size_t target_ = 0;
    bool ready_ = false;
};

} // namespace nexusloss::metric

#endif
