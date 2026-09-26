//
// Created by muham on 24.09.2026.
//

/**
 * @file segmentation.hpp
 * @brief Piksel örtüşmesine dayanan segmentasyon kayıpları.
 *
 * Tahmin ve hedef aynı uzunlukta yumuşak maskelerdir (olasılık, logit değil),
 * Lovász-Softmax hariç: o piksel-major olasılık ve sınıf indeksi ister.
 * Paydada smooth sabiti vardır; tamamen boş maskede sonuç sonlu kalır.
 *
 * @see core/utils.hpp  epsilon ve güvenli bölme
 */
#ifndef NEXUSLOSS_SEGMENTATION_HPP
#define NEXUSLOSS_SEGMENTATION_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../core/utils.hpp"

namespace nexusloss::segmentation {

template<typename T>
void validate(std::span<const T> prediction, std::span<const T> target) {
    if (prediction.empty() || prediction.size() != target.size())
        throw std::invalid_argument("NexusLoss: segmentation inputs must have equal non-zero sizes");
}

template<typename T>
T dice_loss(std::span<const T> prediction, std::span<const T> target, T smooth = T(1)) {
    validate(prediction, target);
    T intersection = T(0), denominator = T(0);
    for (size_t i = 0; i < prediction.size(); ++i) {
        intersection += prediction[i] * target[i];
        denominator += prediction[i] + target[i];
    }
    return T(1) - (T(2) * intersection + smooth) / (denominator + smooth);
}

template<typename T>
T iou_loss(std::span<const T> prediction, std::span<const T> target, T smooth = T(1)) {
    validate(prediction, target);
    T intersection = T(0), total = T(0);
    for (size_t i = 0; i < prediction.size(); ++i) {
        intersection += prediction[i] * target[i];
        total += prediction[i] + target[i] - prediction[i] * target[i];
    }
    return T(1) - (intersection + smooth) / (total + smooth);
}

template<typename T>
T tversky_loss(std::span<const T> prediction, std::span<const T> target,
               T alpha = T(0.5), T beta = T(0.5), T smooth = T(1)) {
    validate(prediction, target);
    if (alpha < T(0) || beta < T(0)) throw std::invalid_argument("NexusLoss: Tversky weights must be non-negative");
    T tp = T(0), fp = T(0), fn = T(0);
    for (size_t i = 0; i < prediction.size(); ++i) {
        tp += prediction[i] * target[i];
        fp += prediction[i] * (T(1) - target[i]);
        fn += (T(1) - prediction[i]) * target[i];
    }
    return T(1) - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth);
}

template<typename T>
T focal_tversky_loss(std::span<const T> prediction, std::span<const T> target,
                     T alpha = T(0.5), T beta = T(0.5), T gamma = T(4) / T(3)) {
    if (gamma <= T(0)) throw std::invalid_argument("NexusLoss: focal Tversky gamma must be positive");
    return std::pow(tversky_loss(prediction, target, alpha, beta), gamma);
}

template<typename T>
T binary_cross_entropy(std::span<const T> probability, std::span<const T> target) {
    validate(probability, target);
    T sum = T(0);
    for (size_t i = 0; i < probability.size(); ++i) {
        const T p = core::clamp(probability[i], core::epsilon<T>, T(1) - core::epsilon<T>);
        sum -= target[i] * std::log(p) + (T(1) - target[i]) * std::log1p(-p);
    }
    return sum / static_cast<T>(probability.size());
}

template<typename T>
T combo_loss(std::span<const T> probability, std::span<const T> target,
             T dice_weight = T(0.5), T smooth = T(1)) {
    if (dice_weight < T(0) || dice_weight > T(1))
        throw std::invalid_argument("NexusLoss: combo dice weight must be in [0, 1]");
    return dice_weight * dice_loss(probability, target, smooth)
           + (T(1) - dice_weight) * binary_cross_entropy(probability, target);
}

template<typename T>
T boundary_loss(std::span<const T> probability, std::span<const T> signed_distance) {
    validate(probability, signed_distance);
    T sum = T(0);
    for (size_t i = 0; i < probability.size(); ++i) sum += probability[i] * signed_distance[i];
    return sum / static_cast<T>(probability.size());
}

template<typename T>
T hausdorff_distance_loss(std::span<const T> probability, std::span<const T> target,
                          std::span<const T> distance_to_boundary) {
    validate(probability, target);
    validate(probability, distance_to_boundary);
    T sum = T(0);
    for (size_t i = 0; i < probability.size(); ++i) {
        const T error = std::abs(probability[i] - target[i]);
        sum += error * distance_to_boundary[i] * distance_to_boundary[i];
    }
    return sum / static_cast<T>(probability.size());
}

template<typename T>
T lovasz_hinge_loss(std::span<const T> logits, std::span<const T> target) {
    validate(logits, target);
    std::vector<std::pair<T, T>> errors;
    errors.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i)
        errors.emplace_back(T(1) - logits[i] * (target[i] > T(0) ? T(1) : T(-1)), target[i]);
    std::sort(errors.begin(), errors.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    T total_positive = T(0);
    for (const auto& item : errors) total_positive += item.second > T(0) ? T(1) : T(0);
    T intersection = total_positive, union_size = total_positive, previous = T(0), loss = T(0);
    for (size_t i = 0; i < errors.size(); ++i) {
        const T positive = errors[i].second > T(0) ? T(1) : T(0);
        intersection -= positive;
        union_size += T(1) - positive;
        const T jaccard = T(1) - intersection / std::max(union_size, core::epsilon<T>);
        const T gradient = i + 1 == errors.size() ? jaccard : jaccard - previous;
        loss += std::max(errors[i].first, T(0)) * gradient;
        previous = jaccard;
    }
    return loss;
}

template<typename T>
T lovasz_softmax_loss(std::span<const T> probabilities, std::span<const size_t> target_classes,
                      size_t pixels, size_t classes) {
    if (!pixels || !classes || probabilities.size() != pixels * classes || target_classes.size() != pixels)
        throw std::invalid_argument("NexusLoss: Lovasz-Softmax expects pixel-major [pixel, class] probabilities");
    T total_loss = T(0);
    size_t present_classes = 0;
    for (size_t klass = 0; klass < classes; ++klass) {
        std::vector<std::pair<T, T>> errors;
        errors.reserve(pixels);
        T positives = T(0);
        for (size_t pixel = 0; pixel < pixels; ++pixel) {
            if (target_classes[pixel] >= classes)
                throw std::out_of_range("NexusLoss: segmentation target class is out of range");
            const T foreground = target_classes[pixel] == klass ? T(1) : T(0);
            positives += foreground;
            errors.emplace_back(std::abs(foreground - probabilities[pixel * classes + klass]), foreground);
        }
        if (positives == T(0)) continue;
        ++present_classes;
        std::sort(errors.begin(), errors.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        T intersection = positives, union_size = positives, previous = T(0);
        for (size_t i = 0; i < pixels; ++i) {
            intersection -= errors[i].second;
            union_size += T(1) - errors[i].second;
            const T jaccard = T(1) - intersection / std::max(union_size, core::epsilon<T>);
            const T gradient = i + 1 == pixels ? jaccard : jaccard - previous;
            total_loss += errors[i].first * gradient;
            previous = jaccard;
        }
    }
    return present_classes ? total_loss / static_cast<T>(present_classes) : T(0);
}

/// @brief Dice loss skalerinin her piksele gore turevi.
/// dL/dp_i = - (2 t_i (D+s) - (2I+s)) / (D+s)^2
template<typename T>
std::vector<T> dice_loss_gradient(std::span<const T> prediction, std::span<const T> target, T smooth = T(1)) {
    validate(prediction, target);
    T intersection = T(0), denominator = T(0);
    for (size_t i = 0; i < prediction.size(); ++i) {
        intersection += prediction[i] * target[i];
        denominator += prediction[i] + target[i];
    }
    const T num = T(2) * intersection + smooth;
    const T den = denominator + smooth;
    std::vector<T> grad(prediction.size());
    const T den2 = den * den;
    for (size_t i = 0; i < prediction.size(); ++i)
        grad[i] = -(T(2) * target[i] * den - num) / den2;
    return grad;
}

/// @brief Soft IoU / Jaccard loss turevi.
template<typename T>
std::vector<T> iou_loss_gradient(std::span<const T> prediction, std::span<const T> target, T smooth = T(1)) {
    validate(prediction, target);
    T intersection = T(0), total = T(0);
    for (size_t i = 0; i < prediction.size(); ++i) {
        intersection += prediction[i] * target[i];
        total += prediction[i] + target[i] - prediction[i] * target[i];
    }
    const T num = intersection + smooth;
    const T den = total + smooth;
    const T den2 = den * den;
    std::vector<T> grad(prediction.size());
    for (size_t i = 0; i < prediction.size(); ++i) {
        const T d_num = target[i];
        const T d_den = T(1) - target[i];
        grad[i] = -(d_num * den - num * d_den) / den2;
    }
    return grad;
}

/// @brief Tversky skalerinin piksel turevi. FP agirligi alpha, FN agirligi beta.
template<typename T>
std::vector<T> tversky_loss_gradient(std::span<const T> prediction, std::span<const T> target,
                                     T alpha = T(0.5), T beta = T(0.5), T smooth = T(1)) {
    validate(prediction, target);
    if (alpha < T(0) || beta < T(0)) throw std::invalid_argument("NexusLoss: Tversky weights must be non-negative");
    T tp = T(0), fp = T(0), fn = T(0);
    for (size_t i = 0; i < prediction.size(); ++i) {
        tp += prediction[i] * target[i];
        fp += prediction[i] * (T(1) - target[i]);
        fn += (T(1) - prediction[i]) * target[i];
    }
    const T num = tp + smooth;
    const T den = tp + alpha * fp + beta * fn + smooth;
    const T den2 = den * den;
    std::vector<T> grad(prediction.size());
    for (size_t i = 0; i < prediction.size(); ++i) {
        const T d_tp = target[i];
        const T d_fp = T(1) - target[i];
        const T d_fn = -target[i];
        const T d_num = d_tp;
        const T d_den = d_tp + alpha * d_fp + beta * d_fn;
        grad[i] = -(d_num * den - num * d_den) / den2;
    }
    return grad;
}

template<typename T>
std::vector<T> focal_tversky_loss_gradient(std::span<const T> prediction, std::span<const T> target,
                                           T alpha = T(0.5), T beta = T(0.5), T gamma = T(4) / T(3)) {
    if (gamma <= T(0)) throw std::invalid_argument("NexusLoss: focal Tversky gamma must be positive");
    const T base = tversky_loss(prediction, target, alpha, beta);
    const T scale = gamma * std::pow(std::max(base, core::epsilon<T>), gamma - T(1));
    auto grad = tversky_loss_gradient(prediction, target, alpha, beta);
    for (T& value : grad) value *= scale;
    return grad;
}

/**
 * @class DiceLoss
 * @brief 1 - 2|X∩Y| / (|X|+|Y|). Sinif dengesizligine L2 piksel kaybindan daha dayanikli.
 *
 * Milletari et al., V-Net, 2016. Payda sifira dusmesin diye smooth (varsayilan 1) eklenir.
 * prediction ve target ayni sekilde, [0,1] yumusak maskelerdir.
 */
template<typename T>
class DiceLoss {
public:
    explicit DiceLoss(T smooth = T(1)) : smooth_(smooth) {}
    T forward(std::span<const T> prediction, std::span<const T> target) {
        cached_prediction_.assign(prediction.begin(), prediction.end());
        cached_target_.assign(target.begin(), target.end());
        ready_ = true;
        return dice_loss(prediction, target, smooth_);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return dice_loss_gradient<T>(cached_prediction_, cached_target_, smooth_);
    }
private:
    T smooth_;
    std::vector<T> cached_prediction_, cached_target_;
    bool ready_ = false;
};

/// @class IoULoss @brief 1 - |X∩Y| / |X∪Y|. Dice'in birlesim paydali kardesi (Jaccard).
template<typename T>
class IoULoss {
public:
    explicit IoULoss(T smooth = T(1)) : smooth_(smooth) {}
    T forward(std::span<const T> prediction, std::span<const T> target) {
        cached_prediction_.assign(prediction.begin(), prediction.end());
        cached_target_.assign(target.begin(), target.end());
        ready_ = true;
        return iou_loss(prediction, target, smooth_);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return iou_loss_gradient<T>(cached_prediction_, cached_target_, smooth_);
    }
private:
    T smooth_;
    std::vector<T> cached_prediction_, cached_target_;
    bool ready_ = false;
};

/**
 * @class TverskyLoss
 * @brief Dice'in FP/FN ayri agirlikli hali. alpha FP'yi, beta FN'yi cezalandirir.
 * @see Salehi et al., Tversky loss function for image segmentation, 2017
 */
template<typename T>
class TverskyLoss {
public:
    TverskyLoss(T alpha = T(0.3), T beta = T(0.7), T smooth = T(1))
        : alpha_(alpha), beta_(beta), smooth_(smooth) {}
    T forward(std::span<const T> prediction, std::span<const T> target) {
        cached_prediction_.assign(prediction.begin(), prediction.end());
        cached_target_.assign(target.begin(), target.end());
        ready_ = true;
        return tversky_loss(prediction, target, alpha_, beta_, smooth_);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return tversky_loss_gradient<T>(cached_prediction_, cached_target_, alpha_, beta_, smooth_);
    }
private:
    T alpha_, beta_, smooth_;
    std::vector<T> cached_prediction_, cached_target_;
    bool ready_ = false;
};

/// @class FocalTverskyLoss @brief Tversky'nin ustel hali. Zor orneklerin gradyanini buyutur.
template<typename T>
class FocalTverskyLoss {
public:
    FocalTverskyLoss(T alpha = T(0.3), T beta = T(0.7), T gamma = T(4) / T(3))
        : alpha_(alpha), beta_(beta), gamma_(gamma) {}
    T forward(std::span<const T> prediction, std::span<const T> target) {
        cached_prediction_.assign(prediction.begin(), prediction.end());
        cached_target_.assign(target.begin(), target.end());
        ready_ = true;
        return focal_tversky_loss(prediction, target, alpha_, beta_, gamma_);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return focal_tversky_loss_gradient<T>(cached_prediction_, cached_target_, alpha_, beta_, gamma_);
    }
private:
    T alpha_, beta_, gamma_;
    std::vector<T> cached_prediction_, cached_target_;
    bool ready_ = false;
};

/**
 * @class LovaszSoftmaxLoss
 * @brief Jaccard kaybinin Lovasz uzantisinin ortalama degeri.
 *
 * Berman, Triki, Blaschko, The Lovasz-Softmax loss, 2018.
 * Girdi piksel-major olasiliktir: uzunluk pixels * classes.
 * Turev, siralama kirilgan oldugu icin (mutlak hatanin permütasyonu) kullaniciya
 * forward degeri uzerinden central difference ile dogrulanir; analitik subgradyan
 * lovasz_softmax_gradient ile ayrica verilir.
 */
template<typename T>
class LovaszSoftmaxLoss {
public:
    T forward(std::span<const T> probabilities, std::span<const size_t> target_classes,
              size_t pixels, size_t classes) {
        cached_probabilities_.assign(probabilities.begin(), probabilities.end());
        cached_targets_.assign(target_classes.begin(), target_classes.end());
        pixels_ = pixels;
        classes_ = classes;
        ready_ = true;
        return lovasz_softmax_loss(probabilities, target_classes, pixels, classes);
    }
    std::vector<T> backward() const;
private:
    std::vector<T> cached_probabilities_;
    std::vector<size_t> cached_targets_;
    size_t pixels_ = 0, classes_ = 0;
    bool ready_ = false;
};

template<typename T>
std::vector<T> LovaszSoftmaxLoss<T>::backward() const {
    if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
    // Subgradyan: her sinif icin siralanmis Jaccard katsayilari, hatanin isaretiyle carpilir.
    // |fg - p| turevi, p > fg ise +1, p < fg ise -1'dir. Esitlikte 0.
    std::vector<T> grad(cached_probabilities_.size(), T(0));
    size_t present = 0;
    for (size_t klass = 0; klass < classes_; ++klass) {
        struct Item { T error; T foreground; size_t pixel; int sign; };
        std::vector<Item> items;
        items.reserve(pixels_);
        T positives = T(0);
        for (size_t pixel = 0; pixel < pixels_; ++pixel) {
            const T foreground = cached_targets_[pixel] == klass ? T(1) : T(0);
            positives += foreground;
            const T probability = cached_probabilities_[pixel * classes_ + klass];
            const T delta = probability - foreground;
            items.push_back({std::abs(delta), foreground, pixel, delta > T(0) ? 1 : (delta < T(0) ? -1 : 0)});
        }
        if (positives == T(0)) continue;
        ++present;
        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.error > b.error; });
        T intersection = positives, union_size = positives, previous = T(0);
        for (size_t i = 0; i < pixels_; ++i) {
            intersection -= items[i].foreground;
            union_size += T(1) - items[i].foreground;
            const T jaccard = T(1) - intersection / std::max(union_size, core::epsilon<T>);
            const T coefficient = i + 1 == pixels_ ? jaccard : jaccard - previous;
            grad[items[i].pixel * classes_ + klass] = coefficient * static_cast<T>(items[i].sign);
            previous = jaccard;
        }
    }
    if (present) for (T& value : grad) value /= static_cast<T>(present);
    return grad;
}

} // namespace nexusloss::segmentation

#endif //NEXUSLOSS_SEGMENTATION_HPP
