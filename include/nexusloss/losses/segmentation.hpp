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

/**
 * @brief Lovasz uzantisinin siralanmis hata uzerindeki agirliklari.
 *
 * `order` hatalari buyukten kucuge siralayan permutasyondur. Berman et al.
 * (2018, Alg. 1) geregi g_0 = J_0 ve g_i = J_i - J_{i-1}; agirliklarin toplami
 * J_{n-1}'dir ve hata sifirken kayip sifirdir. Agirlik `weights[i]` sirali
 * i. elemana aittir.
 */
template<typename T, typename ForegroundAt>
void lovasz_weights(std::span<const size_t> order, T positives, ForegroundAt foreground_at, std::span<T> weights) {
    T intersection = positives, union_size = positives, previous = T(0);
    for (size_t i = 0; i < order.size(); ++i) {
        const T foreground = foreground_at(order[i]);
        intersection -= foreground;
        union_size += T(1) - foreground;
        const T jaccard = T(1) - intersection / std::max(union_size, core::epsilon<T>);
        weights[i] = jaccard - previous;
        previous = jaccard;
    }
}

/**
 * @brief Lovasz kayiplarinin tekrar kullanilan siralama tamponlari.
 * Ilk cagri kapasiteyi kurar; ayni boyutta sonraki cagrilar ayirmaz.
 */
template<typename T>
struct LovaszWorkspace {
    std::vector<size_t> order;
    std::vector<T> errors, weights;

    void prepare(size_t count) {
        order.resize(count);
        errors.resize(count);
        weights.resize(count);
    }
};

namespace detail {

template<typename T>
void sort_by_error_descending(std::span<const T> errors, std::span<size_t> order) {
    std::iota(order.begin(), order.end(), size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return errors[a] > errors[b]; });
}

template<typename T>
T hinge_sign(T target) { return target > T(0) ? T(1) : T(-1); }

} // namespace detail

/// @brief Ikili Lovasz hinge. logits ham skor, target {0, 1} maskesi.
template<typename T>
T lovasz_hinge_loss(std::span<const T> logits, std::span<const T> target, LovaszWorkspace<T>& workspace) {
    validate(logits, target);
    workspace.prepare(logits.size());
    T positives = T(0);
    for (size_t i = 0; i < logits.size(); ++i) {
        workspace.errors[i] = T(1) - logits[i] * detail::hinge_sign(target[i]);
        positives += target[i] > T(0) ? T(1) : T(0);
    }
    detail::sort_by_error_descending<T>(workspace.errors, workspace.order);
    lovasz_weights<T>(workspace.order, positives,
                      [&](size_t index) { return target[index] > T(0) ? T(1) : T(0); }, workspace.weights);
    T loss = T(0);
    for (size_t i = 0; i < logits.size(); ++i)
        loss += std::max(workspace.errors[workspace.order[i]], T(0)) * workspace.weights[i];
    return loss;
}

template<typename T>
T lovasz_hinge_loss(std::span<const T> logits, std::span<const T> target) {
    LovaszWorkspace<T> workspace;
    return lovasz_hinge_loss<T>(logits, target, workspace);
}

/**
 * @brief Lovasz hinge subgradyani: d e_i / d logit_i = -sign_i, ReLU aktifse
 *        siralanmis agirlik ile carpilir. Siralama forward'daki ile aynidir.
 */
template<typename T>
void lovasz_hinge_gradient_into(std::span<const T> logits, std::span<const T> target, std::span<T> grad_out,
                                LovaszWorkspace<T>& workspace) {
    if (grad_out.size() != logits.size())
        throw std::invalid_argument("NexusLoss: grad_out boyutu pred ile ayni olmali");
    lovasz_hinge_loss<T>(logits, target, workspace);
    for (size_t i = 0; i < logits.size(); ++i) {
        const size_t index = workspace.order[i];
        grad_out[index] = workspace.errors[index] > T(0)
            ? -detail::hinge_sign(target[index]) * workspace.weights[i]
            : T(0);
    }
}

namespace detail {

template<typename T>
void validate_lovasz_softmax(std::span<const T> probabilities, std::span<const size_t> target_classes,
                             size_t pixels, size_t classes) {
    if (!pixels || !classes || probabilities.size() != pixels * classes || target_classes.size() != pixels)
        throw std::invalid_argument("NexusLoss: Lovasz-Softmax expects pixel-major [pixel, class] probabilities");
    for (size_t pixel = 0; pixel < pixels; ++pixel)
        if (target_classes[pixel] >= classes)
            throw std::out_of_range("NexusLoss: segmentation target class is out of range");
}

/// Tek sinif icin hata, siralama ve agirliklari kurar. Sinif hedefte yoksa false.
template<typename T>
bool lovasz_softmax_class(std::span<const T> probabilities, std::span<const size_t> target_classes,
                          size_t pixels, size_t classes, size_t klass, LovaszWorkspace<T>& workspace) {
    T positives = T(0);
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
        const T foreground = target_classes[pixel] == klass ? T(1) : T(0);
        positives += foreground;
        workspace.errors[pixel] = std::abs(foreground - probabilities[pixel * classes + klass]);
    }
    if (positives == T(0)) return false;
    sort_by_error_descending<T>(workspace.errors, workspace.order);
    lovasz_weights<T>(workspace.order, positives,
                      [&](size_t pixel) { return target_classes[pixel] == klass ? T(1) : T(0); }, workspace.weights);
    return true;
}

} // namespace detail

template<typename T>
T lovasz_softmax_loss(std::span<const T> probabilities, std::span<const size_t> target_classes,
                      size_t pixels, size_t classes, LovaszWorkspace<T>& workspace) {
    detail::validate_lovasz_softmax(probabilities, target_classes, pixels, classes);
    workspace.prepare(pixels);
    T total_loss = T(0);
    size_t present_classes = 0;
    for (size_t klass = 0; klass < classes; ++klass) {
        if (!detail::lovasz_softmax_class<T>(probabilities, target_classes, pixels, classes, klass, workspace))
            continue;
        ++present_classes;
        for (size_t i = 0; i < pixels; ++i)
            total_loss += workspace.errors[workspace.order[i]] * workspace.weights[i];
    }
    return present_classes ? total_loss / static_cast<T>(present_classes) : T(0);
}

template<typename T>
T lovasz_softmax_loss(std::span<const T> probabilities, std::span<const size_t> target_classes,
                      size_t pixels, size_t classes) {
    LovaszWorkspace<T> workspace;
    return lovasz_softmax_loss<T>(probabilities, target_classes, pixels, classes, workspace);
}

/**
 * @brief Lovasz-Softmax subgradyani. |fg - p| turevi p > fg ise +1, p < fg ise -1,
 *        esitlikte 0. Siralanmis agirlik ile carpilip mevcut sinif sayisina bolunur.
 */
template<typename T>
void lovasz_softmax_gradient_into(std::span<const T> probabilities, std::span<const size_t> target_classes,
                                  size_t pixels, size_t classes, std::span<T> grad_out,
                                  LovaszWorkspace<T>& workspace) {
    detail::validate_lovasz_softmax(probabilities, target_classes, pixels, classes);
    if (grad_out.size() != probabilities.size())
        throw std::invalid_argument("NexusLoss: grad_out boyutu pred ile ayni olmali");
    workspace.prepare(pixels);
    std::fill(grad_out.begin(), grad_out.end(), T(0));
    size_t present = 0;
    for (size_t klass = 0; klass < classes; ++klass) {
        if (!detail::lovasz_softmax_class<T>(probabilities, target_classes, pixels, classes, klass, workspace))
            continue;
        ++present;
        for (size_t i = 0; i < pixels; ++i) {
            const size_t pixel = workspace.order[i];
            const T delta = probabilities[pixel * classes + klass] - (target_classes[pixel] == klass ? T(1) : T(0));
            const T sign = delta > T(0) ? T(1) : (delta < T(0) ? T(-1) : T(0));
            grad_out[pixel * classes + klass] = workspace.weights[i] * sign;
        }
    }
    if (present) {
        const T scale = T(1) / static_cast<T>(present);
        for (T& value : grad_out) value *= scale;
    }
}

/// @brief Dice loss skalerinin her piksele gore turevi.
/// dL/dp_i = - (2 t_i (D+s) - (2I+s)) / (D+s)^2
template<typename T>
void dice_loss_gradient_into(std::span<const T> prediction, std::span<const T> target, std::span<T> grad_out,
                             T smooth = T(1)) {
    validate(prediction, target);
    if (grad_out.size() != prediction.size())
        throw std::invalid_argument("NexusLoss: grad_out boyutu pred ile ayni olmali");
    T intersection = T(0), denominator = T(0);
    for (size_t i = 0; i < prediction.size(); ++i) {
        intersection += prediction[i] * target[i];
        denominator += prediction[i] + target[i];
    }
    const T num = T(2) * intersection + smooth;
    const T den = denominator + smooth;
    const T den2 = den * den;
    for (size_t i = 0; i < prediction.size(); ++i)
        grad_out[i] = -(T(2) * target[i] * den - num) / den2;
}

template<typename T>
std::vector<T> dice_loss_gradient(std::span<const T> prediction, std::span<const T> target, T smooth = T(1)) {
    std::vector<T> grad(prediction.size());
    dice_loss_gradient_into<T>(prediction, target, grad, smooth);
    return grad;
}

/// @brief Soft IoU / Jaccard loss turevi.
template<typename T>
void iou_loss_gradient_into(std::span<const T> prediction, std::span<const T> target, std::span<T> grad_out,
                            T smooth = T(1)) {
    validate(prediction, target);
    if (grad_out.size() != prediction.size())
        throw std::invalid_argument("NexusLoss: grad_out boyutu pred ile ayni olmali");
    T intersection = T(0), total = T(0);
    for (size_t i = 0; i < prediction.size(); ++i) {
        intersection += prediction[i] * target[i];
        total += prediction[i] + target[i] - prediction[i] * target[i];
    }
    const T num = intersection + smooth;
    const T den = total + smooth;
    const T den2 = den * den;
    for (size_t i = 0; i < prediction.size(); ++i) {
        const T d_num = target[i];
        const T d_den = T(1) - target[i];
        grad_out[i] = -(d_num * den - num * d_den) / den2;
    }
}

template<typename T>
std::vector<T> iou_loss_gradient(std::span<const T> prediction, std::span<const T> target, T smooth = T(1)) {
    std::vector<T> grad(prediction.size());
    iou_loss_gradient_into<T>(prediction, target, grad, smooth);
    return grad;
}

/// @brief Tversky skalerinin piksel turevi. FP agirligi alpha, FN agirligi beta.
template<typename T>
void tversky_loss_gradient_into(std::span<const T> prediction, std::span<const T> target, std::span<T> grad_out,
                                T alpha = T(0.5), T beta = T(0.5), T smooth = T(1)) {
    validate(prediction, target);
    if (grad_out.size() != prediction.size())
        throw std::invalid_argument("NexusLoss: grad_out boyutu pred ile ayni olmali");
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
    for (size_t i = 0; i < prediction.size(); ++i) {
        const T d_tp = target[i];
        const T d_fp = T(1) - target[i];
        const T d_fn = -target[i];
        const T d_num = d_tp;
        const T d_den = d_tp + alpha * d_fp + beta * d_fn;
        grad_out[i] = -(d_num * den - num * d_den) / den2;
    }
}

template<typename T>
std::vector<T> tversky_loss_gradient(std::span<const T> prediction, std::span<const T> target,
                                     T alpha = T(0.5), T beta = T(0.5), T smooth = T(1)) {
    std::vector<T> grad(prediction.size());
    tversky_loss_gradient_into<T>(prediction, target, grad, alpha, beta, smooth);
    return grad;
}

template<typename T>
void focal_tversky_loss_gradient_into(std::span<const T> prediction, std::span<const T> target, std::span<T> grad_out,
                                      T alpha = T(0.5), T beta = T(0.5), T gamma = T(4) / T(3)) {
    if (gamma <= T(0)) throw std::invalid_argument("NexusLoss: focal Tversky gamma must be positive");
    const T base = tversky_loss(prediction, target, alpha, beta);
    const T scale = gamma * std::pow(std::max(base, core::epsilon<T>), gamma - T(1));
    tversky_loss_gradient_into<T>(prediction, target, grad_out, alpha, beta);
    for (T& value : grad_out) value *= scale;
}

template<typename T>
std::vector<T> focal_tversky_loss_gradient(std::span<const T> prediction, std::span<const T> target,
                                           T alpha = T(0.5), T beta = T(0.5), T gamma = T(4) / T(3)) {
    std::vector<T> grad(prediction.size());
    focal_tversky_loss_gradient_into<T>(prediction, target, grad, alpha, beta, gamma);
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
    [[nodiscard]] size_t gradient_size() const noexcept { return cached_prediction_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        dice_loss_gradient_into<T>(cached_prediction_, cached_target_, grad_out, smooth_);
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
    [[nodiscard]] size_t gradient_size() const noexcept { return cached_prediction_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        iou_loss_gradient_into<T>(cached_prediction_, cached_target_, grad_out, smooth_);
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
    [[nodiscard]] size_t gradient_size() const noexcept { return cached_prediction_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        tversky_loss_gradient_into<T>(cached_prediction_, cached_target_, grad_out, alpha_, beta_, smooth_);
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
    [[nodiscard]] size_t gradient_size() const noexcept { return cached_prediction_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        focal_tversky_loss_gradient_into<T>(cached_prediction_, cached_target_, grad_out, alpha_, beta_, gamma_);
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
 * Siralama tamponlari forward'da kurulur; backward_into ayni kapasiteyi kullanir
 * ve ayirma yapmaz. Ayni nesneye es zamanli backward cagrilmaz.
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
        const T loss = lovasz_softmax_loss<T>(probabilities, target_classes, pixels, classes, workspace_);
        ready_ = true;
        return loss;
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return cached_probabilities_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        lovasz_softmax_gradient_into<T>(cached_probabilities_, cached_targets_, pixels_, classes_, grad_out,
                                        workspace_);
    }
private:
    std::vector<T> cached_probabilities_;
    std::vector<size_t> cached_targets_;
    size_t pixels_ = 0, classes_ = 0;
    mutable LovaszWorkspace<T> workspace_;
    bool ready_ = false;
};

/**
 * @class LovaszHingeLoss
 * @brief Ikili segmentasyon icin Lovasz hinge (Berman et al., 2018, bol. 3.1).
 * Girdi ham logit ve {0, 1} maskesidir; IoU'nun dogrudan konveks vekilidir.
 */
template<typename T>
class LovaszHingeLoss {
public:
    T forward(std::span<const T> logits, std::span<const T> target) {
        cached_logits_.assign(logits.begin(), logits.end());
        cached_target_.assign(target.begin(), target.end());
        const T loss = lovasz_hinge_loss<T>(logits, target, workspace_);
        ready_ = true;
        return loss;
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return cached_logits_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        lovasz_hinge_gradient_into<T>(cached_logits_, cached_target_, grad_out, workspace_);
    }
private:
    std::vector<T> cached_logits_, cached_target_;
    mutable LovaszWorkspace<T> workspace_;
    bool ready_ = false;
};

} // namespace nexusloss::segmentation

#endif //NEXUSLOSS_SEGMENTATION_HPP
