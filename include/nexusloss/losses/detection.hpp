/**
 * @file detection.hpp
 * @brief Nesne tespiti icin kutu regresyonu ve objectness kayiplari.
 *
 * Kutular [x1, y1, x2, y2] duzenindedir. IoU ailesi kesisim kalitesini,
 * Smooth-L1 ofset regresyonunu, DetectionFocalLoss ise RetinaNet tarzi
 * logit uzerindeki sinif dengesizligini olcer. Gradyanlar tahmin kutusunun
 * dort koordinatina (veya logitlere) gore analitiktir.
 *
 * @see core/utils.hpp  stable_sigmoid ve epsilon
 */
#ifndef NEXUSLOSS_DETECTION_HPP
#define NEXUSLOSS_DETECTION_HPP

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>
#include "../core/utils.hpp"

namespace nexusloss::detection {

template<typename T>
void validate_box(std::span<const T> box) {
    if (box.size() != 4) throw std::invalid_argument("NexusLoss: a box must be [x1, y1, x2, y2]");
    if (box[2] < box[0] || box[3] < box[1])
        throw std::invalid_argument("NexusLoss: box maximum coordinates must not precede minimum coordinates");
}

template<typename T>
T iou(std::span<const T> prediction, std::span<const T> target) {
    validate_box(prediction);
    validate_box(target);
    const T ix = std::max(T(0), std::min(prediction[2], target[2]) - std::max(prediction[0], target[0]));
    const T iy = std::max(T(0), std::min(prediction[3], target[3]) - std::max(prediction[1], target[1]));
    const T intersection = ix * iy;
    const T area_p = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1]);
    const T area_t = (target[2] - target[0]) * (target[3] - target[1]);
    return intersection / std::max(area_p + area_t - intersection, core::epsilon<T>);
}

template<typename T>
T iou_loss(std::span<const T> prediction, std::span<const T> target) {
    return T(1) - iou(prediction, target);
}

template<typename T>
T giou_loss(std::span<const T> prediction, std::span<const T> target) {
    const T overlap = iou(prediction, target);
    const T enclosing_width = std::max(prediction[2], target[2]) - std::min(prediction[0], target[0]);
    const T enclosing_height = std::max(prediction[3], target[3]) - std::min(prediction[1], target[1]);
    const T enclosing_area = enclosing_width * enclosing_height;
    const T area_p = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1]);
    const T area_t = (target[2] - target[0]) * (target[3] - target[1]);
    const T ix = std::max(T(0), std::min(prediction[2], target[2]) - std::max(prediction[0], target[0]));
    const T iy = std::max(T(0), std::min(prediction[3], target[3]) - std::max(prediction[1], target[1]));
    const T union_area = area_p + area_t - ix * iy;
    return T(1) - overlap + (enclosing_area - union_area) / std::max(enclosing_area, core::epsilon<T>);
}

template<typename T>
T diou_loss(std::span<const T> prediction, std::span<const T> target) {
    const T overlap = iou(prediction, target);
    const T dx = (prediction[0] + prediction[2] - target[0] - target[2]) / T(2);
    const T dy = (prediction[1] + prediction[3] - target[1] - target[3]) / T(2);
    const T width = std::max(prediction[2], target[2]) - std::min(prediction[0], target[0]);
    const T height = std::max(prediction[3], target[3]) - std::min(prediction[1], target[1]);
    return T(1) - overlap + (dx * dx + dy * dy) / std::max(width * width + height * height, core::epsilon<T>);
}

template<typename T>
T ciou_loss(std::span<const T> prediction, std::span<const T> target) {
    const T overlap = iou(prediction, target);
    const T dx = (prediction[0] + prediction[2] - target[0] - target[2]) / T(2);
    const T dy = (prediction[1] + prediction[3] - target[1] - target[3]) / T(2);
    const T width = std::max(prediction[2], target[2]) - std::min(prediction[0], target[0]);
    const T height = std::max(prediction[3], target[3]) - std::min(prediction[1], target[1]);
    const T distance = (dx * dx + dy * dy) / std::max(width * width + height * height, core::epsilon<T>);
    const T wp = prediction[2] - prediction[0], hp = prediction[3] - prediction[1];
    const T wt = target[2] - target[0], ht = target[3] - target[1];
    const T angle = T(4) / (std::acos(T(-1)) * std::acos(T(-1)))
                    * std::pow(std::atan(wt / std::max(ht, core::epsilon<T>))
                               - std::atan(wp / std::max(hp, core::epsilon<T>)), T(2));
    const T alpha = angle / std::max(T(1) - overlap + angle, core::epsilon<T>);
    return T(1) - overlap + distance + alpha * angle;
}

namespace detail {

template<typename T>
struct BoxParts {
    T inter{}, union_area{}, iou{}, area_p{};
    T iw{}, ih{};
    bool pred_left{}, pred_right{}, pred_top{}, pred_bottom{};
};

template<typename T>
BoxParts<T> box_parts(std::span<const T> prediction, std::span<const T> target) {
    validate_box(prediction);
    validate_box(target);
    BoxParts<T> parts;
    const T x1 = std::max(prediction[0], target[0]);
    const T y1 = std::max(prediction[1], target[1]);
    const T x2 = std::min(prediction[2], target[2]);
    const T y2 = std::min(prediction[3], target[3]);
    parts.iw = std::max(T(0), x2 - x1);
    parts.ih = std::max(T(0), y2 - y1);
    parts.inter = parts.iw * parts.ih;
    parts.area_p = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1]);
    const T area_t = (target[2] - target[0]) * (target[3] - target[1]);
    parts.union_area = parts.area_p + area_t - parts.inter;
    parts.iou = parts.inter / std::max(parts.union_area, core::epsilon<T>);
    parts.pred_left = prediction[0] >= target[0];
    parts.pred_right = prediction[2] <= target[2];
    parts.pred_top = prediction[1] >= target[1];
    parts.pred_bottom = prediction[3] <= target[3];
    return parts;
}

template<typename T>
void intersection_partials(const BoxParts<T>& parts, T d_inter[4]) {
    const T diw_x1 = (parts.iw > T(0) && parts.pred_left) ? T(-1) : T(0);
    const T diw_x2 = (parts.iw > T(0) && parts.pred_right) ? T(1) : T(0);
    const T dih_y1 = (parts.ih > T(0) && parts.pred_top) ? T(-1) : T(0);
    const T dih_y2 = (parts.ih > T(0) && parts.pred_bottom) ? T(1) : T(0);
    d_inter[0] = parts.ih * diw_x1;
    d_inter[2] = parts.ih * diw_x2;
    d_inter[1] = parts.iw * dih_y1;
    d_inter[3] = parts.iw * dih_y2;
}

template<typename T>
void area_partials(std::span<const T> prediction, T d_area[4]) {
    const T width = prediction[2] - prediction[0];
    const T height = prediction[3] - prediction[1];
    d_area[0] = -height;
    d_area[2] = height;
    d_area[1] = -width;
    d_area[3] = width;
}

template<typename T>
void enclosing_partials(std::span<const T> prediction, std::span<const T> target, T& c_area, T d_c[4]) {
    const T width = std::max(prediction[2], target[2]) - std::min(prediction[0], target[0]);
    const T height = std::max(prediction[3], target[3]) - std::min(prediction[1], target[1]);
    c_area = width * height;
    const T d_cw_x1 = prediction[0] <= target[0] ? T(-1) : T(0);
    const T d_cw_x2 = prediction[2] >= target[2] ? T(1) : T(0);
    const T d_ch_y1 = prediction[1] <= target[1] ? T(-1) : T(0);
    const T d_ch_y2 = prediction[3] >= target[3] ? T(1) : T(0);
    d_c[0] = height * d_cw_x1;
    d_c[2] = height * d_cw_x2;
    d_c[1] = width * d_ch_y1;
    d_c[3] = width * d_ch_y2;
}

} // namespace detail

/**
 * @brief 1 - IoU kaybinin kutu koordinatlarina gore turevi (x1,y1,x2,y2).
 *
 * Kesisim kenari, min/max hangi kutuya aitse o koordinata akar. Iki kutu
 * ic ice degil de kenarlari ayni hizada ise subgradyan tek tarafa yazilir;
 * testler kesisen ve hizasi kaymis kutular kullanmalidir.
 */
template<typename T>
std::vector<T> iou_loss_gradient(std::span<const T> prediction, std::span<const T> target) {
    const auto parts = detail::box_parts<T>(prediction, target);
    T d_inter[4], d_area[4];
    detail::intersection_partials(parts, d_inter);
    detail::area_partials(prediction, d_area);
    const T denom = std::max(parts.union_area, core::epsilon<T>);
    std::vector<T> grad(4);
    for (int i = 0; i < 4; ++i) {
        const T d_union = d_area[i] - d_inter[i];
        const T d_iou = (d_inter[i] * denom - parts.inter * d_union) / (denom * denom);
        grad[static_cast<size_t>(i)] = -d_iou;
    }
    return grad;
}

template<typename T>
std::vector<T> giou_loss_gradient(std::span<const T> prediction, std::span<const T> target) {
    auto grad = iou_loss_gradient(prediction, target);
    const auto parts = detail::box_parts<T>(prediction, target);
    T c_area = T(0), d_c[4], d_inter[4], d_area[4];
    detail::enclosing_partials(prediction, target, c_area, d_c);
    detail::intersection_partials(parts, d_inter);
    detail::area_partials(prediction, d_area);
    const T safe_c = std::max(c_area, core::epsilon<T>);
    for (int i = 0; i < 4; ++i) {
        const T d_union = d_area[i] - d_inter[i];
        const T d_penalty = -(d_union * safe_c - parts.union_area * d_c[i]) / (safe_c * safe_c);
        grad[static_cast<size_t>(i)] += d_penalty;
    }
    return grad;
}

template<typename T>
std::vector<T> diou_loss_gradient(std::span<const T> prediction, std::span<const T> target) {
    auto grad = iou_loss_gradient(prediction, target);
    const T dx = (prediction[0] + prediction[2] - target[0] - target[2]) / T(2);
    const T dy = (prediction[1] + prediction[3] - target[1] - target[3]) / T(2);
    const T rho2 = dx * dx + dy * dy;
    T c_area = T(0), d_c_unused[4];
    detail::enclosing_partials(prediction, target, c_area, d_c_unused);
    const T cw = std::max(prediction[2], target[2]) - std::min(prediction[0], target[0]);
    const T ch = std::max(prediction[3], target[3]) - std::min(prediction[1], target[1]);
    const T c2 = std::max(cw * cw + ch * ch, core::epsilon<T>);
    T d_c2[4] = {
        T(2) * cw * (prediction[0] <= target[0] ? T(-1) : T(0)),
        T(2) * ch * (prediction[1] <= target[1] ? T(-1) : T(0)),
        T(2) * cw * (prediction[2] >= target[2] ? T(1) : T(0)),
        T(2) * ch * (prediction[3] >= target[3] ? T(1) : T(0))
    };
    const T d_rho[4] = {dx, dy, dx, dy};
    for (int i = 0; i < 4; ++i) {
        const T d_ratio = (d_rho[i] * c2 - rho2 * d_c2[i]) / (c2 * c2);
        grad[static_cast<size_t>(i)] += d_ratio;
    }
    (void)c_area;
    return grad;
}

template<typename T>
std::vector<T> ciou_loss_gradient(std::span<const T> prediction, std::span<const T> target) {
    auto grad = diou_loss_gradient(prediction, target);
    const auto parts = detail::box_parts<T>(prediction, target);
    const T pi = std::acos(T(-1));
    const T wp = prediction[2] - prediction[0];
    const T hp = std::max(prediction[3] - prediction[1], core::epsilon<T>);
    const T wt = target[2] - target[0];
    const T ht = std::max(target[3] - target[1], core::epsilon<T>);
    const T angle_delta = std::atan(wt / ht) - std::atan(wp / hp);
    const T v = T(4) / (pi * pi) * angle_delta * angle_delta;
    const T u = std::max(T(1) - parts.iou + v, core::epsilon<T>);
    const T denom_atan = hp * hp + wp * wp;
    const T d_atan_wp = hp / denom_atan;
    const T d_atan_hp = -wp / denom_atan;
    const T d_v_wp = (T(8) / (pi * pi)) * angle_delta * (-d_atan_wp);
    const T d_v_hp = (T(8) / (pi * pi)) * angle_delta * (-d_atan_hp);
    T d_v[4] = {-d_v_wp, -d_v_hp, d_v_wp, d_v_hp};
    auto d_iou = iou_loss_gradient(prediction, target);
    for (T& value : d_iou) value = -value;
    for (int i = 0; i < 4; ++i) {
        const T d_u = -d_iou[static_cast<size_t>(i)] + d_v[i];
        const T d_alpha_v = (T(2) * v * d_v[i] * u - v * v * d_u) / (u * u);
        grad[static_cast<size_t>(i)] += d_alpha_v;
    }
    return grad;
}

/**
 * @class DetectionFocalLoss
 * @brief RetinaNet objectness/sinif kaybi: sigmoid + focal, logit uzerinde kararli.
 *
 * L = -a_t (1-p_t)^g log(p_t), p = sigmoid(x)
 * log(p_t) ayri yazilmaz; BCE-with-logits ile carpilir (Lin et al., 2017).
 * prediction ve target eleman bazli, target 0/1.
 */
template<typename T>
class DetectionFocalLoss {
public:
    DetectionFocalLoss(T alpha = T(0.25), T gamma = T(2)) : alpha_(alpha), gamma_(gamma) {
        if (alpha < T(0) || alpha > T(1) || gamma < T(0))
            throw std::invalid_argument("NexusLoss: invalid detection focal parameters");
    }

    T forward(std::span<const T> logits, std::span<const T> target) {
        if (logits.empty() || logits.size() != target.size())
            throw std::invalid_argument("NexusLoss: detection focal inputs must match");
        cached_logits_.assign(logits.begin(), logits.end());
        cached_target_.assign(target.begin(), target.end());
        ready_ = true;
        T sum = T(0);
        for (size_t i = 0; i < logits.size(); ++i) sum += element(logits[i], target[i]);
        return sum / static_cast<T>(logits.size());
    }

    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        std::vector<T> grad(cached_logits_.size());
        const T inv = T(1) / static_cast<T>(cached_logits_.size());
        for (size_t i = 0; i < cached_logits_.size(); ++i)
            grad[i] = element_gradient(cached_logits_[i], cached_target_[i]) * inv;
        return grad;
    }

private:
    T element(T logit, T target) const {
        const T ce = std::max(logit, T(0)) - logit * target + std::log1p(std::exp(-std::abs(logit)));
        const T probability = core::stable_sigmoid(logit);
        const bool positive = target >= T(0.5);
        const T pt = positive ? probability : T(1) - probability;
        const T alpha_t = positive ? alpha_ : T(1) - alpha_;
        return alpha_t * std::pow(std::max(T(1) - pt, core::epsilon<T>), gamma_) * ce;
    }

    T element_gradient(T logit, T target) const {
        const T probability = core::stable_sigmoid(logit);
        const T ce = std::max(logit, T(0)) - logit * target + std::log1p(std::exp(-std::abs(logit)));
        const T d_ce = probability - target;
        const bool positive = target >= T(0.5);
        const T pt = core::clamp(positive ? probability : T(1) - probability, core::epsilon<T>, T(1));
        const T alpha_t = positive ? alpha_ : T(1) - alpha_;
        const T d_pt = probability * (T(1) - probability) * (positive ? T(1) : T(-1));
        const T one_minus = T(1) - pt;
        const T d_mod = gamma_ == T(0) ? T(0)
                        : gamma_ * std::pow(one_minus, gamma_ - T(1)) * (-d_pt);
        return alpha_t * (d_mod * ce + std::pow(one_minus, gamma_) * d_ce);
    }

    T alpha_, gamma_;
    std::vector<T> cached_logits_, cached_target_;
    bool ready_ = false;
};

/**
 * @class SmoothL1BBoxLoss
 * @brief Faster R-CNN kutu regresyonu: delta vektoru uzerinde Huber/Smooth-L1.
 *
 * Girard / Girshick: |hata| <= beta iken 0.5 hata^2, disinda beta*(|hata|-0.5 beta).
 * prediction ve target ayni uzunlukta kutu ofsetleridir (dx, dy, dw, dh).
 */
template<typename T>
class SmoothL1BBoxLoss {
public:
    explicit SmoothL1BBoxLoss(T beta = T(1)) : beta_(beta) {
        if (!(beta > T(0))) throw std::invalid_argument("NexusLoss: Smooth-L1 beta must be positive");
    }
    T forward(std::span<const T> prediction, std::span<const T> target) {
        if (prediction.empty() || prediction.size() != target.size())
            throw std::invalid_argument("NexusLoss: bbox tensors must match");
        cached_prediction_.assign(prediction.begin(), prediction.end());
        cached_target_.assign(target.begin(), target.end());
        ready_ = true;
        T sum = T(0);
        for (size_t i = 0; i < prediction.size(); ++i) {
            const T error = std::abs(prediction[i] - target[i]);
            sum += error <= beta_ ? T(0.5) * error * error : beta_ * (error - T(0.5) * beta_);
        }
        return sum / static_cast<T>(prediction.size());
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        std::vector<T> grad(cached_prediction_.size());
        const T inv = T(1) / static_cast<T>(grad.size());
        for (size_t i = 0; i < grad.size(); ++i) {
            const T diff = cached_prediction_[i] - cached_target_[i];
            const T raw = std::abs(diff) <= beta_ ? diff : (diff > T(0) ? beta_ : -beta_);
            grad[i] = raw * inv;
        }
        return grad;
    }
private:
    T beta_;
    std::vector<T> cached_prediction_, cached_target_;
    bool ready_ = false;
};

template<typename T>
class IoULoss {
public:
    T forward(std::span<const T> prediction, std::span<const T> target) { remember(prediction, target); return iou_loss(prediction, target); }
    std::vector<T> backward() const { ensure(); return iou_loss_gradient<T>(prediction_, target_); }
protected:
    void remember(std::span<const T> prediction, std::span<const T> target) {
        prediction_.assign(prediction.begin(), prediction.end());
        target_.assign(target.begin(), target.end());
        ready_ = true;
    }
    void ensure() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
    }
    std::vector<T> prediction_, target_;
    bool ready_ = false;
};

template<typename T>
class GIoULoss : public IoULoss<T> {
public:
    T forward(std::span<const T> prediction, std::span<const T> target) {
        this->remember(prediction, target);
        return giou_loss(prediction, target);
    }
    std::vector<T> backward() const { this->ensure(); return giou_loss_gradient<T>(this->prediction_, this->target_); }
};

template<typename T>
class DIoULoss : public IoULoss<T> {
public:
    T forward(std::span<const T> prediction, std::span<const T> target) {
        this->remember(prediction, target);
        return diou_loss(prediction, target);
    }
    std::vector<T> backward() const { this->ensure(); return diou_loss_gradient<T>(this->prediction_, this->target_); }
};

template<typename T>
class CIoULoss : public IoULoss<T> {
public:
    T forward(std::span<const T> prediction, std::span<const T> target) {
        this->remember(prediction, target);
        return ciou_loss(prediction, target);
    }
    std::vector<T> backward() const { this->ensure(); return ciou_loss_gradient<T>(this->prediction_, this->target_); }
};

} // namespace nexusloss::detection

#endif
