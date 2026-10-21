/**
 * @file self_supervised.hpp
 * @brief Etiketsiz temsil ogrenimi icin kontrastif ve cift-goruntu kayiplari.
 *
 * InfoNCE/NT-Xent benzerlik skorlari ve sicaklik alir; payda log-sum-exp'tir.
 * BYOL negatif ornek istemez. Barlow Twins kare capraz-korelasyon matrisi bekler.
 * VICReg ham [N, D] gomuleri alir; istatistik tamponlari `VICRegWorkspace`'tedir.
 * Her sinifin `backward_into` yolu ayirma yapmaz.
 *
 * @see van den Oord et al., CPC / InfoNCE, 2018
 * @see Zbontar et al., Barlow Twins, 2021
 * @see Bardes, Ponce, LeCun, VICReg, 2022
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

/// @brief VICReg terim agirliklari (Bardes, Ponce, LeCun, 2022 varsayilanlari).
template<typename T>
struct VICRegWeights {
    T invariance = T(25);
    T variance = T(25);
    T covariance = T(1);
    T gamma = T(1);
};

/**
 * @brief VICReg istatistik tamponlari. Iki gorunum icin ortalama, standart
 *        sapma ve D x D kovaryans. `prepare` kapasiteyi bir kez kurar;
 *        sabit boyutlu dongude sonraki cagrilar ayirmaz.
 */
template<typename T>
struct VICRegWorkspace {
    std::vector<T> mean_a, mean_b, std_a, std_b, cov_a, cov_b;
    void prepare(size_t dimensions) {
        mean_a.resize(dimensions); mean_b.resize(dimensions);
        std_a.resize(dimensions); std_b.resize(dimensions);
        cov_a.resize(dimensions * dimensions); cov_b.resize(dimensions * dimensions);
    }
};

namespace detail {

template<typename T>
void validate_vicreg(std::span<const T> view_a, std::span<const T> view_b, size_t samples, size_t dimensions,
                     const VICRegWeights<T>& weights) {
    if (!samples || !dimensions || view_a.size() != samples * dimensions || view_b.size() != view_a.size())
        throw std::invalid_argument("NexusLoss: VICReg inputs must be flattened sample-major matrices");
    if (weights.invariance < T(0) || weights.variance < T(0) || weights.covariance < T(0) || weights.gamma < T(0))
        throw std::invalid_argument("NexusLoss: VICReg weights and gamma must be non-negative");
}

/// Sutun ortalamasi, sqrt(yanli varyans + eps) ve yansiz kovaryans (N > 1 ise).
template<typename T>
void vicreg_statistics(std::span<const T> view, size_t samples, size_t dimensions,
                       std::span<T> mean, std::span<T> std_dev, std::span<T> covariance) {
    std::fill(mean.begin(), mean.end(), T(0));
    for (size_t n = 0; n < samples; ++n)
        for (size_t d = 0; d < dimensions; ++d) mean[d] += view[n * dimensions + d];
    const T inverse_samples = T(1) / static_cast<T>(samples);
    for (T& value : mean) value *= inverse_samples;

    std::fill(covariance.begin(), covariance.end(), T(0));
    for (size_t n = 0; n < samples; ++n) {
        const T* row = view.data() + n * dimensions;
        for (size_t i = 0; i < dimensions; ++i) {
            const T xi = row[i] - mean[i];
            for (size_t j = i; j < dimensions; ++j) covariance[i * dimensions + j] += xi * (row[j] - mean[j]);
        }
    }
    for (size_t d = 0; d < dimensions; ++d)
        std_dev[d] = std::sqrt(covariance[d * dimensions + d] * inverse_samples + core::epsilon<T>);
    const T unbiased = samples > 1 ? T(1) / static_cast<T>(samples - 1) : T(0);
    for (size_t i = 0; i < dimensions; ++i)
        for (size_t j = i; j < dimensions; ++j) {
            const T value = covariance[i * dimensions + j] * unbiased;
            covariance[i * dimensions + j] = value;
            covariance[j * dimensions + i] = value;
        }
}

template<typename T>
T vicreg_variance_term(std::span<const T> std_dev, T gamma) {
    T sum = T(0);
    for (T value : std_dev) sum += std::max(T(0), gamma - value);
    return sum;
}

template<typename T>
T vicreg_off_diagonal_energy(std::span<const T> covariance, size_t dimensions) {
    T sum = T(0);
    for (size_t i = 0; i < dimensions; ++i)
        for (size_t j = 0; j < dimensions; ++j)
            if (i != j) sum += covariance[i * dimensions + j] * covariance[i * dimensions + j];
    return sum;
}

/// Tek gorunumun varyans + kovaryans turevini `grad` uzerine ekler.
template<typename T>
void vicreg_view_gradient(std::span<const T> view, size_t samples, size_t dimensions,
                          std::span<const T> mean, std::span<const T> std_dev, std::span<const T> covariance,
                          const VICRegWeights<T>& weights, std::span<T> grad) {
    const T variance_scale = weights.variance / (static_cast<T>(dimensions) * static_cast<T>(samples));
    const T covariance_scale = samples > 1
        ? T(4) * weights.covariance / (static_cast<T>(dimensions) * static_cast<T>(samples - 1))
        : T(0);
    for (size_t n = 0; n < samples; ++n) {
        const T* row = view.data() + n * dimensions;
        T* out = grad.data() + n * dimensions;
        for (size_t k = 0; k < dimensions; ++k) {
            const T centered = row[k] - mean[k];
            if (weights.gamma - std_dev[k] > T(0)) out[k] -= variance_scale * centered / std_dev[k];
            if (covariance_scale != T(0)) {
                T accumulated = T(0);
                const T* cov_row = covariance.data() + k * dimensions;
                for (size_t j = 0; j < dimensions; ++j)
                    if (j != k) accumulated += cov_row[j] * (row[j] - mean[j]);
                out[k] += covariance_scale * accumulated;
            }
        }
    }
}

} // namespace detail

/**
 * @brief VICReg: degismezlik (MSE) + varyans mentesesi + kovaryans cezasi.
 *
 * Girdiler ornek-major [N, D] duzlestirilmis gomulerdir. Varyans terimi
 * sqrt(Var + eps) ile hesaplanir ve D'ye bolunur; kovaryans terimi kosegen
 * disi kare toplamidir ve D'ye bolunur. Tamponlar `workspace` icindedir.
 */
template<typename T>
T vicreg_loss(std::span<const T> view_a, std::span<const T> view_b, size_t samples, size_t dimensions,
              VICRegWorkspace<T>& workspace, const VICRegWeights<T>& weights = {}) {
    detail::validate_vicreg(view_a, view_b, samples, dimensions, weights);
    workspace.prepare(dimensions);
    T invariance = T(0);
    for (size_t i = 0; i < view_a.size(); ++i) { const T d = view_a[i] - view_b[i]; invariance += d * d; }
    invariance /= static_cast<T>(view_a.size());
    detail::vicreg_statistics<T>(view_a, samples, dimensions, workspace.mean_a, workspace.std_a, workspace.cov_a);
    detail::vicreg_statistics<T>(view_b, samples, dimensions, workspace.mean_b, workspace.std_b, workspace.cov_b);
    const T inverse_dimensions = T(1) / static_cast<T>(dimensions);
    const T variance = (detail::vicreg_variance_term<T>(workspace.std_a, weights.gamma)
                      + detail::vicreg_variance_term<T>(workspace.std_b, weights.gamma)) * inverse_dimensions;
    const T covariance = samples > 1
        ? (detail::vicreg_off_diagonal_energy<T>(workspace.cov_a, dimensions)
           + detail::vicreg_off_diagonal_energy<T>(workspace.cov_b, dimensions)) * inverse_dimensions
        : T(0);
    return weights.invariance * invariance + weights.variance * variance + weights.covariance * covariance;
}

template<typename T>
T vicreg_loss(std::span<const T> view_a, std::span<const T> view_b, size_t samples, size_t dimensions,
              T invariance_weight = T(25), T variance_weight = T(25), T covariance_weight = T(1),
              T gamma = T(1)) {
    VICRegWorkspace<T> workspace;
    return vicreg_loss<T>(view_a, view_b, samples, dimensions, workspace,
                          VICRegWeights<T>{invariance_weight, variance_weight, covariance_weight, gamma});
}

/**
 * @brief VICReg turevi, ayirmasiz. grad_out = [dL/dA (N*D), dL/dB (N*D)].
 *
 * Merkezleme turevi kovaryans icin sifirlanir (merkezli sutunlarin toplami 0);
 * varyans icin de ayni nedenle yalnizca (x - mu) / (N * std) kalir.
 * Karmasiklik O(N * D^2).
 */
namespace detail {

/// `workspace` ayni gorunumler icin `vicreg_statistics` ile doldurulmus olmalidir.
template<typename T>
void vicreg_gradient_from_statistics(std::span<const T> view_a, std::span<const T> view_b, size_t samples,
                                     size_t dimensions, std::span<T> grad_out, const VICRegWorkspace<T>& workspace,
                                     const VICRegWeights<T>& weights) {
    if (grad_out.size() != view_a.size() * 2)
        throw std::invalid_argument("NexusLoss: VICReg grad_out must hold both views");
    const std::span<T> grad_a = grad_out.first(view_a.size());
    const std::span<T> grad_b = grad_out.last(view_b.size());
    const T invariance_scale = T(2) * weights.invariance / static_cast<T>(view_a.size());
    for (size_t i = 0; i < view_a.size(); ++i) {
        const T d = invariance_scale * (view_a[i] - view_b[i]);
        grad_a[i] = d;
        grad_b[i] = -d;
    }
    vicreg_view_gradient<T>(view_a, samples, dimensions, workspace.mean_a, workspace.std_a, workspace.cov_a,
                            weights, grad_a);
    vicreg_view_gradient<T>(view_b, samples, dimensions, workspace.mean_b, workspace.std_b, workspace.cov_b,
                            weights, grad_b);
}

} // namespace detail

template<typename T>
void vicreg_gradient_into(std::span<const T> view_a, std::span<const T> view_b, size_t samples, size_t dimensions,
                          std::span<T> grad_out, VICRegWorkspace<T>& workspace, const VICRegWeights<T>& weights = {}) {
    detail::validate_vicreg(view_a, view_b, samples, dimensions, weights);
    workspace.prepare(dimensions);
    detail::vicreg_statistics<T>(view_a, samples, dimensions, workspace.mean_a, workspace.std_a, workspace.cov_a);
    detail::vicreg_statistics<T>(view_b, samples, dimensions, workspace.mean_b, workspace.std_b, workspace.cov_b);
    detail::vicreg_gradient_from_statistics<T>(view_a, view_b, samples, dimensions, grad_out, workspace, weights);
}

template<typename T>
std::vector<T> vicreg_gradient(std::span<const T> view_a, std::span<const T> view_b, size_t samples,
                               size_t dimensions, const VICRegWeights<T>& weights = {}) {
    std::vector<T> grad(view_a.size() * 2);
    VICRegWorkspace<T> workspace;
    vicreg_gradient_into<T>(view_a, view_b, samples, dimensions, grad, workspace, weights);
    return grad;
}

/**
 * @brief InfoNCE/NT-Xent turevi, ayirmasiz. grad_out[0] pozitif benzerlige,
 *        grad_out[1..] negatiflere aittir; uzunluk negatif sayisi + 1.
 */
template<typename T>
void info_nce_gradient_into(T positive_similarity, std::span<const T> negative_similarities,
                            std::span<T> grad_out, T temperature = T(0.1)) {
    if (temperature <= T(0)) throw std::invalid_argument("NexusLoss: InfoNCE temperature must be positive");
    if (grad_out.size() != negative_similarities.size() + 1)
        throw std::invalid_argument("NexusLoss: InfoNCE grad_out must hold positive + negatives");
    const T inverse = T(1) / temperature;
    T maximum = positive_similarity * inverse;
    for (T value : negative_similarities) maximum = std::max(maximum, value * inverse);
    grad_out[0] = std::exp(positive_similarity * inverse - maximum);
    T sum = grad_out[0];
    for (size_t i = 0; i < negative_similarities.size(); ++i) {
        grad_out[i + 1] = std::exp(negative_similarities[i] * inverse - maximum);
        sum += grad_out[i + 1];
    }
    const T scale = inverse / sum;
    for (T& value : grad_out) value *= scale;
    grad_out[0] -= inverse;
}

template<typename T>
std::vector<T> info_nce_gradient(T positive_similarity, std::span<const T> negative_similarities,
                                 T temperature = T(0.1)) {
    std::vector<T> grad(negative_similarities.size() + 1);
    info_nce_gradient_into<T>(positive_similarity, negative_similarities, grad, temperature);
    return grad;
}

/// @brief d(2 - 2 cos(p, t)) / dp, ayirmasiz.
template<typename T>
void byol_gradient_into(std::span<const T> prediction, std::span<const T> target, std::span<T> grad_out) {
    if (grad_out.size() != prediction.size())
        throw std::invalid_argument("NexusLoss: BYOL grad_out must match prediction");
    const T dot = metric::dot(prediction, target);
    const T pred_norm = std::sqrt(std::max(metric::dot(prediction, prediction), core::epsilon<T>));
    const T target_norm = std::sqrt(std::max(metric::dot(target, target), core::epsilon<T>));
    const T cosine = dot / (pred_norm * target_norm);
    for (size_t i = 0; i < prediction.size(); ++i) {
        const T d_cosine = (target[i] / target_norm) / pred_norm
                           - cosine * prediction[i] / (pred_norm * pred_norm);
        grad_out[i] = T(-2) * d_cosine;
    }
}

template<typename T>
std::vector<T> byol_gradient(std::span<const T> prediction, std::span<const T> target) {
    std::vector<T> grad(prediction.size());
    byol_gradient_into<T>(prediction, target, grad);
    return grad;
}

template<typename T>
void barlow_twins_gradient_into(std::span<const T> cross_correlation, size_t dimensions, std::span<T> grad_out,
                                T lambda = T(0.005)) {
    if (!dimensions || cross_correlation.size() != dimensions * dimensions || lambda < T(0))
        throw std::invalid_argument("NexusLoss: Barlow Twins requires a square correlation matrix");
    if (grad_out.size() != cross_correlation.size())
        throw std::invalid_argument("NexusLoss: Barlow Twins grad_out must match the correlation matrix");
    for (size_t i = 0; i < dimensions; ++i)
        for (size_t j = 0; j < dimensions; ++j) {
            const T target = i == j ? T(1) : T(0);
            const T weight = i == j ? T(1) : lambda;
            grad_out[i * dimensions + j] = T(2) * weight * (cross_correlation[i * dimensions + j] - target);
        }
}

template<typename T>
std::vector<T> barlow_twins_gradient(std::span<const T> cross_correlation, size_t dimensions, T lambda = T(0.005)) {
    std::vector<T> grad(cross_correlation.size());
    barlow_twins_gradient_into<T>(cross_correlation, dimensions, grad, lambda);
    return grad;
}

/// @brief DINO ogrenci logit turevi: (softmax(z/tau) * sum(q) - q) / tau.
template<typename T>
void dino_gradient_into(std::span<const T> student_logits, std::span<const T> teacher_probabilities,
                        std::span<T> grad_out, T student_temperature = T(0.1)) {
    if (student_logits.empty() || student_logits.size() != teacher_probabilities.size()
        || student_temperature <= T(0))
        throw std::invalid_argument("NexusLoss: invalid DINO inputs or temperature");
    if (grad_out.size() != student_logits.size())
        throw std::invalid_argument("NexusLoss: DINO grad_out must match student logits");
    core::softmax_into<T>(student_logits, grad_out, student_temperature);
    T teacher_mass = T(0);
    for (T value : teacher_probabilities) {
        if (value < T(0)) throw std::invalid_argument("NexusLoss: teacher probabilities must be non-negative");
        teacher_mass += value;
    }
    const T inverse = T(1) / student_temperature;
    for (size_t i = 0; i < grad_out.size(); ++i)
        grad_out[i] = (grad_out[i] * teacher_mass - teacher_probabilities[i]) * inverse;
}

template<typename T>
std::vector<T> dino_gradient(std::span<const T> student_logits, std::span<const T> teacher_probabilities,
                             T student_temperature = T(0.1)) {
    std::vector<T> grad(student_logits.size());
    dino_gradient_into<T>(student_logits, teacher_probabilities, grad, student_temperature);
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
    [[nodiscard]] size_t gradient_size() const noexcept { return negatives_.size() + 1; }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        info_nce_gradient_into<T>(positive_, negatives_, grad_out, temperature_);
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
    [[nodiscard]] size_t gradient_size() const noexcept { return prediction_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        byol_gradient_into<T>(prediction_, target_, grad_out);
    }
private:
    std::vector<T> prediction_, target_;
    bool ready_ = false;
};

/**
 * @class DINOLoss
 * @brief Ogretmen olasiliklari ile ogrenci softmax'i arasindaki cross-entropy (Caron et al., 2021).
 * Ogretmen merkezleme ve keskinlestirme cagiranin isidir; buraya olasilik gelir.
 */
template<typename T>
class DINOLoss {
public:
    explicit DINOLoss(T student_temperature = T(0.1)) : temperature_(student_temperature) {}
    T forward(std::span<const T> student_logits, std::span<const T> teacher_probabilities) {
        student_.assign(student_logits.begin(), student_logits.end());
        teacher_.assign(teacher_probabilities.begin(), teacher_probabilities.end());
        ready_ = true;
        return dino_loss(student_logits, teacher_probabilities, temperature_);
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return student_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        dino_gradient_into<T>(student_, teacher_, grad_out, temperature_);
    }
private:
    T temperature_;
    std::vector<T> student_, teacher_;
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
    [[nodiscard]] size_t gradient_size() const noexcept { return matrix_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        barlow_twins_gradient_into<T>(matrix_, dimensions_, grad_out, lambda_);
    }
private:
    T lambda_;
    std::vector<T> matrix_;
    size_t dimensions_ = 0;
    bool ready_ = false;
};

/**
 * @class VICRegLoss
 * @brief Variance-Invariance-Covariance regularizasyonu (Bardes et al., 2022).
 *
 * forward iki gorunumu ve istatistikleri saklar; backward_into ayni
 * istatistikleri tekrar hesaplamadan [dL/dA, dL/dB] yazar ve ayirmaz.
 */
template<typename T>
class VICRegLoss {
public:
    explicit VICRegLoss(VICRegWeights<T> weights = {}) : weights_(weights) {}
    T forward(std::span<const T> view_a, std::span<const T> view_b, size_t samples, size_t dimensions) {
        view_a_.assign(view_a.begin(), view_a.end());
        view_b_.assign(view_b.begin(), view_b.end());
        samples_ = samples;
        dimensions_ = dimensions;
        const T loss = vicreg_loss<T>(view_a, view_b, samples, dimensions, workspace_, weights_);
        ready_ = true;
        return loss;
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return view_a_.size() * 2; }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        detail::vicreg_gradient_from_statistics<T>(view_a_, view_b_, samples_, dimensions_, grad_out, workspace_,
                                                   weights_);
    }
private:
    VICRegWeights<T> weights_;
    std::vector<T> view_a_, view_b_;
    size_t samples_ = 0, dimensions_ = 0;
    VICRegWorkspace<T> workspace_;
    bool ready_ = false;
};

} // namespace nexusloss::self_supervised

#endif
