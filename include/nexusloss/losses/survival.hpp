/**
 * @file survival.hpp
 * @brief Sag-kalim modelleri: Cox kismi olabilirlik ve Weibull NLL.
 *
 * Cox girdisi log-risk, sure ve gozlem bayragidir. Breslow risk kumesi
 * `sure >= olay suresi` olanlari tutar. Weibull, log-olcek ve log-sekil
 * parametrelerinde turevlenir; sansurlu gozlem yalnizca sag-kalim terimini tutar.
 *
 * @see Cox, Regression Models and Life-Tables, 1972
 */
#ifndef NEXUSLOSS_SURVIVAL_HPP
#define NEXUSLOSS_SURVIVAL_HPP

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace nexusloss::survival {

template<typename T, typename Flag>
void validate_cox(std::span<const T> log_risk, std::span<const T> durations, std::span<const Flag> observed) {
    if (log_risk.empty() || log_risk.size() != durations.size() || log_risk.size() != observed.size())
        throw std::invalid_argument("NexusLoss: Cox arrays must have equal non-zero sizes");
}

template<typename T, typename Flag>
T cox_ph_loss(std::span<const T> log_risk, std::span<const T> durations,
              std::span<const Flag> observed) {
    validate_cox(log_risk, durations, observed);
    T loss = T(0);
    size_t events = 0;
    for (size_t i = 0; i < log_risk.size(); ++i) if (observed[i]) {
        T maximum = log_risk[i];
        for (size_t j = 0; j < log_risk.size(); ++j)
            if (durations[j] >= durations[i]) maximum = std::max(maximum, log_risk[j]);
        T denominator = T(0);
        for (size_t j = 0; j < log_risk.size(); ++j)
            if (durations[j] >= durations[i]) denominator += std::exp(log_risk[j] - maximum);
        loss += -(log_risk[i] - maximum - std::log(denominator));
        ++events;
    }
    if (!events) throw std::invalid_argument("NexusLoss: Cox loss requires at least one observed event");
    return loss / static_cast<T>(events);
}

/**
 * @brief Cox kismi olabilirliginin log-risk'e gore turevi, ayirmasiz.
 *
 * Her gozlenen olay i icin risk kumesindeki k: softmax agirligi kadar artar,
 * i'nin kendisinden 1 duser. Sonuc olay sayisina bolunur (forward ortalamasi).
 * Breslow: ayni suredekiler hepsi risk kumesinde kalir (durations[j] >= durations[i]).
 */
template<typename T, typename Flag>
void cox_ph_gradient_into(std::span<const T> log_risk, std::span<const T> durations,
                          std::span<const Flag> observed, std::span<T> grad_out) {
    validate_cox(log_risk, durations, observed);
    if (grad_out.size() != log_risk.size())
        throw std::invalid_argument("NexusLoss: Cox grad_out must match log_risk");
    std::fill(grad_out.begin(), grad_out.end(), T(0));
    size_t events = 0;
    for (size_t i = 0; i < log_risk.size(); ++i) if (observed[i]) {
        T maximum = log_risk[i];
        for (size_t j = 0; j < log_risk.size(); ++j)
            if (durations[j] >= durations[i]) maximum = std::max(maximum, log_risk[j]);
        T denominator = T(0);
        for (size_t j = 0; j < log_risk.size(); ++j)
            if (durations[j] >= durations[i]) denominator += std::exp(log_risk[j] - maximum);
        const T inverse = T(1) / denominator;
        for (size_t j = 0; j < log_risk.size(); ++j)
            if (durations[j] >= durations[i])
                grad_out[j] += std::exp(log_risk[j] - maximum) * inverse;
        grad_out[i] -= T(1);
        ++events;
    }
    if (!events) throw std::invalid_argument("NexusLoss: Cox loss requires at least one observed event");
    const T scale = T(1) / static_cast<T>(events);
    for (T& value : grad_out) value *= scale;
}

template<typename T, typename Flag>
std::vector<T> cox_ph_gradient(std::span<const T> log_risk, std::span<const T> durations,
                               std::span<const Flag> observed) {
    std::vector<T> grad(log_risk.size());
    cox_ph_gradient_into<T, Flag>(log_risk, durations, observed, grad);
    return grad;
}

/**
 * @brief Weibull negatif log-olabilirlik.
 *
 * log_scale = log(lambda), log_shape = log(k), time > 0.
 *   z = (t/lambda)^k
 *   nll = z - event * (log k - k log lambda + (k-1) log t)
 * Sansurlu gozlem sadece sag-kalim terimini (z) tutar.
 */
template<typename T>
T weibull_nll(T time, bool event, T log_scale, T log_shape) {
    if (!(time > T(0))) throw std::invalid_argument("NexusLoss: Weibull time must be positive");
    const T shape = std::exp(log_shape);
    const T log_time = std::log(time);
    const T z = std::exp(shape * (log_time - log_scale));
    const T event_term = log_shape - shape * log_scale + (shape - T(1)) * log_time;
    return z - (event ? event_term : T(0));
}

/// @brief grad_out = [d nll / d log_scale, d nll / d log_shape]
template<typename T>
void weibull_nll_gradient_into(T time, bool event, T log_scale, T log_shape, std::span<T> grad_out) {
    if (!(time > T(0))) throw std::invalid_argument("NexusLoss: Weibull time must be positive");
    if (grad_out.size() != 2) throw std::invalid_argument("NexusLoss: Weibull grad_out must have two entries");
    const T shape = std::exp(log_shape);
    const T log_ratio = std::log(time) - log_scale;
    const T z = std::exp(shape * log_ratio);
    grad_out[0] = shape * ((event ? T(1) : T(0)) - z);
    grad_out[1] = shape * z * log_ratio - (event ? (T(1) + shape * log_ratio) : T(0));
}

template<typename T>
std::vector<T> weibull_nll_gradient(T time, bool event, T log_scale, T log_shape) {
    std::vector<T> grad(2);
    weibull_nll_gradient_into<T>(time, event, log_scale, log_shape, grad);
    return grad;
}

template<typename T>
class CoxPHLoss {
public:
    T forward(std::span<const T> log_risk, std::span<const T> durations, std::span<const bool> observed) {
        log_risk_.assign(log_risk.begin(), log_risk.end());
        durations_.assign(durations.begin(), durations.end());
        observed_.resize(observed.size());
        for (size_t i = 0; i < observed.size(); ++i) observed_[i] = observed[i] ? char(1) : char(0);
        ready_ = true;
        return cox_ph_loss(log_risk, durations, observed);
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return log_risk_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        cox_ph_gradient_into<T, char>(log_risk_, durations_, observed_, grad_out);
    }
private:
    std::vector<T> log_risk_, durations_;
    std::vector<char> observed_;
    bool ready_ = false;
};

template<typename T>
class WeibullNLLLoss {
public:
    T forward(T time, bool event, T log_scale, T log_shape) {
        time_ = time; event_ = event; log_scale_ = log_scale; log_shape_ = log_shape; ready_ = true;
        return weibull_nll(time, event, log_scale, log_shape);
    }
    [[nodiscard]] static constexpr size_t gradient_size() noexcept { return 2; }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        weibull_nll_gradient_into<T>(time_, event_, log_scale_, log_shape_, grad_out);
    }
private:
    T time_ = T(1), log_scale_ = T(0), log_shape_ = T(0);
    bool event_ = true, ready_ = false;
};

} // namespace nexusloss::survival

#endif
