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

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace nexusloss::survival {

template<typename T, typename Flag>
T cox_ph_loss(std::span<const T> log_risk, std::span<const T> durations,
              std::span<const Flag> observed) {
    if (log_risk.empty() || log_risk.size() != durations.size() || log_risk.size() != observed.size())
        throw std::invalid_argument("NexusLoss: Cox arrays must have equal non-zero sizes");
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
 * @brief Cox kismi olabilirliginin log-risk'e gore turevi.
 *
 * Her gozlenen olay i icin risk kumesindeki k: softmax agirligi kadar artar,
 * i'nin kendisinden 1 duser. Sonuc olay sayisina bolunur (forward ortalamasi).
 * Breslow: ayni suredekiler hepsi risk kumesinde kalir (durations[j] >= durations[i]).
 */
template<typename T, typename Flag>
std::vector<T> cox_ph_gradient(std::span<const T> log_risk, std::span<const T> durations,
                               std::span<const Flag> observed) {
    if (log_risk.empty() || log_risk.size() != durations.size() || log_risk.size() != observed.size())
        throw std::invalid_argument("NexusLoss: Cox arrays must have equal non-zero sizes");
    std::vector<T> grad(log_risk.size(), T(0));
    size_t events = 0;
    for (size_t i = 0; i < log_risk.size(); ++i) if (observed[i]) {
        T maximum = log_risk[i];
        for (size_t j = 0; j < log_risk.size(); ++j)
            if (durations[j] >= durations[i]) maximum = std::max(maximum, log_risk[j]);
        T denominator = T(0);
        for (size_t j = 0; j < log_risk.size(); ++j)
            if (durations[j] >= durations[i]) denominator += std::exp(log_risk[j] - maximum);
        for (size_t j = 0; j < log_risk.size(); ++j)
            if (durations[j] >= durations[i])
                grad[j] += std::exp(log_risk[j] - maximum) / denominator;
        grad[i] -= T(1);
        ++events;
    }
    if (!events) throw std::invalid_argument("NexusLoss: Cox loss requires at least one observed event");
    for (T& value : grad) value /= static_cast<T>(events);
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

/// @brief [d nll / d log_scale, d nll / d log_shape]
template<typename T>
std::vector<T> weibull_nll_gradient(T time, bool event, T log_scale, T log_shape) {
    if (!(time > T(0))) throw std::invalid_argument("NexusLoss: Weibull time must be positive");
    const T shape = std::exp(log_shape);
    const T log_ratio = std::log(time) - log_scale;
    const T z = std::exp(shape * log_ratio);
    const T d_scale = shape * ((event ? T(1) : T(0)) - z);
    const T d_shape = shape * z * log_ratio - (event ? (T(1) + shape * log_ratio) : T(0));
    return {d_scale, d_shape};
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
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return cox_ph_gradient<T>(log_risk_, durations_, std::span<const char>(observed_));
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
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return weibull_nll_gradient(time_, event_, log_scale_, log_shape_);
    }
private:
    T time_ = T(1), log_scale_ = T(0), log_shape_ = T(0);
    bool event_ = true, ready_ = false;
};

} // namespace nexusloss::survival

#endif
