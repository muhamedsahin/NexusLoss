/**
 * @file reinforcement.hpp
 * @brief Politika gradyani, PPO kirpmasi, deger kaybi ve entropi bonusu.
 *
 * Policy ve PPO, log-olasilik uzerinde turev uretir; deger kaybi tahmin edilen
 * getirinin MSE'sidir. EntropyBonusLoss minimize edilen terim olarak -H doner,
 * boylece entropi buyudukce kayip duser.
 *
 * @see Williams, REINFORCE, 1992
 * @see Schulman et al., PPO, 2017
 */
#ifndef NEXUSLOSS_REINFORCEMENT_HPP
#define NEXUSLOSS_REINFORCEMENT_HPP

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>
#include "../core/utils.hpp"

namespace nexusloss::reinforcement {

template<typename T>
T policy_gradient_loss(std::span<const T> log_probabilities, std::span<const T> advantages) {
    if (log_probabilities.empty() || log_probabilities.size() != advantages.size())
        throw std::invalid_argument("NexusLoss: policy and advantage arrays must have equal non-zero size");
    T sum = T(0);
    for (size_t i = 0; i < log_probabilities.size(); ++i) sum -= log_probabilities[i] * advantages[i];
    return sum / static_cast<T>(log_probabilities.size());
}

template<typename T>
T value_loss(std::span<const T> values, std::span<const T> returns) {
    if (values.empty() || values.size() != returns.size())
        throw std::invalid_argument("NexusLoss: value and return arrays must have equal non-zero size");
    T sum = T(0);
    for (size_t i = 0; i < values.size(); ++i) { const T d = values[i] - returns[i]; sum += d * d; }
    return sum / static_cast<T>(values.size());
}

template<typename T>
T ppo_clipped_surrogate_loss(std::span<const T> new_log_probabilities,
                             std::span<const T> old_log_probabilities,
                             std::span<const T> advantages, T clip = T(0.2)) {
    if (new_log_probabilities.empty() || new_log_probabilities.size() != old_log_probabilities.size()
        || new_log_probabilities.size() != advantages.size() || clip < T(0) || clip >= T(1))
        throw std::invalid_argument("NexusLoss: invalid PPO inputs or clipping range");
    T sum = T(0);
    for (size_t i = 0; i < advantages.size(); ++i) {
        const T ratio = std::exp(new_log_probabilities[i] - old_log_probabilities[i]);
        const T clipped = core::clamp(ratio, T(1) - clip, T(1) + clip);
        sum -= std::min(ratio * advantages[i], clipped * advantages[i]);
    }
    return sum / static_cast<T>(advantages.size());
}

template<typename T>
T entropy_bonus(std::span<const T> probabilities) {
    if (probabilities.empty()) throw std::invalid_argument("NexusLoss: entropy input cannot be empty");
    T entropy = T(0);
    for (T p : probabilities) {
        if (p < T(0)) throw std::invalid_argument("NexusLoss: probabilities must be non-negative");
        if (p > T(0)) entropy -= p * std::log(p);
    }
    return entropy;
}

/// @brief Minimize edilen terim -H(pi). Entropi buyudukce kayip kuculur (kesif tesviki).
template<typename T>
T entropy_bonus_loss(std::span<const T> probabilities) { return -entropy_bonus(probabilities); }

template<typename T>
std::vector<T> policy_gradient(std::span<const T> log_probabilities, std::span<const T> advantages) {
    if (log_probabilities.empty() || log_probabilities.size() != advantages.size())
        throw std::invalid_argument("NexusLoss: policy and advantage arrays must have equal non-zero size");
    std::vector<T> grad(log_probabilities.size());
    const T inv = T(1) / static_cast<T>(log_probabilities.size());
    for (size_t i = 0; i < grad.size(); ++i) grad[i] = -advantages[i] * inv;
    return grad;
}

template<typename T>
std::vector<T> value_loss_gradient(std::span<const T> values, std::span<const T> returns) {
    if (values.empty() || values.size() != returns.size())
        throw std::invalid_argument("NexusLoss: value and return arrays must have equal non-zero size");
    std::vector<T> grad(values.size());
    const T scale = T(2) / static_cast<T>(values.size());
    for (size_t i = 0; i < values.size(); ++i) grad[i] = scale * (values[i] - returns[i]);
    return grad;
}

/// @brief dL / d(new log-prob). Kirpilmis oran secildiyse o adimda turev 0'dir.
template<typename T>
std::vector<T> ppo_clipped_surrogate_gradient(std::span<const T> new_log_probabilities,
                                              std::span<const T> old_log_probabilities,
                                              std::span<const T> advantages, T clip = T(0.2)) {
    if (new_log_probabilities.empty() || new_log_probabilities.size() != old_log_probabilities.size()
        || new_log_probabilities.size() != advantages.size() || clip < T(0) || clip >= T(1))
        throw std::invalid_argument("NexusLoss: invalid PPO inputs or clipping range");
    std::vector<T> grad(advantages.size());
    const T inv = T(1) / static_cast<T>(advantages.size());
    for (size_t i = 0; i < advantages.size(); ++i) {
        const T ratio = std::exp(new_log_probabilities[i] - old_log_probabilities[i]);
        const T unclipped = ratio * advantages[i];
        const T clipped_ratio = core::clamp(ratio, T(1) - clip, T(1) + clip);
        const T clipped = clipped_ratio * advantages[i];
        const bool ratio_inside = ratio > T(1) - clip && ratio < T(1) + clip;
        if (unclipped < clipped || (unclipped == clipped && ratio_inside)) grad[i] = -advantages[i] * ratio * inv;
        else if (unclipped == clipped) grad[i] = T(0);
        else grad[i] = T(0);
    }
    return grad;
}

template<typename T>
std::vector<T> entropy_bonus_gradient(std::span<const T> probabilities) {
    if (probabilities.empty()) throw std::invalid_argument("NexusLoss: entropy input cannot be empty");
    std::vector<T> grad(probabilities.size());
    for (size_t i = 0; i < probabilities.size(); ++i) {
        if (probabilities[i] < T(0)) throw std::invalid_argument("NexusLoss: probabilities must be non-negative");
        grad[i] = probabilities[i] > T(0) ? std::log(probabilities[i]) + T(1) : T(0);
    }
    return grad;
}

template<typename T>
class PolicyGradientLoss {
public:
    T forward(std::span<const T> log_probabilities, std::span<const T> advantages) {
        logs_.assign(log_probabilities.begin(), log_probabilities.end());
        advantages_.assign(advantages.begin(), advantages.end());
        ready_ = true;
        return policy_gradient_loss(log_probabilities, advantages);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return policy_gradient<T>(logs_, advantages_);
    }
private:
    std::vector<T> logs_, advantages_;
    bool ready_ = false;
};

template<typename T>
class PPOClipLoss {
public:
    explicit PPOClipLoss(T clip = T(0.2)) : clip_(clip) {}
    T forward(std::span<const T> new_logs, std::span<const T> old_logs, std::span<const T> advantages) {
        new_.assign(new_logs.begin(), new_logs.end());
        old_.assign(old_logs.begin(), old_logs.end());
        advantages_.assign(advantages.begin(), advantages.end());
        ready_ = true;
        return ppo_clipped_surrogate_loss(new_logs, old_logs, advantages, clip_);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return ppo_clipped_surrogate_gradient<T>(new_, old_, advantages_, clip_);
    }
private:
    T clip_;
    std::vector<T> new_, old_, advantages_;
    bool ready_ = false;
};

template<typename T>
class ValueLoss {
public:
    T forward(std::span<const T> values, std::span<const T> returns) {
        values_.assign(values.begin(), values.end());
        returns_.assign(returns.begin(), returns.end());
        ready_ = true;
        return value_loss(values, returns);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return value_loss_gradient<T>(values_, returns_);
    }
private:
    std::vector<T> values_, returns_;
    bool ready_ = false;
};

template<typename T>
class EntropyBonusLoss {
public:
    T forward(std::span<const T> probabilities) {
        probabilities_.assign(probabilities.begin(), probabilities.end());
        ready_ = true;
        return entropy_bonus_loss(probabilities);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return entropy_bonus_gradient<T>(probabilities_);
    }
private:
    std::vector<T> probabilities_;
    bool ready_ = false;
};

} // namespace nexusloss::reinforcement

#endif
