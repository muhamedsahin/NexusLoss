/**
 * @file ranking.hpp
 * @brief Cift ve liste bazli siralama kayiplari.
 *
 * Margin ve RankNet iki skor karsilastirir. ListNet ve ListMLE bir listenin
 * tum skorlarini relevance ile hizalar. Relevance buyuk olanin onde olmasi beklenir.
 *
 * @see Burges et al., RankNet, 2005
 * @see Xia et al., ListMLE, 2008
 */
#ifndef NEXUSLOSS_RANKING_HPP
#define NEXUSLOSS_RANKING_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>
#include "../core/utils.hpp"

namespace nexusloss::ranking {

template<typename T>
T pairwise_logistic(T preferred_score, T other_score) {
    const T difference = preferred_score - other_score;
    return std::max(T(0), -difference) + std::log1p(std::exp(-std::abs(difference)));
}

template<typename T>
T bpr_loss(T positive_score, T negative_score) {
    const T difference = positive_score - negative_score;
    return std::max(T(0), -difference) + std::log1p(std::exp(-std::abs(difference)));
}

template<typename T>
T listnet_loss(std::span<const T> scores, std::span<const T> relevance) {
    if (scores.empty() || scores.size() != relevance.size())
        throw std::invalid_argument("NexusLoss: ListNet inputs must have equal non-zero sizes");
    const auto distribution = [](std::span<const T> values) {
        const T maximum = *std::max_element(values.begin(), values.end());
        std::vector<T> result(values.size());
        T sum = T(0);
        for (size_t i = 0; i < values.size(); ++i) sum += result[i] = std::exp(values[i] - maximum);
        for (T& value : result) value /= sum;
        return result;
    };
    const auto predicted = distribution(scores);
    const auto desired = distribution(relevance);
    T loss = T(0);
    for (size_t i = 0; i < scores.size(); ++i)
        loss -= desired[i] * std::log(std::max(predicted[i], core::epsilon<T>));
    return loss;
}

template<typename T>
T approx_ndcg_loss(std::span<const T> scores, std::span<const T> relevance, T temperature = T(1)) {
    if (scores.empty() || scores.size() != relevance.size() || !(temperature > T(0)))
        throw std::invalid_argument("NexusLoss: invalid ApproxNDCG inputs or temperature");
    const size_t n = scores.size();
    std::vector<T> ranks(n, T(1));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (i != j) ranks[i] += core::sigmoid((scores[j] - scores[i]) / temperature);
    T dcg = T(0), ideal = T(0);
    std::vector<T> sorted_relevance(relevance.begin(), relevance.end());
    std::sort(sorted_relevance.begin(), sorted_relevance.end(), std::greater<T>());
    for (size_t i = 0; i < n; ++i) {
        dcg += (std::pow(T(2), relevance[i]) - T(1)) / std::log2(ranks[i] + T(1));
        ideal += (std::pow(T(2), sorted_relevance[i]) - T(1)) / std::log2(static_cast<T>(i) + T(2));
    }
    return ideal > T(0) ? T(1) - dcg / ideal : T(0);
}

template<typename T>
T soft_rank_loss(std::span<const T> scores, std::span<const T> relevance, T temperature = T(1)) {
    return approx_ndcg_loss(scores, relevance, temperature);
}

template<typename T>
T lambda_rank_loss(std::span<const T> scores, std::span<const T> relevance, T sigma = T(1)) {
    if (scores.empty() || scores.size() != relevance.size() || !(sigma > T(0)))
        throw std::invalid_argument("NexusLoss: invalid LambdaRank inputs or sigma");
    std::vector<T> ideal_relevance(relevance.begin(), relevance.end());
    std::sort(ideal_relevance.begin(), ideal_relevance.end(), std::greater<T>());
    T ideal_dcg = T(0);
    for (size_t i = 0; i < ideal_relevance.size(); ++i)
        ideal_dcg += (std::pow(T(2), ideal_relevance[i]) - T(1))
                     / std::log2(static_cast<T>(i) + T(2));
    if (ideal_dcg == T(0)) return T(0);
    std::vector<T> soft_ranks(scores.size(), T(1));
    for (size_t i = 0; i < scores.size(); ++i) {
        if (relevance[i] < T(0))
            throw std::invalid_argument("NexusLoss: LambdaRank relevance must be non-negative");
        for (size_t j = 0; j < scores.size(); ++j)
            if (i != j) soft_ranks[i] += core::sigmoid((scores[j] - scores[i]) / sigma);
    }
    T weighted_loss = T(0), total_weight = T(0);
    for (size_t i = 0; i < scores.size(); ++i)
        for (size_t j = i + 1; j < scores.size(); ++j) {
            if (relevance[i] == relevance[j]) continue;
            const T direction = relevance[i] > relevance[j] ? T(1) : T(-1);
            const T discount_i = T(1) / std::log2(soft_ranks[i] + T(1));
            const T discount_j = T(1) / std::log2(soft_ranks[j] + T(1));
            const T gain_i = std::pow(T(2), relevance[i]) - T(1);
            const T gain_j = std::pow(T(2), relevance[j]) - T(1);
            const T weight = std::abs((gain_i - gain_j) * (discount_i - discount_j)) / ideal_dcg;
            const T margin = sigma * direction * (scores[i] - scores[j]);
            weighted_loss += weight * (std::max(T(0), -margin) + std::log1p(std::exp(-std::abs(margin))));
            total_weight += weight;
        }
    return total_weight > T(0) ? weighted_loss / total_weight : T(0);
}

/// @brief max(0, -y*(x1-x2) + margin). y = +1 x1'in onde olmasini ister.
template<typename T>
T margin_ranking_loss(T x1, T x2, T target, T margin = T(0)) {
    return std::max(T(0), -target * (x1 - x2) + margin);
}

template<typename T>
std::vector<T> margin_ranking_gradient(T x1, T x2, T target, T margin = T(0)) {
    if (-target * (x1 - x2) + margin <= T(0)) return {T(0), T(0)};
    return {-target, target};
}

/// @brief RankNet cift kaybi. preferred, other'dan yuksek skor almalidir.
template<typename T>
T ranknet_loss(T preferred, T other) { return pairwise_logistic(preferred, other); }

template<typename T>
std::vector<T> ranknet_gradient(T preferred, T other) {
    const T difference = preferred - other;
    const T d_difference = core::stable_sigmoid(difference) - T(1);
    return {d_difference, -d_difference};
}

template<typename T>
std::vector<T> listnet_gradient(std::span<const T> scores, std::span<const T> relevance) {
    if (scores.empty() || scores.size() != relevance.size())
        throw std::invalid_argument("NexusLoss: ListNet inputs must have equal non-zero sizes");
    const auto distribution = [](std::span<const T> values) {
        const T maximum = *std::max_element(values.begin(), values.end());
        std::vector<T> result(values.size());
        T sum = T(0);
        for (size_t i = 0; i < values.size(); ++i) sum += result[i] = std::exp(values[i] - maximum);
        for (T& value : result) value /= sum;
        return result;
    };
    const auto predicted = distribution(scores);
    const auto desired = distribution(relevance);
    std::vector<T> grad(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) grad[i] = predicted[i] - desired[i];
    return grad;
}

/**
 * @brief ListMLE: dogru siralama permütasyonunun negatif log-olabilirligi.
 * Xia et al., Listwise approach to learning to rank, 2008.
 * relevance buyukten kucuge siralanir; es baglarda girdi sirasi korunur.
 */
template<typename T>
T listmle_loss(std::span<const T> scores, std::span<const T> relevance) {
    if (scores.empty() || scores.size() != relevance.size())
        throw std::invalid_argument("NexusLoss: ListMLE inputs must have equal non-zero sizes");
    std::vector<size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return relevance[a] > relevance[b]; });
    T loss = T(0);
    for (size_t start = 0; start < order.size(); ++start) {
        T maximum = scores[order[start]];
        for (size_t i = start + 1; i < order.size(); ++i) maximum = std::max(maximum, scores[order[i]]);
        T sum = T(0);
        for (size_t i = start; i < order.size(); ++i) sum += std::exp(scores[order[i]] - maximum);
        loss += maximum + std::log(sum) - scores[order[start]];
    }
    return loss;
}

template<typename T>
std::vector<T> listmle_gradient(std::span<const T> scores, std::span<const T> relevance) {
    if (scores.empty() || scores.size() != relevance.size())
        throw std::invalid_argument("NexusLoss: ListMLE inputs must have equal non-zero sizes");
    std::vector<size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return relevance[a] > relevance[b]; });
    std::vector<T> grad(scores.size(), T(0));
    for (size_t start = 0; start < order.size(); ++start) {
        T maximum = scores[order[start]];
        for (size_t i = start + 1; i < order.size(); ++i) maximum = std::max(maximum, scores[order[i]]);
        T sum = T(0);
        for (size_t i = start; i < order.size(); ++i) sum += std::exp(scores[order[i]] - maximum);
        for (size_t i = start; i < order.size(); ++i) {
            const T probability = std::exp(scores[order[i]] - maximum) / sum;
            grad[order[i]] += probability;
        }
        grad[order[start]] -= T(1);
    }
    return grad;
}

template<typename T>
class MarginRankingLoss {
public:
    explicit MarginRankingLoss(T margin = T(0)) : margin_(margin) {}
    T forward(T x1, T x2, T target) {
        x1_ = x1; x2_ = x2; target_ = target; ready_ = true;
        return margin_ranking_loss(x1, x2, target, margin_);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return margin_ranking_gradient(x1_, x2_, target_, margin_);
    }
private:
    T margin_, x1_ = T(0), x2_ = T(0), target_ = T(1);
    bool ready_ = false;
};

template<typename T>
class RankNetLoss {
public:
    T forward(T preferred, T other) {
        preferred_ = preferred; other_ = other; ready_ = true;
        return ranknet_loss(preferred, other);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return ranknet_gradient(preferred_, other_);
    }
private:
    T preferred_ = T(0), other_ = T(0);
    bool ready_ = false;
};

template<typename T>
class ListNetLoss {
public:
    T forward(std::span<const T> scores, std::span<const T> relevance) {
        scores_.assign(scores.begin(), scores.end());
        relevance_.assign(relevance.begin(), relevance.end());
        ready_ = true;
        return listnet_loss(scores, relevance);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return listnet_gradient<T>(scores_, relevance_);
    }
private:
    std::vector<T> scores_, relevance_;
    bool ready_ = false;
};

template<typename T>
class ListMLELoss {
public:
    T forward(std::span<const T> scores, std::span<const T> relevance) {
        scores_.assign(scores.begin(), scores.end());
        relevance_.assign(relevance.begin(), relevance.end());
        ready_ = true;
        return listmle_loss(scores, relevance);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return listmle_gradient<T>(scores_, relevance_);
    }
private:
    std::vector<T> scores_, relevance_;
    bool ready_ = false;
};

} // namespace nexusloss::ranking

#endif
