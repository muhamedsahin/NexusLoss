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

namespace detail {

template<typename T>
void require_pair_gradient(std::span<T> grad_out) {
    if (grad_out.size() != 2) throw std::invalid_argument("NexusLoss: pairwise grad_out must have two entries");
}

template<typename T>
void validate_list(std::span<const T> scores, std::span<const T> relevance, std::span<T> grad_out) {
    if (scores.empty() || scores.size() != relevance.size())
        throw std::invalid_argument("NexusLoss: listwise inputs must have equal non-zero sizes");
    if (grad_out.size() != scores.size())
        throw std::invalid_argument("NexusLoss: listwise grad_out must match scores");
}

} // namespace detail

/// @brief grad_out = [dL/dx1, dL/dx2]. Marjin disinda sifir.
template<typename T>
void margin_ranking_gradient_into(T x1, T x2, T target, std::span<T> grad_out, T margin = T(0)) {
    detail::require_pair_gradient(grad_out);
    const bool active = -target * (x1 - x2) + margin > T(0);
    grad_out[0] = active ? -target : T(0);
    grad_out[1] = active ? target : T(0);
}

template<typename T>
std::vector<T> margin_ranking_gradient(T x1, T x2, T target, T margin = T(0)) {
    std::vector<T> grad(2);
    margin_ranking_gradient_into<T>(x1, x2, target, grad, margin);
    return grad;
}

/// @brief RankNet cift kaybi. preferred, other'dan yuksek skor almalidir.
template<typename T>
T ranknet_loss(T preferred, T other) { return pairwise_logistic(preferred, other); }

template<typename T>
void ranknet_gradient_into(T preferred, T other, std::span<T> grad_out) {
    detail::require_pair_gradient(grad_out);
    const T d_difference = core::stable_sigmoid(preferred - other) - T(1);
    grad_out[0] = d_difference;
    grad_out[1] = -d_difference;
}

template<typename T>
std::vector<T> ranknet_gradient(T preferred, T other) {
    std::vector<T> grad(2);
    ranknet_gradient_into<T>(preferred, other, grad);
    return grad;
}

/**
 * @brief ListNet turevi softmax(skor) - softmax(relevance). `relevance_scratch`
 *        hedef dagilim icin cagiranin tamponudur; uzunlugu skorla aynidir.
 */
template<typename T>
void listnet_gradient_into(std::span<const T> scores, std::span<const T> relevance, std::span<T> grad_out,
                           std::span<T> relevance_scratch) {
    detail::validate_list(scores, relevance, grad_out);
    if (relevance_scratch.size() != scores.size())
        throw std::invalid_argument("NexusLoss: ListNet scratch must match scores");
    core::softmax_into<T>(scores, grad_out);
    core::softmax_into<T>(relevance, relevance_scratch);
    for (size_t i = 0; i < scores.size(); ++i) grad_out[i] -= relevance_scratch[i];
}

template<typename T>
std::vector<T> listnet_gradient(std::span<const T> scores, std::span<const T> relevance) {
    std::vector<T> grad(scores.size()), scratch(scores.size());
    listnet_gradient_into<T>(scores, relevance, grad, scratch);
    return grad;
}

/// @brief relevance'a gore azalan, es baglarda kararli siralama. `order` cagiranin tamponudur.
template<typename T>
void listmle_order(std::span<const T> relevance, std::span<size_t> order) {
    if (order.size() != relevance.size())
        throw std::invalid_argument("NexusLoss: ListMLE order buffer must match relevance");
    std::iota(order.begin(), order.end(), size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return relevance[a] > relevance[b]; });
}

namespace detail {

template<typename T>
T log_add(T a, T b) {
    const T maximum = std::max(a, b);
    return maximum + std::log1p(std::exp(std::min(a, b) - maximum));
}

/// lse_k = log sum_{j>=k} exp(s_{o_j}); sondan basa tek gecis.
template<typename T>
void listmle_suffix(std::span<const T> scores, std::span<const size_t> order, std::span<T> suffix_lse) {
    const size_t n = scores.size();
    suffix_lse[n - 1] = scores[order[n - 1]];
    for (size_t k = n - 1; k-- > 0;) suffix_lse[k] = log_add(scores[order[k]], suffix_lse[k + 1]);
}

} // namespace detail

/// @brief Siralama ve suffix tablosu hazirken ListMLE degeri: sum_k (lse_k - s_{o_k}).
template<typename T>
T listmle_loss_from_order(std::span<const T> scores, std::span<const size_t> order, std::span<T> suffix_lse) {
    const size_t n = scores.size();
    if (!n || order.size() != n || suffix_lse.size() != n)
        throw std::invalid_argument("NexusLoss: ListMLE buffers must match scores");
    detail::listmle_suffix(scores, order, suffix_lse);
    T loss = T(0);
    for (size_t k = 0; k < n; ++k) loss += suffix_lse[k] - scores[order[k]];
    return loss;
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
    std::vector<T> suffix(scores.size());
    listmle_order<T>(relevance, order);
    return listmle_loss_from_order<T>(scores, order, suffix);
}

/**
 * @brief Siralama sabitken ListMLE turevi, O(n).
 *
 * Plackett-Luce adimi k'nin paydasi lse_k. o_i konumunun turevi
 * sum_{k<=i} exp(s_{o_i} - lse_k) - 1'dir. Toplam log-uzayda on-ek olarak
 * tutulur: L_i = logaddexp(L_{i-1}, -lse_i). Her k <= i icin s_{o_i} <= lse_k
 * oldugundan s_{o_i} + L_i <= log(i+1); ussel tasmaz.
 */
template<typename T>
void listmle_gradient_from_order(std::span<const T> scores, std::span<const size_t> order,
                                 std::span<T> grad_out, std::span<T> suffix_lse) {
    const size_t n = scores.size();
    if (!n || order.size() != n || grad_out.size() != n || suffix_lse.size() != n)
        throw std::invalid_argument("NexusLoss: ListMLE buffers must match scores");
    detail::listmle_suffix(scores, order, suffix_lse);
    T prefix = -suffix_lse[0];
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) prefix = detail::log_add(prefix, -suffix_lse[i]);
        grad_out[order[i]] = std::exp(scores[order[i]] + prefix) - T(1);
    }
}

template<typename T>
std::vector<T> listmle_gradient(std::span<const T> scores, std::span<const T> relevance) {
    if (scores.empty() || scores.size() != relevance.size())
        throw std::invalid_argument("NexusLoss: ListMLE inputs must have equal non-zero sizes");
    std::vector<size_t> order(scores.size());
    std::vector<T> grad(scores.size()), suffix(scores.size());
    listmle_order<T>(relevance, order);
    listmle_gradient_from_order<T>(scores, order, grad, suffix);
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
    [[nodiscard]] static constexpr size_t gradient_size() noexcept { return 2; }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        margin_ranking_gradient_into<T>(x1_, x2_, target_, grad_out, margin_);
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
    [[nodiscard]] static constexpr size_t gradient_size() noexcept { return 2; }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        ranknet_gradient_into<T>(preferred_, other_, grad_out);
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
        scratch_.resize(scores.size());
        ready_ = true;
        return listnet_loss(scores, relevance);
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return scores_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    /// @note Hedef dagilim nesnenin `scratch_` tamponuna yazilir; es zamanli backward cagrilmaz.
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        listnet_gradient_into<T>(scores_, relevance_, grad_out, scratch_);
    }
private:
    std::vector<T> scores_, relevance_;
    mutable std::vector<T> scratch_;
    bool ready_ = false;
};

/// @note Siralama forward'da bir kez yapilir; backward_into ayni permutasyonu kullanir.
template<typename T>
class ListMLELoss {
public:
    T forward(std::span<const T> scores, std::span<const T> relevance) {
        if (scores.empty() || scores.size() != relevance.size())
            throw std::invalid_argument("NexusLoss: ListMLE inputs must have equal non-zero sizes");
        scores_.assign(scores.begin(), scores.end());
        order_.resize(scores.size());
        suffix_.resize(scores.size());
        listmle_order<T>(relevance, order_);
        ready_ = true;
        return listmle_loss_from_order<T>(scores_, order_, suffix_);
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return scores_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        listmle_gradient_from_order<T>(scores_, order_, grad_out, suffix_);
    }
private:
    std::vector<T> scores_;
    std::vector<size_t> order_;
    mutable std::vector<T> suffix_;
    bool ready_ = false;
};

} // namespace nexusloss::ranking

#endif

