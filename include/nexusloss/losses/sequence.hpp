/**
 * @file sequence.hpp
 * @brief Hizalanmamis ve maskeli dizi kayiplari.
 *
 * CTC logitleri zaman-major duz vektordur: time * classes + class.
 * Hedef blank icermez. Maskeli cross-entropy sadece isaretli token'larin
 * ortalamasini alir; pad konumlarinin gradyani sifirdir.
 *
 * CTC ileri gecisi log-softmax ve alpha tablosunu `CTCWorkspace` icinde
 * birakir. Geri gecis yalnizca beta ve posterioru ekler; ileri gecisin isi
 * tekrar yapilmaz ve tekrar kullanilan tablolar sicak dongude bellek ayirmaz.
 *
 * @see Graves et al., CTC, 2006
 * @see core/utils.hpp
 */
#ifndef NEXUSLOSS_SEQUENCE_HPP
#define NEXUSLOSS_SEQUENCE_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
#include "classification.hpp"

namespace nexusloss::sequence {

/**
 * @brief CTC dinamik programlama tablolari.
 *
 * `log_prob` [time, class], `alpha` ve `beta` [time, state] duzenindedir.
 * `log_likelihood` son ileri gecisin log Z degeridir.
 */
template<typename T>
struct CTCWorkspace {
    std::vector<size_t> extended;
    std::vector<T> log_prob, alpha, beta;
    T log_likelihood = T(0);
    size_t timesteps = 0, classes = 0, states = 0, blank = 0;
};

namespace detail {

template<typename T>
void validate_ctc(std::span<const T> logits, size_t timesteps, size_t classes,
                  std::span<const size_t> labels, size_t blank) {
    if (!timesteps || !classes || logits.size() != timesteps * classes || blank >= classes)
        throw std::invalid_argument("NexusLoss: CTC logits must be flattened time-major [time, class]");
    if (labels.size() > timesteps) throw std::invalid_argument("NexusLoss: CTC target is longer than input");
    for (size_t label : labels)
        if (label >= classes || label == blank) throw std::invalid_argument("NexusLoss: invalid CTC target label");
}

template<typename T>
T log_add(T a, T b) {
    constexpr T negative_infinity = -std::numeric_limits<T>::infinity();
    if (a == negative_infinity) return b;
    if (b == negative_infinity) return a;
    const T maximum = std::max(a, b);
    return maximum + std::log1p(std::exp(std::min(a, b) - maximum));
}

} // namespace detail

/// @brief CTC ileri gecisi. log-softmax ve alpha tablosunu workspace'e yazar, -log Z dondurur.
template<typename T>
T ctc_forward(std::span<const T> logits, size_t timesteps, size_t classes,
              std::span<const size_t> labels, size_t blank, CTCWorkspace<T>& workspace) {
    detail::validate_ctc(logits, timesteps, classes, labels, blank);
    constexpr T negative_infinity = -std::numeric_limits<T>::infinity();
    const size_t states = labels.size() * 2 + 1;
    workspace.timesteps = timesteps;
    workspace.classes = classes;
    workspace.states = states;
    workspace.blank = blank;
    workspace.extended.assign(states, blank);
    for (size_t i = 0; i < labels.size(); ++i) workspace.extended[i * 2 + 1] = labels[i];
    workspace.log_prob.resize(timesteps * classes);
    for (size_t time = 0; time < timesteps; ++time) {
        const auto row = logits.subspan(time * classes, classes);
        core::log_softmax_into<T>(row, std::span<T>(workspace.log_prob).subspan(time * classes, classes));
    }
    const auto& extended = workspace.extended;
    const auto& log_prob = workspace.log_prob;
    auto& alpha = workspace.alpha;
    alpha.assign(timesteps * states, negative_infinity);
    alpha[0] = log_prob[blank];
    if (states > 1) alpha[1] = log_prob[extended[1]];
    for (size_t time = 1; time < timesteps; ++time) {
        for (size_t state = 0; state < states; ++state) {
            T total = alpha[(time - 1) * states + state];
            if (state > 0) total = detail::log_add(total, alpha[(time - 1) * states + state - 1]);
            if (state > 1 && extended[state] != blank && extended[state] != extended[state - 2])
                total = detail::log_add(total, alpha[(time - 1) * states + state - 2]);
            alpha[time * states + state] = total + log_prob[time * classes + extended[state]];
        }
    }
    const size_t last = (timesteps - 1) * states;
    workspace.log_likelihood = states == 1
        ? alpha[last]
        : detail::log_add(alpha[last + states - 1], alpha[last + states - 2]);
    return -workspace.log_likelihood;
}

/**
 * @brief Workspace'teki ileri gecisten CTC turevini yazar (Graves et al., 2006).
 *
 * Bir (t, k) icin posterior, o etiketi basan tum hizalama durumlarinin
 * alpha*beta kutlesidir. Loss -log Z oldugu icin dL/d logit = softmax - posterior.
 * Hedef verilen surede hizalanamiyorsa (Z = 0) turev sifir yazilir.
 */
template<typename T>
void ctc_backward_into(CTCWorkspace<T>& workspace, std::span<T> grad_out) {
    constexpr T negative_infinity = -std::numeric_limits<T>::infinity();
    const size_t timesteps = workspace.timesteps, classes = workspace.classes, states = workspace.states;
    const size_t blank = workspace.blank;
    if (grad_out.size() != timesteps * classes)
        throw std::invalid_argument("NexusLoss: CTC grad_out must match [time, class]");
    if (workspace.log_likelihood == negative_infinity) {
        std::fill(grad_out.begin(), grad_out.end(), T(0));
        return;
    }
    const auto& extended = workspace.extended;
    const auto& log_prob = workspace.log_prob;
    const auto& alpha = workspace.alpha;
    auto& beta = workspace.beta;
    beta.assign(timesteps * states, negative_infinity);
    beta[(timesteps - 1) * states + states - 1] = log_prob[(timesteps - 1) * classes + extended[states - 1]];
    if (states > 1)
        beta[(timesteps - 1) * states + states - 2] = log_prob[(timesteps - 1) * classes + extended[states - 2]];
    for (size_t time = timesteps - 1; time-- > 0;) {
        for (size_t state = 0; state < states; ++state) {
            T total = beta[(time + 1) * states + state];
            if (state + 1 < states) total = detail::log_add(total, beta[(time + 1) * states + state + 1]);
            if (state + 2 < states && extended[state + 2] != blank && extended[state + 2] != extended[state])
                total = detail::log_add(total, beta[(time + 1) * states + state + 2]);
            beta[time * states + state] = log_prob[time * classes + extended[state]] + total;
        }
    }
    std::fill(grad_out.begin(), grad_out.end(), T(0));
    for (size_t time = 0; time < timesteps; ++time)
        for (size_t state = 0; state < states; ++state) {
            const T emission = log_prob[time * classes + extended[state]];
            const T occupancy = alpha[time * states + state] + beta[time * states + state]
                                - emission - workspace.log_likelihood;
            if (occupancy == negative_infinity) continue;
            grad_out[time * classes + extended[state]] += std::exp(occupancy);
        }
    for (size_t index = 0; index < grad_out.size(); ++index)
        grad_out[index] = std::exp(log_prob[index]) - grad_out[index];
}

template<typename T>
T ctc_loss(std::span<const T> logits, size_t timesteps, size_t classes,
           std::span<const size_t> labels, size_t blank = 0) {
    CTCWorkspace<T> workspace;
    return ctc_forward<T>(logits, timesteps, classes, labels, blank, workspace);
}

template<typename T>
void ctc_loss_gradient_into(std::span<const T> logits, size_t timesteps, size_t classes,
                            std::span<const size_t> labels, std::span<T> grad_out,
                            CTCWorkspace<T>& workspace, size_t blank = 0) {
    ctc_forward<T>(logits, timesteps, classes, labels, blank, workspace);
    ctc_backward_into<T>(workspace, grad_out);
}

template<typename T>
std::vector<T> ctc_loss_gradient(std::span<const T> logits, size_t timesteps, size_t classes,
                                 std::span<const size_t> labels, size_t blank = 0) {
    std::vector<T> grad(logits.size());
    CTCWorkspace<T> workspace;
    ctc_loss_gradient_into<T>(logits, timesteps, classes, labels, grad, workspace, blank);
    return grad;
}

namespace detail {

template<typename T>
void validate_masked(size_t logits_size, size_t token_count, size_t classes,
                     std::span<const size_t> masked_positions, std::span<const size_t> masked_targets) {
    if (!token_count || !classes || logits_size != token_count * classes
        || masked_positions.empty() || masked_positions.size() != masked_targets.size())
        throw std::invalid_argument("NexusLoss: MLM expects [token, class] logits and paired masked targets");
    for (size_t i = 0; i < masked_positions.size(); ++i) {
        if (masked_positions[i] >= token_count)
            throw std::out_of_range("NexusLoss: masked token position is out of range");
        if (masked_targets[i] >= classes)
            throw std::out_of_range("NexusLoss: masked target class is out of range");
    }
}

} // namespace detail

template<typename T>
T masked_language_model_loss(std::span<const T> token_logits, size_t token_count, size_t classes,
                             std::span<const size_t> masked_positions,
                             std::span<const size_t> masked_targets) {
    detail::validate_masked<T>(token_logits.size(), token_count, classes, masked_positions, masked_targets);
    T sum = T(0);
    for (size_t i = 0; i < masked_positions.size(); ++i) {
        const auto row = token_logits.subspan(masked_positions[i] * classes, classes);
        sum += core::log_sum_exp<T>(row) - row[masked_targets[i]];
    }
    return sum / static_cast<T>(masked_positions.size());
}

template<typename T>
T next_sentence_prediction_loss(T logit, T is_next) {
    return classification::binary_cross_entropy_with_logits(logit, is_next);
}

/**
 * @brief Maskeli CE turevi, ayirmasiz. Ayni konum iki kez maskelenirse
 *        forward iki terimi topladigi icin turev de toplanir.
 */
template<typename T>
void masked_language_model_gradient_into(std::span<const T> token_logits, size_t token_count, size_t classes,
                                         std::span<const size_t> masked_positions,
                                         std::span<const size_t> masked_targets, std::span<T> grad_out) {
    detail::validate_masked<T>(token_logits.size(), token_count, classes, masked_positions, masked_targets);
    if (grad_out.size() != token_logits.size())
        throw std::invalid_argument("NexusLoss: MLM grad_out must match token logits");
    std::fill(grad_out.begin(), grad_out.end(), T(0));
    const T inv = T(1) / static_cast<T>(masked_positions.size());
    for (size_t i = 0; i < masked_positions.size(); ++i) {
        const size_t offset = masked_positions[i] * classes;
        const auto row = token_logits.subspan(offset, classes);
        const T log_z = core::log_sum_exp<T>(row);
        for (size_t c = 0; c < classes; ++c)
            grad_out[offset + c] += (std::exp(row[c] - log_z) - (c == masked_targets[i] ? T(1) : T(0))) * inv;
    }
}

template<typename T>
std::vector<T> masked_language_model_gradient(std::span<const T> token_logits, size_t token_count, size_t classes,
                                              std::span<const size_t> masked_positions,
                                              std::span<const size_t> masked_targets) {
    std::vector<T> grad(token_logits.size());
    masked_language_model_gradient_into<T>(token_logits, token_count, classes, masked_positions, masked_targets, grad);
    return grad;
}

/**
 * @class CTCLoss
 * @brief Hizalanmamis dizi etiketleme kaybi (konusma, OCR).
 *
 * Graves, Fernandez, Gomez, Schmidhuber, Connectionist Temporal Classification, 2006.
 * Logitler zaman-major duz vektordur: index = time * classes + class.
 * Hedef, blank sinifini icermeyen etiket dizisidir.
 *
 * `zero_infinity` acikken hizalanamayan hedef (Z = 0) +inf yerine 0 kayip ve
 * 0 turev uretir; boylece tek bir bozuk ornek tum batch'i NaN yapmaz.
 */
template<typename T>
class CTCLoss {
public:
    explicit CTCLoss(bool zero_infinity = true) : zero_infinity_(zero_infinity) {}

    T forward(std::span<const T> logits, size_t timesteps, size_t classes,
              std::span<const size_t> labels, size_t blank = 0) {
        const T loss = ctc_forward<T>(logits, timesteps, classes, labels, blank, workspace_);
        workspace_.beta.reserve(timesteps * workspace_.states);
        ready_ = true;
        if (zero_infinity_ && !std::isfinite(loss)) return T(0);
        return loss;
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return workspace_.timesteps * workspace_.classes; }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    /// @note `beta` tablosu nesnenin icinde tutulur; ayni nesneye es zamanli backward cagrilmaz.
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        ctc_backward_into<T>(workspace_, grad_out);
    }
private:
    mutable CTCWorkspace<T> workspace_;
    bool zero_infinity_;
    bool ready_ = false;
};

/**
 * @class MaskedSequenceCrossEntropyLoss
 * @brief Sadece maskelenmis token'larin cross-entropy ortalamasi. Pad hesaba girmez.
 */
template<typename T>
class MaskedSequenceCrossEntropyLoss {
public:
    T forward(std::span<const T> token_logits, size_t token_count, size_t classes,
              std::span<const size_t> masked_positions, std::span<const size_t> masked_targets) {
        logits_.assign(token_logits.begin(), token_logits.end());
        positions_.assign(masked_positions.begin(), masked_positions.end());
        targets_.assign(masked_targets.begin(), masked_targets.end());
        tokens_ = token_count;
        classes_ = classes;
        ready_ = true;
        return masked_language_model_loss(token_logits, token_count, classes, masked_positions, masked_targets);
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return logits_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        masked_language_model_gradient_into<T>(logits_, tokens_, classes_, positions_, targets_, grad_out);
    }
private:
    std::vector<T> logits_;
    std::vector<size_t> positions_, targets_;
    size_t tokens_ = 0, classes_ = 0;
    bool ready_ = false;
};

} // namespace nexusloss::sequence

#endif
