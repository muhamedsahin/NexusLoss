/**
 * @file sequence.hpp
 * @brief Hizalanmamis ve maskeli dizi kayiplari.
 *
 * CTC logitleri zaman-major duz vektordur: time * classes + class.
 * Hedef blank icermez. Maskeli cross-entropy sadece isaretli token'larin
 * ortalamasini alir; pad konumlarinin gradyani sifirdir.
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

template<typename T>
T ctc_loss(std::span<const T> logits, size_t timesteps, size_t classes,
           std::span<const size_t> labels, size_t blank = 0) {
    if (!timesteps || !classes || logits.size() != timesteps * classes || blank >= classes)
        throw std::invalid_argument("NexusLoss: CTC logits must be flattened time-major [time, class]");
    if (labels.size() > timesteps) throw std::invalid_argument("NexusLoss: CTC target is longer than input");
    for (size_t label : labels)
        if (label >= classes || label == blank) throw std::invalid_argument("NexusLoss: invalid CTC target label");
    const T negative_infinity = -std::numeric_limits<T>::infinity();
    const size_t states = labels.size() * 2 + 1;
    std::vector<size_t> extended(states, blank);
    for (size_t i = 0; i < labels.size(); ++i) extended[i * 2 + 1] = labels[i];
    std::vector<T> previous(states, negative_infinity), current(states, negative_infinity);
    const auto log_probability = [&](size_t time, size_t klass) {
        T maximum = logits[time * classes];
        for (size_t c = 1; c < classes; ++c) maximum = std::max(maximum, logits[time * classes + c]);
        T sum = T(0);
        for (size_t c = 0; c < classes; ++c) sum += std::exp(logits[time * classes + c] - maximum);
        return logits[time * classes + klass] - maximum - std::log(sum);
    };
    const auto log_add = [negative_infinity](T a, T b) {
        if (a == negative_infinity) return b;
        if (b == negative_infinity) return a;
        const T maximum = std::max(a, b);
        return maximum + std::log1p(std::exp(std::min(a, b) - maximum));
    };
    previous[0] = log_probability(0, blank);
    if (states > 1) previous[1] = log_probability(0, extended[1]);
    for (size_t t = 1; t < timesteps; ++t) {
        std::fill(current.begin(), current.end(), negative_infinity);
        for (size_t s = 0; s < states; ++s) {
            T total = previous[s];
            if (s > 0) total = log_add(total, previous[s - 1]);
            if (s > 1 && extended[s] != blank && extended[s] != extended[s - 2])
                total = log_add(total, previous[s - 2]);
            current[s] = total + log_probability(t, extended[s]);
        }
        previous.swap(current);
    }
    const T log_likelihood = states == 1 ? previous[0] : log_add(previous[states - 1], previous[states - 2]);
    return -log_likelihood;
}

template<typename T>
T masked_language_model_loss(std::span<const T> token_logits, size_t token_count, size_t classes,
                             std::span<const size_t> masked_positions,
                             std::span<const size_t> masked_targets) {
    if (!token_count || !classes || token_logits.size() != token_count * classes
        || masked_positions.empty() || masked_positions.size() != masked_targets.size())
        throw std::invalid_argument("NexusLoss: MLM expects [token, class] logits and paired masked targets");
    T sum = T(0);
    for (size_t i = 0; i < masked_positions.size(); ++i) {
        if (masked_positions[i] >= token_count)
            throw std::out_of_range("NexusLoss: masked token position is out of range");
        const auto row = token_logits.subspan(masked_positions[i] * classes, classes);
        sum += classification::sparse_categorical_cross_entropy<T>(row, masked_targets[i]);
    }
    return sum / static_cast<T>(masked_positions.size());
}

template<typename T>
T next_sentence_prediction_loss(T logit, T is_next) {
    return classification::binary_cross_entropy_with_logits(logit, is_next);
}

/**
 * @brief CTC kaybinin logitlere gore analitik turevi (Graves et al., 2006).
 *
 * Ileri-geri dinamik programlama log-uzayda alpha ve beta uretir. Bir (t, k)
 * icin posterior, o etiketi basan tum hizalama durumlarinin alpha*beta kutlesidir.
 * Loss -log Z oldugu icin dL/d logit = softmax(logit) - posterior.
 * Bos hedefte tek durum blank'tir.
 */
template<typename T>
std::vector<T> ctc_loss_gradient(std::span<const T> logits, size_t timesteps, size_t classes,
                                 std::span<const size_t> labels, size_t blank = 0) {
    if (!timesteps || !classes || logits.size() != timesteps * classes || blank >= classes)
        throw std::invalid_argument("NexusLoss: CTC logits must be flattened time-major [time, class]");
    if (labels.size() > timesteps) throw std::invalid_argument("NexusLoss: CTC target is longer than input");
    for (size_t label : labels)
        if (label >= classes || label == blank) throw std::invalid_argument("NexusLoss: invalid CTC target label");
    const T negative_infinity = -std::numeric_limits<T>::infinity();
    const size_t states = labels.size() * 2 + 1;
    std::vector<size_t> extended(states, blank);
    for (size_t i = 0; i < labels.size(); ++i) extended[i * 2 + 1] = labels[i];

    std::vector<T> log_prob(timesteps * classes);
    for (size_t time = 0; time < timesteps; ++time) {
        T maximum = logits[time * classes];
        for (size_t c = 1; c < classes; ++c) maximum = std::max(maximum, logits[time * classes + c]);
        T sum = T(0);
        for (size_t c = 0; c < classes; ++c) sum += std::exp(logits[time * classes + c] - maximum);
        const T log_z = maximum + std::log(sum);
        for (size_t c = 0; c < classes; ++c)
            log_prob[time * classes + c] = logits[time * classes + c] - log_z;
    }
    const auto log_add = [negative_infinity](T a, T b) {
        if (a == negative_infinity) return b;
        if (b == negative_infinity) return a;
        const T maximum = std::max(a, b);
        return maximum + std::log1p(std::exp(std::min(a, b) - maximum));
    };
    std::vector<T> alpha(timesteps * states, negative_infinity), beta(timesteps * states, negative_infinity);
    alpha[0] = log_prob[blank];
    if (states > 1) alpha[1] = log_prob[extended[1]];
    for (size_t time = 1; time < timesteps; ++time) {
        for (size_t state = 0; state < states; ++state) {
            T total = alpha[(time - 1) * states + state];
            if (state > 0) total = log_add(total, alpha[(time - 1) * states + state - 1]);
            if (state > 1 && extended[state] != blank && extended[state] != extended[state - 2])
                total = log_add(total, alpha[(time - 1) * states + state - 2]);
            alpha[time * states + state] = total + log_prob[time * classes + extended[state]];
        }
    }
    beta[(timesteps - 1) * states + states - 1] = log_prob[(timesteps - 1) * classes + extended[states - 1]];
    if (states > 1)
        beta[(timesteps - 1) * states + states - 2] = log_prob[(timesteps - 1) * classes + extended[states - 2]];
    for (size_t time = timesteps - 1; time-- > 0;) {
        for (size_t state = 0; state < states; ++state) {
            T total = beta[(time + 1) * states + state];
            if (state + 1 < states) total = log_add(total, beta[(time + 1) * states + state + 1]);
            if (state + 2 < states && extended[state + 2] != blank && extended[state + 2] != extended[state])
                total = log_add(total, beta[(time + 1) * states + state + 2]);
            beta[time * states + state] = log_prob[time * classes + extended[state]] + total;
        }
    }
    const T log_likelihood = states == 1
        ? alpha[(timesteps - 1) * states]
        : log_add(alpha[(timesteps - 1) * states + states - 1], alpha[(timesteps - 1) * states + states - 2]);
    std::vector<T> posterior(logits.size(), T(0));
    for (size_t time = 0; time < timesteps; ++time)
        for (size_t state = 0; state < states; ++state) {
            const T emission = log_prob[time * classes + extended[state]];
            const T occupancy = alpha[time * states + state] + beta[time * states + state] - emission - log_likelihood;
            if (occupancy == negative_infinity) continue;
            posterior[time * classes + extended[state]] += std::exp(occupancy);
        }
    std::vector<T> grad(logits.size());
    for (size_t time = 0; time < timesteps; ++time)
        for (size_t klass = 0; klass < classes; ++klass)
            grad[time * classes + klass] = std::exp(log_prob[time * classes + klass]) - posterior[time * classes + klass];
    return grad;
}

template<typename T>
std::vector<T> masked_language_model_gradient(std::span<const T> token_logits, size_t token_count, size_t classes,
                                              std::span<const size_t> masked_positions,
                                              std::span<const size_t> masked_targets) {
    if (!token_count || !classes || token_logits.size() != token_count * classes
        || masked_positions.empty() || masked_positions.size() != masked_targets.size())
        throw std::invalid_argument("NexusLoss: MLM expects [token, class] logits and paired masked targets");
    std::vector<T> grad(token_logits.size(), T(0));
    const T inv = T(1) / static_cast<T>(masked_positions.size());
    for (size_t i = 0; i < masked_positions.size(); ++i) {
        if (masked_positions[i] >= token_count)
            throw std::out_of_range("NexusLoss: masked token position is out of range");
        const auto row = token_logits.subspan(masked_positions[i] * classes, classes);
        std::vector<T> one_hot(classes, T(0));
        if (masked_targets[i] >= classes) throw std::out_of_range("NexusLoss: masked target class is out of range");
        one_hot[masked_targets[i]] = T(1);
        const auto local = classification::categorical_cross_entropy_gradient<T>(row, one_hot);
        for (size_t c = 0; c < classes; ++c)
            grad[masked_positions[i] * classes + c] = local[c] * inv;
    }
    return grad;
}

/**
 * @class CTCLoss
 * @brief Hizalanmamis dizi etiketleme kaybi (konusma, OCR).
 *
 * Graves, Fernandez, Gomez, Schmidhuber, Connectionist Temporal Classification, 2006.
 * Logitler zaman-major duz vektordur: index = time * classes + class.
 * Hedef, blank sinifini icermeyen etiket dizisidir.
 */
template<typename T>
class CTCLoss {
public:
    T forward(std::span<const T> logits, size_t timesteps, size_t classes,
              std::span<const size_t> labels, size_t blank = 0) {
        logits_.assign(logits.begin(), logits.end());
        labels_.assign(labels.begin(), labels.end());
        timesteps_ = timesteps;
        classes_ = classes;
        blank_ = blank;
        ready_ = true;
        return ctc_loss(logits, timesteps, classes, labels, blank);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return ctc_loss_gradient<T>(logits_, timesteps_, classes_, labels_, blank_);
    }
private:
    std::vector<T> logits_;
    std::vector<size_t> labels_;
    size_t timesteps_ = 0, classes_ = 0, blank_ = 0;
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
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return masked_language_model_gradient<T>(logits_, tokens_, classes_, positions_, targets_);
    }
private:
    std::vector<T> logits_;
    std::vector<size_t> positions_, targets_;
    size_t tokens_ = 0, classes_ = 0;
    bool ready_ = false;
};

} // namespace nexusloss::sequence

#endif
