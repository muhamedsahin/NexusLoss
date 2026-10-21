/**
 * @file generative.hpp
 * @brief GAN, VAE ve difuzyon egitiminde kullanilan kayiplar.
 *
 * Ayirtici ve uretici skorlari logit veya WGAN'da sinirlanmamis skordur.
 * Vanilla GAN minimax formu logit uzerinde kararli BCE ile yazilir.
 * ELBO, rekonstruksiyon skaleri ile ortalama/log-varyans uzerindeki KL'yi toplar.
 *
 * @see core/utils.hpp  stable_sigmoid
 */
#ifndef NEXUSLOSS_GENERATIVE_HPP
#define NEXUSLOSS_GENERATIVE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>
#include "../core/utils.hpp"

namespace nexusloss::generative {

template<typename T>
T mean(std::span<const T> values) {
    if (values.empty()) throw std::invalid_argument("NexusLoss: loss input cannot be empty");
    T sum = T(0);
    for (T value : values) sum += value;
    return sum / static_cast<T>(values.size());
}

template<typename T>
T wasserstein_discriminator_loss(std::span<const T> real_scores, std::span<const T> fake_scores) {
    return mean(fake_scores) - mean(real_scores);
}
template<typename T>
T wasserstein_generator_loss(std::span<const T> fake_scores) { return -mean(fake_scores); }

template<typename T>
T least_squares_gan_loss(std::span<const T> scores, T target) {
    if (scores.empty()) throw std::invalid_argument("NexusLoss: loss input cannot be empty");
    T sum = T(0);
    for (T score : scores) { const T error = score - target; sum += error * error; }
    return sum / static_cast<T>(scores.size());
}

template<typename T>
T hinge_discriminator_loss(std::span<const T> real_scores, std::span<const T> fake_scores) {
    if (real_scores.empty() || fake_scores.empty())
        throw std::invalid_argument("NexusLoss: GAN score inputs cannot be empty");
    T sum = T(0);
    for (T score : real_scores) sum += std::max(T(0), T(1) - score);
    for (T score : fake_scores) sum += std::max(T(0), T(1) + score);
    return sum / static_cast<T>(real_scores.size() + fake_scores.size());
}
template<typename T>
T hinge_generator_loss(std::span<const T> fake_scores) { return -mean(fake_scores); }

template<typename T>
T elbo_loss(T reconstruction_loss, std::span<const T> mean, std::span<const T> log_variance,
            T beta = T(1)) {
    if (mean.empty() || mean.size() != log_variance.size() || beta < T(0))
        throw std::invalid_argument("NexusLoss: invalid VAE parameters");
    T kl = T(0);
    for (size_t i = 0; i < mean.size(); ++i)
        kl += T(-0.5) * (T(1) + log_variance[i] - mean[i] * mean[i] - std::exp(log_variance[i]));
    return reconstruction_loss + beta * kl;
}

template<typename T>
T l1_loss(std::span<const T> a, std::span<const T> b) {
    if (a.empty() || a.size() != b.size()) throw std::invalid_argument("NexusLoss: vectors must have equal non-zero size");
    T sum = T(0);
    for (size_t i = 0; i < a.size(); ++i) sum += std::abs(a[i] - b[i]);
    return sum / static_cast<T>(a.size());
}
template<typename T>
T l2_loss(std::span<const T> a, std::span<const T> b) {
    if (a.empty() || a.size() != b.size()) throw std::invalid_argument("NexusLoss: vectors must have equal non-zero size");
    T sum = T(0);
    for (size_t i = 0; i < a.size(); ++i) { const T d = a[i] - b[i]; sum += d * d; }
    return sum / static_cast<T>(a.size());
}
template<typename T>
T feature_matching_loss(std::span<const T> real_features, std::span<const T> fake_features) {
    return l1_loss(real_features, fake_features);
}
template<typename T>
T cycle_consistency_loss(std::span<const T> reconstructed, std::span<const T> original) {
    return l1_loss(reconstructed, original);
}
template<typename T>
T identity_loss(std::span<const T> identity_output, std::span<const T> input) {
    return l1_loss(identity_output, input);
}
template<typename T>
T perceptual_loss(std::span<const T> predicted_features, std::span<const T> target_features) {
    return l2_loss(predicted_features, target_features);
}

template<typename T>
T style_loss(std::span<const T> predicted, std::span<const T> target, size_t channels, size_t positions) {
    if (channels == 0 || positions == 0 || predicted.size() != channels * positions
        || target.size() != predicted.size())
        throw std::invalid_argument("NexusLoss: style features must be flattened channel-major matrices");
    T sum = T(0);
    const T normalizer = T(1) / static_cast<T>(channels * positions);
    for (size_t i = 0; i < channels; ++i) {
        for (size_t j = 0; j < channels; ++j) {
            T gram_p = T(0), gram_t = T(0);
            for (size_t k = 0; k < positions; ++k) {
                gram_p += predicted[i * positions + k] * predicted[j * positions + k];
                gram_t += target[i * positions + k] * target[j * positions + k];
            }
            const T error = (gram_p - gram_t) * normalizer;
            sum += error * error;
        }
    }
    return sum / static_cast<T>(channels * channels);
}

template<typename T>
T total_variation_loss(std::span<const T> image, size_t height, size_t width, size_t channels = 1) {
    if (!height || !width || !channels || image.size() != height * width * channels)
        throw std::invalid_argument("NexusLoss: image size must match height, width, and channels");
    T sum = T(0);
    size_t count = 0;
    for (size_t y = 0; y < height; ++y)
        for (size_t x = 0; x < width; ++x)
            for (size_t c = 0; c < channels; ++c) {
                const size_t index = (y * width + x) * channels + c;
                if (x + 1 < width) { sum += std::abs(image[index] - image[index + channels]); ++count; }
                if (y + 1 < height) { sum += std::abs(image[index] - image[index + width * channels]); ++count; }
            }
    return count ? sum / static_cast<T>(count) : T(0);
}

template<typename T>
T diffusion_epsilon_loss(std::span<const T> predicted_noise, std::span<const T> noise) {
    return l2_loss(predicted_noise, noise);
}

/// @brief Vanilla GAN ayirtici kaybi: -log D(gercek) - log(1-D(sahte)), logit uzerinde kararli.
template<typename T>
T adversarial_discriminator_loss(std::span<const T> real_logits, std::span<const T> fake_logits) {
    if (real_logits.empty() || fake_logits.empty())
        throw std::invalid_argument("NexusLoss: GAN score inputs cannot be empty");
    T sum = T(0);
    for (T logit : real_logits)
        sum += std::max(logit, T(0)) - logit + std::log1p(std::exp(-std::abs(logit)));
    for (T logit : fake_logits)
        sum += std::max(logit, T(0)) + std::log1p(std::exp(-std::abs(logit)));
    return sum / static_cast<T>(real_logits.size() + fake_logits.size());
}

/// @brief Uretici minimax terimi: log(1-D(G(z))). Doymaya egilimlidir; pratikte LSGAN/WGAN tercih edilir.
template<typename T>
T adversarial_generator_loss(std::span<const T> fake_logits) {
    if (fake_logits.empty()) throw std::invalid_argument("NexusLoss: loss input cannot be empty");
    T sum = T(0);
    for (T logit : fake_logits)
        sum += -std::max(logit, T(0)) - std::log1p(std::exp(-std::abs(logit)));
    return sum / static_cast<T>(fake_logits.size());
}

namespace detail {

template<typename T>
void require_pair_scores(std::span<const T> real, std::span<const T> fake, std::span<T> grad_out) {
    if (real.empty() || fake.empty())
        throw std::invalid_argument("NexusLoss: GAN score inputs cannot be empty");
    if (grad_out.size() != real.size() + fake.size())
        throw std::invalid_argument("NexusLoss: discriminator grad_out must hold [real | fake]");
}

template<typename T>
void require_scores(std::span<const T> scores, std::span<T> grad_out) {
    if (scores.empty()) throw std::invalid_argument("NexusLoss: loss input cannot be empty");
    if (grad_out.size() != scores.size()) throw std::invalid_argument("NexusLoss: grad_out must match scores");
}

} // namespace detail

/// @brief Vanilla GAN ayirtici turevi. grad_out = [gercek | sahte] logitleri, ayirmasiz.
template<typename T>
void adversarial_discriminator_gradient_into(std::span<const T> real_logits, std::span<const T> fake_logits,
                                             std::span<T> grad_out) {
    detail::require_pair_scores(real_logits, fake_logits, grad_out);
    const T inv = T(1) / static_cast<T>(grad_out.size());
    for (size_t i = 0; i < real_logits.size(); ++i)
        grad_out[i] = (core::stable_sigmoid(real_logits[i]) - T(1)) * inv;
    for (size_t i = 0; i < fake_logits.size(); ++i)
        grad_out[real_logits.size() + i] = core::stable_sigmoid(fake_logits[i]) * inv;
}

template<typename T>
std::vector<T> adversarial_discriminator_gradient(std::span<const T> real_logits, std::span<const T> fake_logits) {
    std::vector<T> grad(real_logits.size() + fake_logits.size());
    adversarial_discriminator_gradient_into<T>(real_logits, fake_logits, grad);
    return grad;
}

template<typename T>
void adversarial_generator_gradient_into(std::span<const T> fake_logits, std::span<T> grad_out) {
    detail::require_scores(fake_logits, grad_out);
    const T inv = T(1) / static_cast<T>(grad_out.size());
    for (size_t i = 0; i < fake_logits.size(); ++i)
        grad_out[i] = -core::stable_sigmoid(fake_logits[i]) * inv;
}

template<typename T>
std::vector<T> adversarial_generator_gradient(std::span<const T> fake_logits) {
    std::vector<T> grad(fake_logits.size());
    adversarial_generator_gradient_into<T>(fake_logits, grad);
    return grad;
}

/**
 * @brief Doymayan uretici kaybi -log D(G(z)) (Goodfellow et al., 2014, bol. 3).
 * Minimax formun aksine ayirtici kazanirken de gradyan sonmez.
 */
template<typename T>
T non_saturating_generator_loss(std::span<const T> fake_logits) {
    if (fake_logits.empty()) throw std::invalid_argument("NexusLoss: loss input cannot be empty");
    T sum = T(0);
    for (T logit : fake_logits) sum += std::max(logit, T(0)) - logit + std::log1p(std::exp(-std::abs(logit)));
    return sum / static_cast<T>(fake_logits.size());
}

template<typename T>
void non_saturating_generator_gradient_into(std::span<const T> fake_logits, std::span<T> grad_out) {
    detail::require_scores(fake_logits, grad_out);
    const T inv = T(1) / static_cast<T>(grad_out.size());
    for (size_t i = 0; i < fake_logits.size(); ++i)
        grad_out[i] = (core::stable_sigmoid(fake_logits[i]) - T(1)) * inv;
}

template<typename T>
void wasserstein_discriminator_gradient_into(std::span<const T> real_scores, std::span<const T> fake_scores,
                                             std::span<T> grad_out) {
    detail::require_pair_scores(real_scores, fake_scores, grad_out);
    const T real_value = T(-1) / static_cast<T>(real_scores.size());
    const T fake_value = T(1) / static_cast<T>(fake_scores.size());
    std::fill(grad_out.begin(), grad_out.begin() + static_cast<std::ptrdiff_t>(real_scores.size()), real_value);
    std::fill(grad_out.begin() + static_cast<std::ptrdiff_t>(real_scores.size()), grad_out.end(), fake_value);
}

template<typename T>
std::vector<T> wasserstein_discriminator_gradient(std::span<const T> real_scores, std::span<const T> fake_scores) {
    std::vector<T> grad(real_scores.size() + fake_scores.size());
    wasserstein_discriminator_gradient_into<T>(real_scores, fake_scores, grad);
    return grad;
}

template<typename T>
void wasserstein_generator_gradient_into(std::span<const T> fake_scores, std::span<T> grad_out) {
    detail::require_scores(fake_scores, grad_out);
    std::fill(grad_out.begin(), grad_out.end(), T(-1) / static_cast<T>(fake_scores.size()));
}

/// @brief Hinge GAN ayirtici turevi: gercek icin -1[1-s>0], sahte icin +1[1+s>0], ortalama.
template<typename T>
void hinge_discriminator_gradient_into(std::span<const T> real_scores, std::span<const T> fake_scores,
                                       std::span<T> grad_out) {
    detail::require_pair_scores(real_scores, fake_scores, grad_out);
    const T inv = T(1) / static_cast<T>(grad_out.size());
    for (size_t i = 0; i < real_scores.size(); ++i)
        grad_out[i] = T(1) - real_scores[i] > T(0) ? -inv : T(0);
    for (size_t i = 0; i < fake_scores.size(); ++i)
        grad_out[real_scores.size() + i] = T(1) + fake_scores[i] > T(0) ? inv : T(0);
}

template<typename T>
void least_squares_gradient_into(std::span<const T> scores, T target, std::span<T> grad_out) {
    detail::require_scores(scores, grad_out);
    const T scale = T(2) / static_cast<T>(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) grad_out[i] = scale * (scores[i] - target);
}

template<typename T>
std::vector<T> least_squares_gradient(std::span<const T> scores, T target) {
    std::vector<T> grad(scores.size());
    least_squares_gradient_into<T>(scores, target, grad);
    return grad;
}

/// @brief ELBO'nun [ortalama | log-varyans] turevi. Rekonstruksiyon terimi ayri loss'tan gelir.
template<typename T>
void elbo_gradient_into(std::span<const T> mean, std::span<const T> log_variance, std::span<T> grad_out,
                        T beta = T(1)) {
    if (mean.empty() || mean.size() != log_variance.size() || beta < T(0))
        throw std::invalid_argument("NexusLoss: invalid VAE parameters");
    if (grad_out.size() != mean.size() * 2)
        throw std::invalid_argument("NexusLoss: ELBO grad_out must hold [mean | log_variance]");
    for (size_t i = 0; i < mean.size(); ++i) {
        grad_out[i] = beta * mean[i];
        grad_out[mean.size() + i] = beta * T(-0.5) * (T(1) - std::exp(log_variance[i]));
    }
}

template<typename T>
std::vector<T> elbo_gradient(std::span<const T> mean, std::span<const T> log_variance, T beta = T(1)) {
    std::vector<T> grad(mean.size() * 2);
    elbo_gradient_into<T>(mean, log_variance, grad, beta);
    return grad;
}

/**
 * @class AdversarialLoss
 * @brief Vanilla GAN. `non_saturating` acikken uretici -log D(G(z)) kullanir.
 * Durumsuzdur: girdiler her cagrida acikca verilir, `*_backward_into` ayirmaz.
 */
template<typename T>
class AdversarialLoss {
public:
    explicit AdversarialLoss(bool non_saturating = false) : non_saturating_(non_saturating) {}
    T discriminator(std::span<const T> real_logits, std::span<const T> fake_logits) {
        return adversarial_discriminator_loss(real_logits, fake_logits);
    }
    std::vector<T> discriminator_backward(std::span<const T> real_logits, std::span<const T> fake_logits) const {
        std::vector<T> grad(real_logits.size() + fake_logits.size());
        discriminator_backward_into(real_logits, fake_logits, grad);
        return grad;
    }
    void discriminator_backward_into(std::span<const T> real_logits, std::span<const T> fake_logits,
                                     std::span<T> grad_out) const {
        adversarial_discriminator_gradient_into<T>(real_logits, fake_logits, grad_out);
    }
    T generator(std::span<const T> fake_logits) {
        return non_saturating_ ? non_saturating_generator_loss(fake_logits) : adversarial_generator_loss(fake_logits);
    }
    std::vector<T> generator_backward(std::span<const T> fake_logits) const {
        std::vector<T> grad(fake_logits.size());
        generator_backward_into(fake_logits, grad);
        return grad;
    }
    void generator_backward_into(std::span<const T> fake_logits, std::span<T> grad_out) const {
        if (non_saturating_) non_saturating_generator_gradient_into<T>(fake_logits, grad_out);
        else adversarial_generator_gradient_into<T>(fake_logits, grad_out);
    }
private:
    bool non_saturating_;
};

template<typename T>
class WassersteinLoss {
public:
    T discriminator(std::span<const T> real_scores, std::span<const T> fake_scores) {
        return wasserstein_discriminator_loss(real_scores, fake_scores);
    }
    std::vector<T> discriminator_backward(std::span<const T> real_scores, std::span<const T> fake_scores) const {
        return wasserstein_discriminator_gradient<T>(real_scores, fake_scores);
    }
    void discriminator_backward_into(std::span<const T> real_scores, std::span<const T> fake_scores,
                                     std::span<T> grad_out) const {
        wasserstein_discriminator_gradient_into<T>(real_scores, fake_scores, grad_out);
    }
    T generator(std::span<const T> fake_scores) { return wasserstein_generator_loss(fake_scores); }
    std::vector<T> generator_backward(std::span<const T> fake_scores) const {
        std::vector<T> grad(fake_scores.size());
        generator_backward_into(fake_scores, grad);
        return grad;
    }
    void generator_backward_into(std::span<const T> fake_scores, std::span<T> grad_out) const {
        wasserstein_generator_gradient_into<T>(fake_scores, grad_out);
    }
};

/// @brief Hinge GAN (Miyato et al., SN-GAN 2018). Uretici terimi -mean(D(G(z))) WGAN ile aynidir.
template<typename T>
class HingeGANLoss {
public:
    T discriminator(std::span<const T> real_scores, std::span<const T> fake_scores) {
        return hinge_discriminator_loss(real_scores, fake_scores);
    }
    void discriminator_backward_into(std::span<const T> real_scores, std::span<const T> fake_scores,
                                     std::span<T> grad_out) const {
        hinge_discriminator_gradient_into<T>(real_scores, fake_scores, grad_out);
    }
    T generator(std::span<const T> fake_scores) { return hinge_generator_loss(fake_scores); }
    void generator_backward_into(std::span<const T> fake_scores, std::span<T> grad_out) const {
        wasserstein_generator_gradient_into<T>(fake_scores, grad_out);
    }
};

template<typename T>
class LSGANLoss {
public:
    T forward(std::span<const T> scores, T target) { return least_squares_gan_loss(scores, target); }
    std::vector<T> backward(std::span<const T> scores, T target) const {
        return least_squares_gradient<T>(scores, target);
    }
    void backward_into(std::span<const T> scores, T target, std::span<T> grad_out) const {
        least_squares_gradient_into<T>(scores, target, grad_out);
    }
};

template<typename T>
class VAEELBOLoss {
public:
    explicit VAEELBOLoss(T beta = T(1)) : beta_(beta) {}
    T forward(T reconstruction, std::span<const T> mean, std::span<const T> log_variance) {
        mean_.assign(mean.begin(), mean.end());
        log_variance_.assign(log_variance.begin(), log_variance.end());
        ready_ = true;
        return elbo_loss(reconstruction, mean, log_variance, beta_);
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return mean_.size() * 2; }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        elbo_gradient_into<T>(mean_, log_variance_, grad_out, beta_);
    }
private:
    T beta_;
    std::vector<T> mean_, log_variance_;
    bool ready_ = false;
};

} // namespace nexusloss::generative

#endif
