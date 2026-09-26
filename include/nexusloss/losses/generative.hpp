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

#include <cmath>
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

template<typename T>
std::vector<T> adversarial_discriminator_gradient(std::span<const T> real_logits, std::span<const T> fake_logits) {
    if (real_logits.empty() || fake_logits.empty())
        throw std::invalid_argument("NexusLoss: GAN score inputs cannot be empty");
    std::vector<T> grad(real_logits.size() + fake_logits.size());
    const T inv = T(1) / static_cast<T>(grad.size());
    for (size_t i = 0; i < real_logits.size(); ++i)
        grad[i] = (core::stable_sigmoid(real_logits[i]) - T(1)) * inv;
    for (size_t i = 0; i < fake_logits.size(); ++i)
        grad[real_logits.size() + i] = core::stable_sigmoid(fake_logits[i]) * inv;
    return grad;
}

template<typename T>
std::vector<T> adversarial_generator_gradient(std::span<const T> fake_logits) {
    if (fake_logits.empty()) throw std::invalid_argument("NexusLoss: loss input cannot be empty");
    std::vector<T> grad(fake_logits.size());
    const T inv = T(1) / static_cast<T>(grad.size());
    for (size_t i = 0; i < fake_logits.size(); ++i)
        grad[i] = -core::stable_sigmoid(fake_logits[i]) * inv;
    return grad;
}

template<typename T>
std::vector<T> wasserstein_discriminator_gradient(std::span<const T> real_scores, std::span<const T> fake_scores) {
    if (real_scores.empty() || fake_scores.empty())
        throw std::invalid_argument("NexusLoss: loss input cannot be empty");
    std::vector<T> grad(real_scores.size() + fake_scores.size());
    for (size_t i = 0; i < real_scores.size(); ++i) grad[i] = T(-1) / static_cast<T>(real_scores.size());
    for (size_t i = 0; i < fake_scores.size(); ++i)
        grad[real_scores.size() + i] = T(1) / static_cast<T>(fake_scores.size());
    return grad;
}

template<typename T>
std::vector<T> least_squares_gradient(std::span<const T> scores, T target) {
    if (scores.empty()) throw std::invalid_argument("NexusLoss: loss input cannot be empty");
    std::vector<T> grad(scores.size());
    const T scale = T(2) / static_cast<T>(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) grad[i] = scale * (scores[i] - target);
    return grad;
}

/// @brief ELBO'nun [ortalama | log-varyans] turevi. Rekonstruksiyon terimi ayri loss'tan gelir.
template<typename T>
std::vector<T> elbo_gradient(std::span<const T> mean, std::span<const T> log_variance, T beta = T(1)) {
    if (mean.empty() || mean.size() != log_variance.size() || beta < T(0))
        throw std::invalid_argument("NexusLoss: invalid VAE parameters");
    std::vector<T> grad(mean.size() * 2);
    for (size_t i = 0; i < mean.size(); ++i) {
        grad[i] = beta * mean[i];
        grad[mean.size() + i] = beta * T(-0.5) * (T(1) - std::exp(log_variance[i]));
    }
    return grad;
}

template<typename T>
class AdversarialLoss {
public:
    T discriminator(std::span<const T> real_logits, std::span<const T> fake_logits) {
        return adversarial_discriminator_loss(real_logits, fake_logits);
    }
    std::vector<T> discriminator_backward(std::span<const T> real_logits, std::span<const T> fake_logits) const {
        return adversarial_discriminator_gradient<T>(real_logits, fake_logits);
    }
    T generator(std::span<const T> fake_logits) { return adversarial_generator_loss(fake_logits); }
    std::vector<T> generator_backward(std::span<const T> fake_logits) const {
        return adversarial_generator_gradient<T>(fake_logits);
    }
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
    T generator(std::span<const T> fake_scores) { return wasserstein_generator_loss(fake_scores); }
};

template<typename T>
class LSGANLoss {
public:
    T forward(std::span<const T> scores, T target) { return least_squares_gan_loss(scores, target); }
    std::vector<T> backward(std::span<const T> scores, T target) const {
        return least_squares_gradient<T>(scores, target);
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
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return elbo_gradient<T>(mean_, log_variance_, beta_);
    }
private:
    T beta_;
    std::vector<T> mean_, log_variance_;
    bool ready_ = false;
};

} // namespace nexusloss::generative

#endif
