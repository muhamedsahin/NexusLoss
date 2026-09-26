#include <nexusloss/nexusloss.hpp>
#include <vector>

#include "grad_check.hpp"

TEST(Generative, AdversarialWassersteinAndElbo) {
    const std::vector<double> real{0.5};
    expect_matches_finite_difference(
        [&](const std::vector<double>& fake) {
            return nexusloss::generative::adversarial_discriminator_loss<double>(real, fake);
        },
        [&](const std::vector<double>& fake) {
            const auto full = nexusloss::generative::adversarial_discriminator_gradient<double>(real, fake);
            return std::vector<double>{full[1]};
        },
        {-0.2});
    expect_matches_finite_difference(
        [&](const std::vector<double>& fake) { return nexusloss::generative::adversarial_generator_loss<double>(fake); },
        [&](const std::vector<double>& fake) { return nexusloss::generative::adversarial_generator_gradient<double>(fake); },
        {0.4, -0.7});
    const std::vector<double> real_score{2.0};
    const std::vector<double> fake_score{0.5};
    EXPECT_NEAR(nexusloss::generative::wasserstein_discriminator_loss<double>(real_score, fake_score), -1.5, 1e-12);
    expect_matches_finite_difference(
        [&](const std::vector<double>& scores) { return nexusloss::generative::least_squares_gan_loss<double>(scores, 1.0); },
        [&](const std::vector<double>& scores) { return nexusloss::generative::least_squares_gradient<double>(scores, 1.0); },
        {0.2, 1.4});
    nexusloss::generative::VAEELBOLoss<double> elbo;
    const std::vector<double> mean{0.2, -0.1};
    const std::vector<double> log_variance{0.0, -0.5};
    EXPECT_TRUE(std::isfinite(elbo.forward(0.3, mean, log_variance)));
    const auto gradient = elbo.backward();
    EXPECT_NEAR(gradient[0], 0.2, 1e-12);
    EXPECT_NEAR(gradient[1], -0.1, 1e-12);
    expect_finite(gradient);
}
