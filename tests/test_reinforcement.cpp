#include <nexusloss/nexusloss.hpp>
#include <vector>

#include "grad_check.hpp"

TEST(Reinforcement, PolicyValuePpoEntropy) {
    const std::vector<double> advantages{1.0, -0.5};
    expect_matches_finite_difference(
        [&](const std::vector<double>& logs) { return nexusloss::reinforcement::policy_gradient_loss<double>(logs, advantages); },
        [&](const std::vector<double>& logs) { return nexusloss::reinforcement::policy_gradient<double>(logs, advantages); },
        {-0.2, -1.0});
    const std::vector<double> returns{1.0, 2.0};
    expect_matches_finite_difference(
        [&](const std::vector<double>& values) { return nexusloss::reinforcement::value_loss<double>(values, returns); },
        [&](const std::vector<double>& values) { return nexusloss::reinforcement::value_loss_gradient<double>(values, returns); },
        {0.5, 1.2});
    const std::vector<double> old_logs{0.0, -0.2};
    const std::vector<double> ppo_advantages{0.8, 0.3};
    expect_matches_finite_difference(
        [&](const std::vector<double>& logs) {
            return nexusloss::reinforcement::ppo_clipped_surrogate_loss<double>(logs, old_logs, ppo_advantages, 0.2);
        },
        [&](const std::vector<double>& logs) {
            return nexusloss::reinforcement::ppo_clipped_surrogate_gradient<double>(logs, old_logs, ppo_advantages, 0.2);
        },
        {0.0, -0.2});
    const std::vector<double> policy{0.2, 0.5, 0.3};
    expect_matches_finite_difference(
        [&](const std::vector<double>& probabilities) {
            return nexusloss::reinforcement::entropy_bonus_loss<double>(probabilities);
        },
        [&](const std::vector<double>& probabilities) {
            return nexusloss::reinforcement::entropy_bonus_gradient<double>(probabilities);
        },
        policy);
}
