#include <nexusloss/nexusloss.hpp>
#include <span>
#include <vector>

#include "grad_check.hpp"

TEST(Survival, CoxAndWeibullGradients) {
    const std::vector<double> durations{1.0, 2.0, 3.0};
    const std::vector<char> observed{1, 0, 1};
    expect_matches_finite_difference(
        [&](const std::vector<double>& risk) {
            return nexusloss::survival::cox_ph_loss<double>(risk, durations, std::span<const char>(observed));
        },
        [&](const std::vector<double>& risk) {
            return nexusloss::survival::cox_ph_gradient<double>(risk, durations, std::span<const char>(observed));
        },
        {0.1, -0.2, 0.4});
    auto weibull = [](const std::vector<double>& parameters) {
        return nexusloss::survival::weibull_nll(1.7, true, parameters[0], parameters[1]);
    };
    auto weibull_grad = [](const std::vector<double>& parameters) {
        return nexusloss::survival::weibull_nll_gradient(1.7, true, parameters[0], parameters[1]);
    };
    expect_matches_finite_difference(weibull, weibull_grad, {0.3, -0.2});
    EXPECT_TRUE(std::isfinite(nexusloss::survival::weibull_nll(2.0, false, 0.0, 0.0)));
}
