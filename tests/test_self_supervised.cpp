#include <nexusloss/nexusloss.hpp>
#include <span>
#include <vector>

#include "grad_check.hpp"

TEST(SelfSupervised, InfoNceByolBarlow) {
    const std::vector<double> negatives{-0.2, 0.1};
    auto loss = [&](const std::vector<double>& similarities) {
        return nexusloss::self_supervised::info_nce_loss(similarities[0],
            std::span<const double>(similarities.data() + 1, 2), 0.5);
    };
    auto gradient = [&](const std::vector<double>& similarities) {
        return nexusloss::self_supervised::info_nce_gradient(similarities[0],
            std::span<const double>(similarities.data() + 1, 2), 0.5);
    };
    expect_matches_finite_difference(loss, gradient, {1.2, -0.2, 0.1});
    const std::vector<double> unit{1.0, 0.0};
    EXPECT_NEAR(nexusloss::self_supervised::byol_loss<double>(unit, unit), 0.0, 1e-12);
    const std::vector<double> target{1.0, 0.2};
    expect_matches_finite_difference(
        [&](const std::vector<double>& prediction) { return nexusloss::self_supervised::byol_loss<double>(prediction, target); },
        [&](const std::vector<double>& prediction) { return nexusloss::self_supervised::byol_gradient<double>(prediction, target); },
        {0.4, 0.8});
    const std::vector<double> correlation{1.1, 0.2, -0.1, 0.7};
    expect_matches_finite_difference(
        [&](const std::vector<double>& matrix) { return nexusloss::self_supervised::barlow_twins_loss<double>(matrix, 2, 0.1); },
        [&](const std::vector<double>& matrix) { return nexusloss::self_supervised::barlow_twins_gradient<double>(matrix, 2, 0.1); },
        correlation);
}
