#include <nexusloss/nexusloss.hpp>
#include <vector>

#include "grad_check.hpp"

TEST(Regression, MseMatchesClosedFormAndReduction) {
    const std::vector<double> prediction{1.0, 3.0};
    const std::vector<double> target{1.0, 1.0};
    nexusloss::MSELoss<double> mean_loss;
    nexusloss::MSELoss<double> sum_loss(nexusloss::core::ReductionType::Sum);
    EXPECT_NEAR(mean_loss.compute(prediction, target)[0], 2.0, 1e-12);
    EXPECT_NEAR(sum_loss.compute(prediction, target)[0], 4.0, 1e-12);
    const auto gradient = mean_loss.gradient(prediction, target);
    EXPECT_NEAR(gradient[0], 0.0, 1e-12);
    EXPECT_NEAR(gradient[1], 2.0, 1e-12);
    EXPECT_THROW(mean_loss.backward(), std::logic_error);
    EXPECT_NEAR(mean_loss.forward(prediction, target)[0], 2.0, 1e-12);
    EXPECT_NEAR(mean_loss.backward()[1], 2.0, 1e-12);
}

TEST(Regression, AnalyticGradientsMatchFiniteDifference) {
    const std::vector<double> target{0.5, 1.5};
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::MSELoss<double>{}.compute(point, target)[0]; },
        [&](const std::vector<double>& point) { return nexusloss::MSELoss<double>{}.gradient(point, target); },
        {0.2, 1.0});
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::MAELoss<double>{}.compute(point, target)[0]; },
        [&](const std::vector<double>& point) { return nexusloss::MAELoss<double>{}.gradient(point, target); },
        {0.2, 1.0});
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::HuberLoss<double>{1.0}.compute(point, target)[0]; },
        [&](const std::vector<double>& point) { return nexusloss::HuberLoss<double>{1.0}.gradient(point, target); },
        {0.2, 4.0});
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::LogCoshLoss<double>{}.compute(point, target)[0]; },
        [&](const std::vector<double>& point) { return nexusloss::LogCoshLoss<double>{}.gradient(point, target); },
        {0.2, 3.0});
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::QuantileLoss<double>{0.8}.compute(point, target)[0]; },
        [&](const std::vector<double>& point) { return nexusloss::QuantileLoss<double>{0.8}.gradient(point, target); },
        {0.2, 2.0});
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::PoissonNLLLoss<double>{}.compute(point, target)[0]; },
        [&](const std::vector<double>& point) { return nexusloss::PoissonNLLLoss<double>{}.gradient(point, target); },
        {0.2, -0.4});
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::TweedieLoss<double>{1.5}.compute(point, target)[0]; },
        [&](const std::vector<double>& point) { return nexusloss::TweedieLoss<double>{1.5}.gradient(point, target); },
        {1.2, 2.5});
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::MSLELoss<double>{}.compute(point, target)[0]; },
        [&](const std::vector<double>& point) { return nexusloss::MSLELoss<double>{}.gradient(point, target); },
        {1.0, 2.0});
}

TEST(Regression, ZeroPredictionStaysFinite) {
    const std::vector<double> zeros{0.0, 0.0};
    EXPECT_TRUE(std::isfinite(nexusloss::HuberLoss<double>{}.compute(zeros, zeros)[0]));
    const std::vector<double> shifted{1.0, -4.0};
    EXPECT_TRUE(std::isfinite(nexusloss::CauchyLoss<double>{}.compute(zeros, shifted)[0]));
}
