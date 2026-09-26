#include <cmath>
#include <nexusloss/nexusloss.hpp>
#include <vector>

#include "grad_check.hpp"

TEST(Classification, BceWithLogitsMatchesReference) {
    nexusloss::BCEWithLogitsLoss<double> loss;
    const std::vector<double> prediction{0.0};
    const std::vector<double> target{1.0};
    EXPECT_NEAR(loss.compute(prediction, target)[0], std::log(2.0), 1e-12);
    EXPECT_NEAR(loss.gradient(prediction, target)[0], -0.5, 1e-12);
    EXPECT_NEAR(nexusloss::classification::binary_cross_entropy_with_logits(1000.0, 1.0), 0.0, 1e-9);
}

TEST(Classification, CrossEntropyGradientIsSoftmaxMinusOneHot) {
    const std::vector<double> logits{1.0, 2.0, 3.0};
    nexusloss::classification::CrossEntropyLoss<double> loss;
    EXPECT_NEAR(loss.forward(logits, 2), 0.4076059644443803, 1e-9);
    const auto gradient = loss.backward();
    const auto probs = nexusloss::core::softmax<double>(logits);
    EXPECT_NEAR(gradient[0], probs[0], 1e-12);
    EXPECT_NEAR(gradient[1], probs[1], 1e-12);
    EXPECT_NEAR(gradient[2], probs[2] - 1.0, 1e-12);
    expect_finite(gradient);
}

TEST(Classification, ElementLossGradientsMatchFiniteDifference) {
    const std::vector<double> target{1.0, 0.0};
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::BCEWithLogitsLoss<double>{}.compute(point, target)[0]; },
        [&](const std::vector<double>& point) { return nexusloss::BCEWithLogitsLoss<double>{}.gradient(point, target); },
        {0.4, -1.2});
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::BCELoss<double>{}.compute(point, target)[0]; },
        [&](const std::vector<double>& point) { return nexusloss::BCELoss<double>{}.gradient(point, target); },
        {0.7, 0.2});
    const std::vector<double> signs{1.0, -1.0};
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::HingeLoss<double>{}.compute(point, signs)[0]; },
        [&](const std::vector<double>& point) { return nexusloss::HingeLoss<double>{}.gradient(point, signs); },
        {0.2, 0.3});
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::FocalLoss<double>{}.compute(point, target)[0]; },
        [&](const std::vector<double>& point) { return nexusloss::FocalLoss<double>{}.gradient(point, target); },
        {0.6, 0.25});
}

TEST(Classification, KlAndLabelSmoothingAreFinite) {
    const std::vector<double> prediction{0.2, 0.8};
    const std::vector<double> target{0.5, 0.5};
    nexusloss::classification::KLDivLoss<double> kl;
    EXPECT_TRUE(std::isfinite(kl.forward(prediction, target)));
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::classification::kl_divergence<double>(point, target); },
        [&](const std::vector<double>& point) { return nexusloss::classification::kl_divergence_gradient<double>(point, target); },
        prediction);
    nexusloss::classification::LabelSmoothingCrossEntropyLoss<double> smooth(0.1);
    const std::vector<double> smooth_logits{1.0, 2.0, 0.5};
    EXPECT_GT(smooth.forward(smooth_logits, 1), 0.0);
    expect_finite(smooth.backward());
}
