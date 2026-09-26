#include <nexusloss/nexusloss.hpp>
#include <vector>

#include "grad_check.hpp"

TEST(Segmentation, IdenticalMasksHaveZeroOverlapLoss) {
    const std::vector<double> mask{1.0, 0.0, 1.0};
    EXPECT_NEAR(nexusloss::segmentation::dice_loss<double>(mask, mask, 0.0), 0.0, 1e-12);
    EXPECT_NEAR(nexusloss::segmentation::iou_loss<double>(mask, mask, 0.0), 0.0, 1e-12);
}

TEST(Segmentation, EmptyAndDisjointMasksStayFinite) {
    const std::vector<double> zeros{0.0, 0.0, 0.0};
    const std::vector<double> ones{1.0, 1.0, 1.0};
    EXPECT_TRUE(std::isfinite(nexusloss::segmentation::dice_loss<double>(zeros, zeros)));
    EXPECT_GT(nexusloss::segmentation::dice_loss<double>(ones, zeros), 0.0);
    EXPECT_TRUE(std::isfinite(nexusloss::segmentation::tversky_loss<double>(ones, zeros)));
}

TEST(Segmentation, OverlapGradientsMatchFiniteDifference) {
    const std::vector<double> target{1.0, 0.0, 1.0, 0.0};
    const std::vector<double> start{0.8, 0.2, 0.4, 0.1};
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::segmentation::dice_loss<double>(point, target); },
        [&](const std::vector<double>& point) { return nexusloss::segmentation::dice_loss_gradient<double>(point, target); },
        start);
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::segmentation::iou_loss<double>(point, target); },
        [&](const std::vector<double>& point) { return nexusloss::segmentation::iou_loss_gradient<double>(point, target); },
        start);
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) {
            return nexusloss::segmentation::focal_tversky_loss<double>(point, target, 0.3, 0.7, 1.5);
        },
        [&](const std::vector<double>& point) {
            return nexusloss::segmentation::focal_tversky_loss_gradient<double>(point, target, 0.3, 0.7, 1.5);
        },
        start);
    nexusloss::segmentation::DiceLoss<double> dice;
    dice.forward(start, target);
    expect_finite(dice.backward());
}
