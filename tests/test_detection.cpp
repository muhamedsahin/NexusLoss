#include <nexusloss/nexusloss.hpp>
#include <vector>

#include "grad_check.hpp"

TEST(Detection, ExactBoxesHaveZeroLoss) {
    const std::vector<double> box{0.0, 0.0, 2.0, 3.0};
    EXPECT_NEAR(nexusloss::detection::iou_loss<double>(box, box), 0.0, 1e-12);
    EXPECT_NEAR(nexusloss::detection::giou_loss<double>(box, box), 0.0, 1e-12);
    EXPECT_NEAR(nexusloss::detection::diou_loss<double>(box, box), 0.0, 1e-12);
    EXPECT_NEAR(nexusloss::detection::ciou_loss<double>(box, box), 0.0, 1e-12);
}

TEST(Detection, BoxGradientsMatchFiniteDifference) {
    const std::vector<double> target{1.0, 0.5, 3.0, 2.5};
    const std::vector<double> prediction{0.0, 0.0, 2.2, 2.0};
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::detection::iou_loss<double>(point, target); },
        [&](const std::vector<double>& point) { return nexusloss::detection::iou_loss_gradient<double>(point, target); },
        prediction, 1e-5);
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::detection::giou_loss<double>(point, target); },
        [&](const std::vector<double>& point) { return nexusloss::detection::giou_loss_gradient<double>(point, target); },
        prediction, 1e-5);
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::detection::diou_loss<double>(point, target); },
        [&](const std::vector<double>& point) { return nexusloss::detection::diou_loss_gradient<double>(point, target); },
        prediction, 1e-5);
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::detection::ciou_loss<double>(point, target); },
        [&](const std::vector<double>& point) { return nexusloss::detection::ciou_loss_gradient<double>(point, target); },
        prediction, 1e-4);
}

TEST(Detection, FocalAndSmoothL1) {
    nexusloss::detection::DetectionFocalLoss<double> focal;
    const std::vector<double> target{1.0};
    const std::vector<double> logit{0.3};
    EXPECT_TRUE(std::isfinite(focal.forward(logit, target)));
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::detection::DetectionFocalLoss<double>{}.forward(point, target); },
        [&](const std::vector<double>& point) {
            nexusloss::detection::DetectionFocalLoss<double> loss;
            loss.forward(point, target);
            return loss.backward();
        },
        {0.3});
    nexusloss::detection::SmoothL1BBoxLoss<double> bbox;
    const std::vector<double> delta_target{0.0, 0.0, 0.0, 0.0};
    const std::vector<double> deltas{0.2, -1.5, 0.0, 3.0};
    EXPECT_GT(bbox.forward(deltas, delta_target), 0.0);
    expect_finite(bbox.backward());
}
