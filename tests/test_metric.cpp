#include <nexusloss/nexusloss.hpp>
#include <vector>

#include "grad_check.hpp"

TEST(Metric, TripletAndContrastive) {
    const std::vector<double> anchor{1.0, 0.0};
    const std::vector<double> positive{0.8, 0.2};
    const std::vector<double> negative{-0.4, 0.3};
    EXPECT_GT(nexusloss::metric::triplet_loss<double>(anchor, positive, negative, 2.0), 0.0);
    const auto triplet = nexusloss::metric::triplet_loss_gradient<double>(anchor, positive, negative, 2.0);
    ASSERT_EQ(triplet.size(), 6u);
    expect_finite(triplet);
    const auto contrast = nexusloss::metric::contrastive_loss_gradient<double>(anchor, positive, true, 1.0);
    expect_finite(contrast);
    EXPECT_NEAR(nexusloss::metric::contrastive_loss<double>(anchor, anchor, true), 0.0, 1e-12);
}

TEST(Metric, ArcFaceGradientMatchesFiniteDifference) {
    const std::vector<double> cosine{0.2, 0.5, -0.1};
    auto pack = [](const std::vector<double>& point, size_t index, double value) {
        auto copy = point;
        copy[index] = value;
        return copy;
    };
    const auto gradient = nexusloss::metric::angular_margin_gradient<double>(cosine, 1, 0.2, 8.0, 0);
    for (size_t index = 0; index < cosine.size(); ++index) {
        const double step = 1e-6;
        const double numerical =
            (nexusloss::metric::arcface_loss<double>(pack(cosine, index, cosine[index] + step), 1, 0.2, 8.0)
             - nexusloss::metric::arcface_loss<double>(pack(cosine, index, cosine[index] - step), 1, 0.2, 8.0))
            / (2.0 * step);
        EXPECT_NEAR(gradient[index], numerical, 1e-4) << index;
    }
}
