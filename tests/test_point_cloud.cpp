#include <cstddef>
#include <nexusloss/nexusloss.hpp>
#include <vector>

#include "grad_check.hpp"

TEST(PointCloud, IdenticalCloudsHaveZeroDistance) {
    const std::vector<double> cloud{0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    EXPECT_NEAR(nexusloss::point_cloud::chamfer_distance<double>(cloud, cloud), 0.0, 1e-12);
    EXPECT_NEAR(nexusloss::point_cloud::earth_movers_distance<double>(cloud, cloud), 0.0, 1e-12);
}

TEST(PointCloud, ChamferGradientMatchesFiniteDifference) {
    const std::vector<double> other{0.2, 0.1, 3.0, 0.0};
    auto loss = [&](const std::vector<double>& point) {
        return nexusloss::point_cloud::chamfer_distance<double>(point, other, 2);
    };
    auto full_gradient = [&](const std::vector<double>& point) {
        const auto gradient = nexusloss::point_cloud::chamfer_distance_gradient<double>(point, other, 2);
        return std::vector<double>(gradient.begin(), gradient.begin() + static_cast<std::ptrdiff_t>(point.size()));
    };
    expect_matches_finite_difference(loss, full_gradient, {0.0, 0.0, 2.2, 0.4}, 1e-5);
}

TEST(PointCloud, EarthMoverGradientMatchesFiniteDifference) {
    const std::vector<double> other{1.0, 0.2, 0.1, 1.5};
    auto loss = [&](const std::vector<double>& point) {
        return nexusloss::point_cloud::earth_movers_distance<double>(point, other, 2);
    };
    auto full_gradient = [&](const std::vector<double>& point) {
        const auto gradient = nexusloss::point_cloud::earth_movers_gradient<double>(point, other, 2);
        return std::vector<double>(gradient.begin(), gradient.begin() + static_cast<std::ptrdiff_t>(point.size()));
    };
    expect_matches_finite_difference(loss, full_gradient, {0.0, 0.0, 0.8, 1.2}, 1e-5);
}
