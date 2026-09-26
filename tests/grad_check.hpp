#pragma once

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

inline void expect_finite(const std::vector<double>& values) {
    for (double value : values) EXPECT_TRUE(std::isfinite(value));
}

template<typename LossAt, typename Analytic>
void expect_matches_finite_difference(LossAt loss_at, Analytic analytic, std::vector<double> point,
                                      double tolerance = 1e-5, double step = 1e-6) {
    const std::vector<double> gradient = analytic(point);
    ASSERT_EQ(gradient.size(), point.size());
    for (size_t index = 0; index < point.size(); ++index) {
        std::vector<double> above = point;
        std::vector<double> below = point;
        above[index] += step;
        below[index] -= step;
        const double numerical = (loss_at(above) - loss_at(below)) / (2.0 * step);
        EXPECT_NEAR(gradient[index], numerical, tolerance) << "coordinate " << index;
    }
    expect_finite(gradient);
}
