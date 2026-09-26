#include <nexusloss/nexusloss.hpp>
#include <vector>

#include "grad_check.hpp"

TEST(Ranking, PairwiseAndListwise) {
    EXPECT_NEAR(nexusloss::ranking::margin_ranking_loss(1.0, 0.2, 1.0, 0.5), 0.0, 1e-12);
    const auto margin = nexusloss::ranking::margin_ranking_gradient(0.0, 0.2, 1.0, 0.5);
    EXPECT_NEAR(margin[0], -1.0, 1e-12);
    EXPECT_NEAR(margin[1], 1.0, 1e-12);
    expect_matches_finite_difference(
        [](const std::vector<double>& point) { return nexusloss::ranking::ranknet_loss(point[0], point[1]); },
        [](const std::vector<double>& point) { return nexusloss::ranking::ranknet_gradient(point[0], point[1]); },
        {0.4, -0.3});
    const std::vector<double> relevance{2.0, 0.0, 1.0};
    expect_matches_finite_difference(
        [&](const std::vector<double>& scores) { return nexusloss::ranking::listnet_loss<double>(scores, relevance); },
        [&](const std::vector<double>& scores) { return nexusloss::ranking::listnet_gradient<double>(scores, relevance); },
        {0.2, 1.0, -0.4});
    expect_matches_finite_difference(
        [&](const std::vector<double>& scores) { return nexusloss::ranking::listmle_loss<double>(scores, relevance); },
        [&](const std::vector<double>& scores) { return nexusloss::ranking::listmle_gradient<double>(scores, relevance); },
        {0.2, 1.0, -0.4});
}
