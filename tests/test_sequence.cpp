#include <nexusloss/nexusloss.hpp>
#include <vector>

#include "grad_check.hpp"

TEST(Sequence, CtcBlankSequenceAndGradient) {
    const std::vector<double> blank_logits{10.0, 0.0, 10.0, 0.0};
    EXPECT_NEAR(nexusloss::sequence::ctc_loss<double>(blank_logits, 2, 2, {}, 0), 0.0000908, 1e-6);
    const std::vector<size_t> label{1};
    const std::vector<double> logits{0.2, -0.1, 0.4, 0.3, -0.2, 0.8};
    nexusloss::sequence::CTCLoss<double> loss;
    EXPECT_TRUE(std::isfinite(loss.forward(logits, 3, 2, label, 0)));
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::sequence::ctc_loss<double>(point, 3, 2, label, 0); },
        [&](const std::vector<double>& point) { return nexusloss::sequence::ctc_loss_gradient<double>(point, 3, 2, label, 0); },
        logits, 1e-4);
}

TEST(Sequence, MaskedCrossEntropyIgnoresUnmaskedTokens) {
    const std::vector<double> logits{0.0, 2.0, 3.0, 0.0, 1.0, 0.5};
    const std::vector<size_t> positions{1};
    const std::vector<size_t> targets{0};
    nexusloss::sequence::MaskedSequenceCrossEntropyLoss<double> loss;
    EXPECT_GT(loss.forward(logits, 3, 2, positions, targets), 0.0);
    const auto gradient = loss.backward();
    EXPECT_NEAR(gradient[0], 0.0, 1e-12);
    EXPECT_NEAR(gradient[1], 0.0, 1e-12);
    expect_finite(gradient);
}
