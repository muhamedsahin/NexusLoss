//
// Created by muham on 24.09.2026.
//
#include <gtest/gtest.h>
#include <nexusloss/nexusloss.hpp>
#include <array>
#include <vector>

TEST(Smoke, CalismaKontrolu) {
    EXPECT_EQ(1 + 1, 2);
}

TEST(Utils, SoftmaxToplamiBirOlur) {
    std::vector<double> logits{2.0, 1.0, 0.1};
    auto probs = nexusloss::core::softmax<double>(logits);
    double toplam = probs[0] + probs[1] + probs[2];
    EXPECT_NEAR(toplam, 1.0, 1e-9);
}

TEST(Utils, SoftmaxBuyukSayidaPatlamaz) {
    std::vector<double> logits{1000.0, 1.0, 0.1};
    auto probs = nexusloss::core::softmax<double>(logits);
    EXPECT_TRUE(std::isfinite(probs[0]));
}

TEST(Regression, RobustAndDistributionLosses) {
    const std::vector<double> pred{0.0, 10.0};
    const std::vector<double> target{0.0, 0.0};
    nexusloss::HuberLoss<double> huber;
    EXPECT_NEAR(huber.compute(pred, target)[0], 4.75, 1e-12);
    nexusloss::TukeyBiweightLoss<double> tukey(1.0);
    EXPECT_NEAR(tukey.compute(pred, target)[0], 1.0 / 12.0, 1e-12);
    nexusloss::QuantileLoss<double> quantile(0.9);
    EXPECT_NEAR(quantile.compute(std::vector<double>{1.0}, std::vector<double>{0.0})[0], 0.1, 1e-12);
    EXPECT_THROW(nexusloss::MSLELoss<double>{}.compute(
        std::vector<double>{-1.0}, std::vector<double>{0.0}), std::invalid_argument);
}

TEST(Classification, LogitsAndStructuredTargets) {
    EXPECT_NEAR(nexusloss::classification::binary_cross_entropy_with_logits(1000.0, 1.0), 0.0, 1e-12);
    const std::vector<double> logits{1.0, 2.0, 3.0};
    EXPECT_NEAR(nexusloss::classification::sparse_categorical_cross_entropy<double>(logits, 2),
                0.4076059644, 1e-8);
    EXPECT_GT(nexusloss::classification::focal_loss(0.99, 1.0), 0.0);
    EXPECT_EQ(nexusloss::classification::effective_number_class_weights<double>(
        std::vector<size_t>{10, 100}).size(), 2u);
}

TEST(Segmentation, OverlapAndBoundaryLosses) {
    const std::vector<double> probability{1.0, 0.0, 1.0};
    const std::vector<double> target{1.0, 0.0, 1.0};
    EXPECT_NEAR(nexusloss::segmentation::dice_loss<double>(probability, target), 0.0, 1e-12);
    EXPECT_NEAR(nexusloss::segmentation::iou_loss<double>(probability, target), 0.0, 1e-12);
    EXPECT_NEAR(nexusloss::segmentation::lovasz_hinge_loss<double>(
        std::vector<double>{2.0, -2.0, 2.0}, target), 0.0, 1e-12);
}

TEST(Detection, IoUVariantsAreZeroForExactMatch) {
    const std::array<double, 4> box{0.0, 0.0, 2.0, 3.0};
    EXPECT_NEAR(nexusloss::detection::iou_loss<double>(box, box), 0.0, 1e-12);
    EXPECT_NEAR(nexusloss::detection::giou_loss<double>(box, box), 0.0, 1e-12);
    EXPECT_NEAR(nexusloss::detection::diou_loss<double>(box, box), 0.0, 1e-12);
    EXPECT_NEAR(nexusloss::detection::ciou_loss<double>(box, box), 0.0, 1e-12);
}

TEST(Advanced, MetricRankingAndSelfSupervision) {
    const std::vector<double> a{1.0, 0.0}, b{1.0, 0.0}, c{-1.0, 0.0};
    EXPECT_NEAR(nexusloss::metric::contrastive_loss<double>(a, b, true), 0.0, 1e-12);
    EXPECT_NEAR(nexusloss::metric::triplet_loss<double>(a, b, c), 0.0, 1e-12);
    EXPECT_GT(nexusloss::ranking::bpr_loss(2.0, 0.0), 0.0);
    EXPECT_NEAR(nexusloss::self_supervised::byol_loss<double>(a, b), 0.0, 1e-12);
    EXPECT_NEAR(nexusloss::generative::diffusion_epsilon_loss<double>(a, b), 0.0, 1e-12);
}

TEST(Advanced, CtcPointCloudAndPpo) {
    const std::vector<double> blank_logits{10.0, 0.0, 10.0, 0.0};
    EXPECT_NEAR(nexusloss::sequence::ctc_loss<double>(blank_logits, 2, 2, std::vector<size_t>{}, 0),
                0.0000908, 1e-6);
    const std::vector<double> cloud{0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    EXPECT_NEAR(nexusloss::point_cloud::earth_movers_distance<double>(cloud, cloud), 0.0, 1e-12);
    const std::vector<double> logs{0.0}, old_logs{0.0}, advantages{1.0};
    EXPECT_NEAR(nexusloss::reinforcement::ppo_clipped_surrogate_loss<double>(logs, old_logs, advantages),
                -1.0, 1e-12);
}

TEST(Advanced, RankingLanguageAndStableContrastive) {
    EXPECT_NEAR(nexusloss::self_supervised::info_nce_loss<double>(
        1000.0, std::vector<double>{-1000.0, -500.0}, 0.1), 0.0, 1e-12);
    EXPECT_LT(nexusloss::ranking::lambda_rank_loss<double>(
        std::vector<double>{3.0, 1.0, 0.0}, std::vector<double>{2.0, 1.0, 0.0}), 0.1);
    const std::vector<double> token_logits{0.0, 3.0, 3.0, 0.0};
    EXPECT_LT(nexusloss::sequence::masked_language_model_loss<double>(
        token_logits, 2, 2, std::vector<size_t>{0}, std::vector<size_t>{1}), 0.1);
    EXPECT_NEAR(nexusloss::sequence::next_sentence_prediction_loss(0.0, 1.0),
                std::log(2.0), 1e-12);
}