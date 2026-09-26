#include <nexusloss/nexusloss.hpp>
#include <vector>

#include "grad_check.hpp"

TEST(Distillation, TemperaturedKlAndFeatures) {
    const std::vector<double> teacher{1.0, 0.0, -0.5};
    nexusloss::distillation::KnowledgeDistillationLoss<double> kd(2.0);
    EXPECT_NEAR(kd.forward(teacher, teacher), 0.0, 1e-12);
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) { return nexusloss::distillation::kl_divergence<double>(point, teacher, 2.0); },
        [&](const std::vector<double>& point) { return nexusloss::distillation::kl_divergence_gradient<double>(point, teacher, 2.0); },
        {0.2, -0.4, 1.1});
    const std::vector<double> teacher_features{1.0, 2.0, 3.0};
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) {
            return nexusloss::distillation::feature_distillation_loss<double>(point, teacher_features);
        },
        [&](const std::vector<double>& point) {
            return nexusloss::distillation::feature_distillation_gradient<double>(point, teacher_features);
        },
        {0.5, 2.5, 2.0});
    const std::vector<double> teacher_map{1.0, 0.0, 2.0};
    expect_matches_finite_difference(
        [&](const std::vector<double>& point) {
            return nexusloss::distillation::attention_transfer_loss<double>(point, teacher_map);
        },
        [&](const std::vector<double>& point) {
            return nexusloss::distillation::attention_transfer_gradient<double>(point, teacher_map);
        },
        {0.4, 1.0, 0.2});
}
