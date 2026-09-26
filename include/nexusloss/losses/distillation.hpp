/**
 * @file distillation.hpp
 * @brief Ogretmen modelinden ogrenciye bilgi aktaran kayiplar.
 *
 * KnowledgeDistillationLoss sicaklikla yumusatilmis logitler uzerinde KL
 * kullanir (Hinton et al., 2015). Feature ve attention kayiplari ara
 * temsilleri hizalar. Girdi logit veya ozellik vektorudur; agirlik guncellemesi
 * bu dosyanin disindadir.
 *
 * @see core/utils.hpp  log-sum-exp / softmax kararliligi
 */
#ifndef NEXUSLOSS_DISTILLATION_HPP
#define NEXUSLOSS_DISTILLATION_HPP

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>
#include "../core/utils.hpp"

namespace nexusloss::distillation {

template<typename T>
T kl_divergence(std::span<const T> student_logits, std::span<const T> teacher_logits,
                T temperature = T(1)) {
    if (student_logits.empty() || student_logits.size() != teacher_logits.size() || temperature <= T(0))
        throw std::invalid_argument("NexusLoss: KL inputs must have equal size and positive temperature");
    const auto log_softmax = [temperature](std::span<const T> values) {
        const T maximum = *std::max_element(values.begin(), values.end()) / temperature;
        T exp_sum = T(0);
        for (T value : values) exp_sum += std::exp(value / temperature - maximum);
        const T log_z = maximum + std::log(exp_sum);
        std::vector<T> result(values.size());
        for (size_t i = 0; i < values.size(); ++i) result[i] = values[i] / temperature - log_z;
        return result;
    };
    const auto log_student = log_softmax(student_logits);
    const auto log_teacher = log_softmax(teacher_logits);
    T result = T(0);
    for (size_t i = 0; i < student_logits.size(); ++i) {
        const T q = std::exp(log_teacher[i]);
        result += q * (log_teacher[i] - log_student[i]);
    }
    return result * temperature * temperature;
}

template<typename T>
T feature_distillation_loss(std::span<const T> student_features, std::span<const T> teacher_features) {
    if (student_features.empty() || student_features.size() != teacher_features.size())
        throw std::invalid_argument("NexusLoss: distillation features must have equal non-zero sizes");
    T sum = T(0);
    for (size_t i = 0; i < student_features.size(); ++i) {
        const T error = student_features[i] - teacher_features[i];
        sum += error * error;
    }
    return sum / static_cast<T>(student_features.size());
}

/// @brief dL/d(student logit) = T * (softmax(z_s/T) - softmax(z_t/T)). Hinton et al., 2015.
template<typename T>
std::vector<T> kl_divergence_gradient(std::span<const T> student_logits, std::span<const T> teacher_logits,
                                      T temperature = T(1)) {
    if (student_logits.empty() || student_logits.size() != teacher_logits.size() || temperature <= T(0))
        throw std::invalid_argument("NexusLoss: KL inputs must have equal size and positive temperature");
    const auto probabilities = [temperature](std::span<const T> values) {
        const T maximum = *std::max_element(values.begin(), values.end()) / temperature;
        std::vector<T> result(values.size());
        T sum = T(0);
        for (size_t i = 0; i < values.size(); ++i) sum += result[i] = std::exp(values[i] / temperature - maximum);
        for (T& value : result) value /= sum;
        return result;
    };
    const auto student = probabilities(student_logits);
    const auto teacher = probabilities(teacher_logits);
    std::vector<T> grad(student.size());
    for (size_t i = 0; i < student.size(); ++i) grad[i] = temperature * (student[i] - teacher[i]);
    return grad;
}

template<typename T>
std::vector<T> feature_distillation_gradient(std::span<const T> student_features,
                                             std::span<const T> teacher_features) {
    if (student_features.empty() || student_features.size() != teacher_features.size())
        throw std::invalid_argument("NexusLoss: distillation features must have equal non-zero sizes");
    std::vector<T> grad(student_features.size());
    const T scale = T(2) / static_cast<T>(student_features.size());
    for (size_t i = 0; i < student_features.size(); ++i)
        grad[i] = scale * (student_features[i] - teacher_features[i]);
    return grad;
}

/**
 * @brief Attention transfer: normalize edilmis haritalarin L2 farki.
 * Zagoruyko & Komodakis, Paying More Attention to Attention, 2017.
 * L = 0.5 * || s/||s|| - t/||t|| ||^2
 */
template<typename T>
T attention_transfer_loss(std::span<const T> student, std::span<const T> teacher) {
    if (student.empty() || student.size() != teacher.size())
        throw std::invalid_argument("NexusLoss: attention maps must have equal non-zero sizes");
    auto norm = [](std::span<const T> values) {
        T sum = T(0);
        for (T value : values) sum += value * value;
        return std::sqrt(std::max(sum, core::epsilon<T>));
    };
    const T student_norm = norm(student), teacher_norm = norm(teacher);
    T loss = T(0);
    for (size_t i = 0; i < student.size(); ++i) {
        const T delta = student[i] / student_norm - teacher[i] / teacher_norm;
        loss += delta * delta;
    }
    return T(0.5) * loss;
}

template<typename T>
std::vector<T> attention_transfer_gradient(std::span<const T> student, std::span<const T> teacher) {
    if (student.empty() || student.size() != teacher.size())
        throw std::invalid_argument("NexusLoss: attention maps must have equal non-zero sizes");
    T sum = T(0);
    for (T value : student) sum += value * value;
    const T student_norm = std::sqrt(std::max(sum, core::epsilon<T>));
    T teacher_sum = T(0);
    for (T value : teacher) teacher_sum += value * value;
    const T teacher_norm = std::sqrt(std::max(teacher_sum, core::epsilon<T>));
    std::vector<T> direction(student.size());
    T projection = T(0);
    for (size_t i = 0; i < student.size(); ++i) {
        direction[i] = student[i] / student_norm - teacher[i] / teacher_norm;
        projection += (student[i] / student_norm) * direction[i];
    }
    std::vector<T> grad(student.size());
    for (size_t i = 0; i < student.size(); ++i)
        grad[i] = (direction[i] - (student[i] / student_norm) * projection) / student_norm;
    return grad;
}

/**
 * @class KnowledgeDistillationLoss
 * @brief Sicaklik T ile yumusatilmis ogretmen/ogrenci dagilimlarinin KL'si, T^2 ile olcekli.
 */
template<typename T>
class KnowledgeDistillationLoss {
public:
    explicit KnowledgeDistillationLoss(T temperature = T(2)) : temperature_(temperature) {}
    T forward(std::span<const T> student_logits, std::span<const T> teacher_logits) {
        student_.assign(student_logits.begin(), student_logits.end());
        teacher_.assign(teacher_logits.begin(), teacher_logits.end());
        ready_ = true;
        return kl_divergence(student_logits, teacher_logits, temperature_);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return kl_divergence_gradient<T>(student_, teacher_, temperature_);
    }
private:
    T temperature_;
    std::vector<T> student_, teacher_;
    bool ready_ = false;
};

template<typename T>
class FeatureDistillationLoss {
public:
    T forward(std::span<const T> student, std::span<const T> teacher) {
        student_.assign(student.begin(), student.end());
        teacher_.assign(teacher.begin(), teacher.end());
        ready_ = true;
        return feature_distillation_loss(student, teacher);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return feature_distillation_gradient<T>(student_, teacher_);
    }
private:
    std::vector<T> student_, teacher_;
    bool ready_ = false;
};

template<typename T>
class AttentionTransferLoss {
public:
    T forward(std::span<const T> student, std::span<const T> teacher) {
        student_.assign(student.begin(), student.end());
        teacher_.assign(teacher.begin(), teacher.end());
        ready_ = true;
        return attention_transfer_loss(student, teacher);
    }
    std::vector<T> backward() const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        return attention_transfer_gradient<T>(student_, teacher_);
    }
private:
    std::vector<T> student_, teacher_;
    bool ready_ = false;
};

} // namespace nexusloss::distillation

#endif
