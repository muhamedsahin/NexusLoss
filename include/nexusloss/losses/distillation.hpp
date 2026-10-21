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

/**
 * @brief dL/d(student logit) = T * (softmax(z_s/T) - softmax(z_t/T)). Hinton et al., 2015.
 * `teacher_scratch` ogretmen dagilimi icin cagiranin tamponudur.
 */
template<typename T>
void kl_divergence_gradient_into(std::span<const T> student_logits, std::span<const T> teacher_logits,
                                 std::span<T> grad_out, std::span<T> teacher_scratch, T temperature = T(1)) {
    if (student_logits.empty() || student_logits.size() != teacher_logits.size() || temperature <= T(0))
        throw std::invalid_argument("NexusLoss: KL inputs must have equal size and positive temperature");
    if (grad_out.size() != student_logits.size() || teacher_scratch.size() != student_logits.size())
        throw std::invalid_argument("NexusLoss: distillation buffers must match the logits");
    core::softmax_into<T>(student_logits, grad_out, temperature);
    core::softmax_into<T>(teacher_logits, teacher_scratch, temperature);
    for (size_t i = 0; i < grad_out.size(); ++i) grad_out[i] = temperature * (grad_out[i] - teacher_scratch[i]);
}

template<typename T>
std::vector<T> kl_divergence_gradient(std::span<const T> student_logits, std::span<const T> teacher_logits,
                                      T temperature = T(1)) {
    std::vector<T> grad(student_logits.size()), scratch(student_logits.size());
    kl_divergence_gradient_into<T>(student_logits, teacher_logits, grad, scratch, temperature);
    return grad;
}

template<typename T>
void feature_distillation_gradient_into(std::span<const T> student_features, std::span<const T> teacher_features,
                                        std::span<T> grad_out) {
    if (student_features.empty() || student_features.size() != teacher_features.size())
        throw std::invalid_argument("NexusLoss: distillation features must have equal non-zero sizes");
    if (grad_out.size() != student_features.size())
        throw std::invalid_argument("NexusLoss: distillation grad_out must match the features");
    const T scale = T(2) / static_cast<T>(student_features.size());
    for (size_t i = 0; i < student_features.size(); ++i)
        grad_out[i] = scale * (student_features[i] - teacher_features[i]);
}

template<typename T>
std::vector<T> feature_distillation_gradient(std::span<const T> student_features,
                                             std::span<const T> teacher_features) {
    std::vector<T> grad(student_features.size());
    feature_distillation_gradient_into<T>(student_features, teacher_features, grad);
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

/**
 * @brief Attention transfer turevi, iki gecis, ayirmasiz.
 * d = s/|s| - t/|t|, grad = (d - (s/|s|) <s/|s|, d>) / |s|. Yon vektoru saklanmaz.
 */
template<typename T>
void attention_transfer_gradient_into(std::span<const T> student, std::span<const T> teacher, std::span<T> grad_out) {
    if (student.empty() || student.size() != teacher.size())
        throw std::invalid_argument("NexusLoss: attention maps must have equal non-zero sizes");
    if (grad_out.size() != student.size())
        throw std::invalid_argument("NexusLoss: attention grad_out must match the student map");
    T sum = T(0), teacher_sum = T(0);
    for (size_t i = 0; i < student.size(); ++i) {
        sum += student[i] * student[i];
        teacher_sum += teacher[i] * teacher[i];
    }
    const T inverse_student = T(1) / std::sqrt(std::max(sum, core::epsilon<T>));
    const T inverse_teacher = T(1) / std::sqrt(std::max(teacher_sum, core::epsilon<T>));
    T projection = T(0);
    for (size_t i = 0; i < student.size(); ++i) {
        const T unit = student[i] * inverse_student;
        projection += unit * (unit - teacher[i] * inverse_teacher);
    }
    for (size_t i = 0; i < student.size(); ++i) {
        const T unit = student[i] * inverse_student;
        const T direction = unit - teacher[i] * inverse_teacher;
        grad_out[i] = (direction - unit * projection) * inverse_student;
    }
}

template<typename T>
std::vector<T> attention_transfer_gradient(std::span<const T> student, std::span<const T> teacher) {
    std::vector<T> grad(student.size());
    attention_transfer_gradient_into<T>(student, teacher, grad);
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
        scratch_.resize(student_logits.size());
        ready_ = true;
        return kl_divergence(student_logits, teacher_logits, temperature_);
    }
    [[nodiscard]] size_t gradient_size() const noexcept { return student_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    /// @note Ogretmen dagilimi `scratch_` icine yazilir; ayni nesneye es zamanli backward cagrilmaz.
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        kl_divergence_gradient_into<T>(student_, teacher_, grad_out, scratch_, temperature_);
    }
private:
    T temperature_;
    std::vector<T> student_, teacher_;
    mutable std::vector<T> scratch_;
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
    [[nodiscard]] size_t gradient_size() const noexcept { return student_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        feature_distillation_gradient_into<T>(student_, teacher_, grad_out);
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
    [[nodiscard]] size_t gradient_size() const noexcept { return student_.size(); }
    std::vector<T> backward() const {
        std::vector<T> grad(gradient_size());
        backward_into(grad);
        return grad;
    }
    void backward_into(std::span<T> grad_out) const {
        if (!ready_) throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
        attention_transfer_gradient_into<T>(student_, teacher_, grad_out);
    }
private:
    std::vector<T> student_, teacher_;
    bool ready_ = false;
};

} // namespace nexusloss::distillation

#endif
