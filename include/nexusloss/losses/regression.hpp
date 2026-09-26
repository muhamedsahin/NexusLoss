//
// Created by muham on 24.09.2026.
//

/**
 * @file regression.hpp
 * @brief Regresyon loss aileleri: klasik, robust ve yarı-robust yöntemler.
 *
 * Bu dosya, MSE/MAE gibi geleneksel loss'ları ve Huber, Log-Cosh, Quantile, Tweedie,
 * Poisson, Cauchy, Charbonnier ve Tukey gibi robust loss'ları içerir.
 */

#ifndef NEXUSLOSS_REGRESSION_HPP
#define NEXUSLOSS_REGRESSION_HPP

#include <vector>
#include <span>
#include <cmath>
#include <stdexcept>
#include "loss_base.hpp"

namespace nexusloss {

    /**
     * @brief Mean Squared Error (MSE) Loss fonksiyonu.
     *
     * Formül: (pred - target)²
     *
     * @tparam T Veri tipi (float, double, vb.)
     */
    template<typename T>
    class MSELoss : public LossBase<T> {
    public:
        /**
         * @brief MSELoss yapıcı metodu.
         * @param reduction İndirgeme tipi (Varsayılan: Mean)
         */
        explicit MSELoss(core::ReductionType reduction = core::ReductionType::Mean)
            : LossBase<T>(reduction) {}

        /**
         * @brief Her eleman için MSE hatasını hesaplar.
         *
         * @param pred Model tahminleri
         * @param target Gerçek değerler
         * @return std::vector<T> Eleman bazlı karesel hatalar
         */
        std::vector<T> compute_element_wise(
            std::span<const T> pred,
            std::span<const T> target
        ) override {
            std::vector<T> losses(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                T diff = pred[i] - target[i];
                losses[i] = diff * diff; // (pred - target)²
            }
            return losses;
        }

        // d/dx (x-y)^2 = 2(x-y). Mean reduction bunu ayrica 1/N ile carpar.
        std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
            std::vector<T> grad(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) grad[i] = T(2) * (pred[i] - target[i]);
            return grad;
        }
    };

    /**
     * @brief Mean Absolute Error (MAE) Loss fonksiyonu (L1 Loss).
     *
     * Formül: |pred - target|
     *
     * @tparam T Veri tipi (float, double, vb.)
     */
    template<typename T>
    class MAELoss : public LossBase<T> {
    public:
        /**
         * @brief MAELoss yapıcı metodu.
         * @param reduction İndirgeme tipi (Varsayılan: Mean)
         */
        explicit MAELoss(core::ReductionType reduction = core::ReductionType::Mean)
            : LossBase<T>(reduction) {}

        /**
         * @brief Her eleman için MAE hatasını hesaplar.
         *
         * @param pred Model tahminleri
         * @param target Gerçek değerler
         * @return std::vector<T> Eleman bazlı mutlak hatalar
         */
        std::vector<T> compute_element_wise(
            std::span<const T> pred,
            std::span<const T> target
        ) override {
            std::vector<T> losses(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                // std::abs mutlak değer alır
                losses[i] = std::abs(pred[i] - target[i]);
            }
            return losses;
        }

        // Subgradyan: sign(x-y). Tam esitlikte 0 secilir (simetrik subgradyan).
        std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
            std::vector<T> grad(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T diff = pred[i] - target[i];
                grad[i] = diff > T(0) ? T(1) : (diff < T(0) ? T(-1) : T(0));
            }
            return grad;
        }
    };

    template<typename T>
    class HuberLoss : public LossBase<T> {
        T delta_;
    public:
        explicit HuberLoss(T delta = T(1), core::ReductionType reduction = core::ReductionType::Mean)
            : LossBase<T>(reduction), delta_(delta) {
            if (!(delta > T(0))) throw std::invalid_argument("NexusLoss: Huber delta must be positive");
        }
        std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
            std::vector<T> losses(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T error = std::abs(pred[i] - target[i]);
                losses[i] = error <= delta_ ? T(0.5) * error * error : delta_ * (error - T(0.5) * delta_);
            }
            return losses;
        }

        // |hata| <= delta iken turev ham hata (L2 bolgesi), disarida delta * sign (L1 bolgesi).
        std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
            std::vector<T> grad(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T diff = pred[i] - target[i];
                const T magnitude = std::abs(diff);
                grad[i] = magnitude <= delta_ ? diff : (diff > T(0) ? delta_ : -delta_);
            }
            return grad;
        }
    };

    template<typename T>
    class LogCoshLoss : public LossBase<T> {
    public:
        using LossBase<T>::LossBase;
        std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
            std::vector<T> losses(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T error = std::abs(pred[i] - target[i]);
                losses[i] = error + std::log1p(std::exp(T(-2) * error)) - std::log(T(2));
            }
            return losses;
        }

        // d/dx log(cosh(x-y)) = tanh(x-y). Buyuk |x| icin tanh isaretine doygunlasir, patlamaz.
        std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
            std::vector<T> grad(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T diff = pred[i] - target[i];
                const T halved = std::exp(T(-2) * std::abs(diff));
                const T tanh_abs = (T(1) - halved) / (T(1) + halved);
                grad[i] = diff >= T(0) ? tanh_abs : -tanh_abs;
            }
            return grad;
        }
    };

    template<typename T>
    class MSLELoss : public LossBase<T> {
    public:
        using LossBase<T>::LossBase;
        std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
            std::vector<T> losses(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                if (pred[i] < T(0) || target[i] < T(0))
                    throw std::invalid_argument("NexusLoss: MSLE requires non-negative predictions and targets");
                const T error = std::log1p(pred[i]) - std::log1p(target[i]);
                losses[i] = error * error;
            }
            return losses;
        }

        // L = (log1p(x) - log1p(y))^2  =>  dL/dx = 2 * hata / (1+x)
        std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
            std::vector<T> grad(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                if (pred[i] < T(0) || target[i] < T(0))
                    throw std::invalid_argument("NexusLoss: MSLE requires non-negative predictions and targets");
                const T error = std::log1p(pred[i]) - std::log1p(target[i]);
                grad[i] = T(2) * error / (T(1) + pred[i]);
            }
            return grad;
        }
    };

    template<typename T>
    class QuantileLoss : public LossBase<T> {
        T quantile_;
    public:
        explicit QuantileLoss(T quantile, core::ReductionType reduction = core::ReductionType::Mean)
            : LossBase<T>(reduction), quantile_(quantile) {
            if (!(quantile > T(0) && quantile < T(1)))
                throw std::invalid_argument("NexusLoss: quantile must be between 0 and 1");
        }
        std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
            std::vector<T> losses(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T error = target[i] - pred[i];
                losses[i] = error >= T(0) ? quantile_ * error : (quantile_ - T(1)) * error;
            }
            return losses;
        }

        // hata = y - x. x < y iken dL/dx = -q (alti tahmini q kadar cezalandir),
        // x > y iken dL/dx = 1-q (ustu tahmini 1-q kadar cezalandir).
        std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
            std::vector<T> grad(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                grad[i] = target[i] >= pred[i] ? -quantile_ : (T(1) - quantile_);
            }
            return grad;
        }
    };

    template<typename T>
    class PoissonLoss : public LossBase<T> {
    public:
        using LossBase<T>::LossBase;
        // Predictions are log rates, as in the canonical Poisson deviance objective.
        std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
            std::vector<T> losses(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                if (target[i] < T(0)) throw std::invalid_argument("NexusLoss: Poisson targets must be non-negative");
                losses[i] = std::exp(pred[i]) - target[i] * pred[i];
            }
            return losses;
        }

        // L = exp(x) - y*x  (x = log oran). dL/dx = exp(x) - y
        std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
            std::vector<T> grad(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                if (target[i] < T(0)) throw std::invalid_argument("NexusLoss: Poisson targets must be non-negative");
                grad[i] = std::exp(pred[i]) - target[i];
            }
            return grad;
        }
    };

    /// @brief PoissonLoss ile ayni. Isim, sayim verisi NLL sozlesmesini acikca soyler.
    template<typename T>
    using PoissonNLLLoss = PoissonLoss<T>;

    template<typename T>
    class TweedieLoss : public LossBase<T> {
        T power_;
    public:
        explicit TweedieLoss(T power, core::ReductionType reduction = core::ReductionType::Mean)
            : LossBase<T>(reduction), power_(power) {
            if (!(power > T(1) && power < T(2)))
                throw std::invalid_argument("NexusLoss: Tweedie power must be between 1 and 2");
        }
        // Predictions are positive means; this is the unit-dispersion Tweedie deviance.
        std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
            std::vector<T> losses(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T mean = pred[i], y = target[i];
                if (!(mean > T(0)) || y < T(0))
                    throw std::invalid_argument("NexusLoss: Tweedie requires positive predictions and non-negative targets");
                losses[i] = T(2) * (
                    (y == T(0) ? T(0) : std::pow(y, T(2) - power_))
                        / ((T(1) - power_) * (T(2) - power_))
                    - y * std::pow(mean, T(1) - power_) / (T(1) - power_)
                    + std::pow(mean, T(2) - power_) / (T(2) - power_)
                );
            }
            return losses;
        }

        // Unit-dispersion deviance'in ortalamaya gore turevi: 2 * mean^{-p} * (mean - y)
        std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
            std::vector<T> grad(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T mean = pred[i], y = target[i];
                if (!(mean > T(0)) || y < T(0))
                    throw std::invalid_argument("NexusLoss: Tweedie requires positive predictions and non-negative targets");
                grad[i] = T(2) * std::pow(mean, -power_) * (mean - y);
            }
            return grad;
        }
    };

    template<typename T>
    class CauchyLoss : public LossBase<T> {
        T scale_;
    public:
        explicit CauchyLoss(T scale = T(1), core::ReductionType reduction = core::ReductionType::Mean)
            : LossBase<T>(reduction), scale_(scale) {
            if (!(scale > T(0))) throw std::invalid_argument("NexusLoss: Cauchy scale must be positive");
        }
        std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
            std::vector<T> losses(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T error = (pred[i] - target[i]) / scale_;
                losses[i] = T(0.5) * scale_ * scale_ * std::log1p(error * error);
            }
            return losses;
        }

        // L = 0.5 * s^2 * log(1+(e/s)^2), dL/de = e / (1 + (e/s)^2)
        std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
            std::vector<T> grad(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T diff = pred[i] - target[i];
                const T scaled = diff / scale_;
                grad[i] = diff / (T(1) + scaled * scaled);
            }
            return grad;
        }
    };

    template<typename T>
    class CharbonnierLoss : public LossBase<T> {
        T epsilon_;
    public:
        explicit CharbonnierLoss(T epsilon = T(1e-3), core::ReductionType reduction = core::ReductionType::Mean)
            : LossBase<T>(reduction), epsilon_(epsilon) {
            if (!(epsilon > T(0))) throw std::invalid_argument("NexusLoss: Charbonnier epsilon must be positive");
        }
        std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
            std::vector<T> losses(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T error = pred[i] - target[i];
                losses[i] = std::sqrt(error * error + epsilon_ * epsilon_);
            }
            return losses;
        }

        std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
            std::vector<T> grad(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T diff = pred[i] - target[i];
                grad[i] = diff / std::sqrt(diff * diff + epsilon_ * epsilon_);
            }
            return grad;
        }
    };

    template<typename T>
    class TukeyBiweightLoss : public LossBase<T> {
        T scale_;
    public:
        explicit TukeyBiweightLoss(T scale = T(4.685), core::ReductionType reduction = core::ReductionType::Mean)
            : LossBase<T>(reduction), scale_(scale) {
            if (!(scale > T(0))) throw std::invalid_argument("NexusLoss: Tukey scale must be positive");
        }
        std::vector<T> compute_element_wise(std::span<const T> pred, std::span<const T> target) override {
            std::vector<T> losses(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T error = std::abs(pred[i] - target[i]) / scale_;
                if (error < T(1)) {
                    const T inner = T(1) - error * error;
                    losses[i] = scale_ * scale_ * (T(1) - inner * inner * inner) / T(6);
                } else {
                    losses[i] = scale_ * scale_ / T(6);
                }
            }
            return losses;
        }

        // Olcegin icinde psi turevi e*(1-r^2)^2, disarida Influence sifirlanir (redescending).
        std::vector<T> gradient_element_wise(std::span<const T> pred, std::span<const T> target) const override {
            std::vector<T> grad(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) {
                const T diff = pred[i] - target[i];
                const T ratio = std::abs(diff) / scale_;
                if (ratio < T(1)) {
                    const T inner = T(1) - ratio * ratio;
                    grad[i] = diff * inner * inner;
                } else {
                    grad[i] = T(0);
                }
            }
            return grad;
        }
    };

} // namespace nexusloss

#endif // NEXUSLOSS_REGRESSION_HPP