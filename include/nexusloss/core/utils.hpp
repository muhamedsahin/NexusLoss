/**
 * @file utils.hpp
 * @brief Sayısal kararlılık ve ortak yardımcı fonksiyonlar.
 *
 * Bu dosya, loss hesaplamalarında sıklıkla ortaya çıkan sayısal sorunları önler:
 * log(0), overflow, softmax ve bölme işlemlerinde sıfıra yaklaşan değerler.
 * Yazılımın her loss sınıfı bu yardımcıları kullanır; böylece model eğitimi sırasında
 * patlama, NaN veya inf gibi problemlerin önüne geçilir.
 *
 * @see reduction.hpp  Mean/Sum/None reduction stratejileri
 * @see loss_base.hpp  tüm loss sınıflarının ortak arayüzü
 */

#ifndef NEXUSLOSS_UTILS_HPP
#define NEXUSLOSS_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>


namespace nexusloss::core {

    // Çok hassas değerleri sıfıra yuvarlamasını engelleyen sabit
    template<typename T>
    inline constexpr T epsilon = static_cast<T>(1e-7);

    // pred ve target'ın boyutları uyuşuyor mu diye bakar.
    // weight opsiyonel: verilmezse kontrol edilmez, verilirse o da aynı boyutta olmalı.
    template<typename T>
    void check_same_size(std::span<const T> pred,
                          std::span<const T> target,
                          std::span<const T> weight = {}) {
        if (pred.size() != target.size()) {
            throw std::invalid_argument("NexusLoss: pred ve target boyutlari ayni degil");
        }
        if (pred.empty()) {
            throw std::invalid_argument("NexusLoss: bos girdi verilemez");
        }
        if (!weight.empty() && weight.size() != pred.size()) {
            throw std::invalid_argument("NexusLoss: weight boyutu pred ile ayni olmali");
        }
    }

    // Bir değeri belirli aralık içinde tutar
    template<typename T>
    T clamp(T val, T lo, T hi) {
        return std::min(std::max(val, lo), hi);
    }

    // log(0) ya da log(negatif) olursa patlamaması icin korumali log
    template<typename T>
    T safe_log(T x) {
        static_assert(std::is_floating_point_v<T>, "NexusLoss: safe_log sadece float/double ile calisir");
        return std::log(std::max(x, epsilon<T>));
    }

    // Ham logit'i [0, 1] olasiliga cevirir. Buyuk |x| degerlerinde exp tasmamasi
    // icin isaret ayrimi yapilir; bu yuzden isim olarak stable_sigmoid de sunulur.
    template<typename T>
    T sigmoid(T x) {
        static_assert(std::is_floating_point_v<T>, "NexusLoss: sigmoid sadece float/double ile calisir");
        if (x >= T(0)) {
            T z = std::exp(-x);
            return T(1) / (T(1) + z);
        }
        T z = std::exp(x);
        return z / (T(1) + z);
    }

    /// @brief `sigmoid` ile ayni, tasmayan formul. Isim, logit tabanli loss'larda niyeti belirtir.
    template<typename T>
    T stable_sigmoid(T x) {
        return sigmoid(x);
    }

    /// @brief Olasiligi (eps, 1-eps) araligina sikistirir; log(0) ve 1-p=0 onlenir.
    template<typename T>
    T clamp_probability(T probability, T eps = epsilon<T>) {
        return clamp(probability, eps, T(1) - eps);
    }

    /**
     * @brief Tasmasiz log-sum-exp.
     *
     * softmax ve cross-entropy paydasinda `sum(exp(x))` buyuk logit'lerde inf olur.
     * Once maksimum cikarilir:
     *   log(sum exp(x_i)) = m + log(sum exp(x_i - m)),  m = max(x)
     */
    template<typename T>
    T log_sum_exp(std::span<const T> values) {
        static_assert(std::is_floating_point_v<T>, "NexusLoss: log_sum_exp sadece float/double ile calisir");
        if (values.empty()) {
            throw std::invalid_argument("NexusLoss: bos liste ile log_sum_exp alinamaz");
        }
        const T maximum = *std::max_element(values.begin(), values.end());
        T sum = T(0);
        for (T value : values) sum += std::exp(value - maximum);
        return maximum + std::log(std::max(sum, epsilon<T>));
    }

    /**
     * @brief Sinif indekslerini one-hot satirlara acar (bellekte duz, sinif-major degil ornek-major).
     * @param indices     N adet sinif indeksi
     * @param num_classes one-hot genisligi
     * @return            uzunlugu N * num_classes olan vektor; her ornek kendi satirinda
     */
    template<typename T>
    std::vector<T> one_hot(std::span<const std::size_t> indices, std::size_t num_classes) {
        if (num_classes == 0) {
            throw std::invalid_argument("NexusLoss: one_hot sinif sayisi sifir olamaz");
        }
        std::vector<T> encoded(indices.size() * num_classes, T(0));
        for (std::size_t row = 0; row < indices.size(); ++row) {
            if (indices[row] >= num_classes) {
                throw std::out_of_range("NexusLoss: one_hot sinif indeksi aralik disi");
            }
            encoded[row * num_classes + indices[row]] = T(1);
        }
        return encoded;
    }

    /**
     * @brief Padding veya ignore maskesini eleman bazinda uygular.
     *
     * Maske 0 olan konumlar loss'a girmez (carpim 0). Maske ve degerler ayni uzunlukta olmalidir.
     * Sequence loss'larinda pad token'i bu yardimci ile disari atilir.
     */
    template<typename T>
    std::vector<T> apply_mask(std::span<const T> values, std::span<const T> mask) {
        if (values.size() != mask.size()) {
            throw std::invalid_argument("NexusLoss: maske ile degerler ayni uzunlukta olmali");
        }
        std::vector<T> masked(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) masked[i] = values[i] * mask[i];
        return masked;
    }

    // Bir liste (logit) alir, olasilik dagilimina cevirir (toplamlari 1 olur)
    // Tasmayi onlemek icin once en buyuk degeri her elemandan cikarir
    template<typename T>
    std::vector<T> softmax(std::span<const T> logits) {
        static_assert(std::is_floating_point_v<T>, "NexusLoss: softmax sadece float/double ile calisir");
        if (logits.empty()) {
            throw std::invalid_argument("NexusLoss: bos liste ile softmax alinamaz");
        }

        T max_val = *std::max_element(logits.begin(), logits.end());

        std::vector<T> result(logits.size());
        T sum = T(0);
        for (size_t i = 0; i < logits.size(); ++i) {
            result[i] = std::exp(logits[i] - max_val);
            sum += result[i];
        }

        sum = std::max(sum, epsilon<T>);
        for (T& v : result) {
            v /= sum;
        }
        return result;
    }

    // Karekök, negatif sayida patlamamasi icin korumali
    template<typename T>
    T safe_sqrt(T x) {
        return std::sqrt(std::max(x, T(0)));
    }

    // Dice Loss'ta payda sifir olabilir, bunun icin korumali bolme
    template<typename T>
    T safe_divide(T numerator, T denominator) {
        return numerator / std::max(denominator, epsilon<T>);
    }

}  // namespace nexusloss::core

#endif //NEXUSLOSS_UTILS_HPP