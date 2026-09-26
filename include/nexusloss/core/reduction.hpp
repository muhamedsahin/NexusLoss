//
// Created by muham on 24.09.2026.
//

/**
 * @file reduction.hpp
 * @brief Loss değerlerini tek bir skaler veya eleman bazlı vektöre indirgeme stratejileri.
 *
 * Bir model bir batch içinde birden fazla örnek üretirse, her örnek için ayrı hata değeri
 * doğar. Eğitimde çoğu zaman tek bir scalar loss gerektiği için bu değerler mean/sum/
 * none biçiminde indirgenir. Ayrıca örneklere göre ağırlık vermek için weighted reduction
 * desteği de sunulur.
 */

#ifndef NEXUSLOSS_REDUCTION_HPP
#define NEXUSLOSS_REDUCTION_HPP

#include <vector>
#include <span>
#include <numeric>
#include <algorithm>
#include <stdexcept>

// utils.hpp dosyasındaki epsilon<T> sabitini kullanabilmek için
#include "utils.hpp"

namespace nexusloss::core {
    /**
     * @brief Loss sonucunun ornekler uzerinde nasil ozetlenecegini belirler.
     *
     * Mean      Tum elemanlarin ortalamasi (varsayilan).
     * Sum       Toplam. Gradyan olcegi Mean'den N kat buyuktur.
     * None      Eleman bazli tensor; ozet yok.
     * BatchMean Duz vektorde Mean ile ayni. Satir-bazli loss'larda
     *           (KL gibi) cagiran, batch boyutuna bolmeyi kendisi yapar;
     *           burada ayrintinin ismini korumak icin vardir.
     */
    enum class ReductionType {
        Mean,
        Sum,
        None,
        BatchMean
    };

    /**
     * @brief Ağırlıksız (unweighted) indirgeme işlemi yapar.
     *
     * @param losses İndirgenecek hata değerleri (1 boyutlu span)
     * @param reduction İndirgeme türü (Mean, Sum, None)
     * @return std::vector<T> Mean veya Sum için 1 elemanlı, None için aynı boyutta vektör.
     *
     * not: bu fonksyonda tek bir hata sayısı dönecektir fakat bu durumda döneceği hata ağırlıklardan
     *      bağımsızdır tamamen her nöronun çıktısının hata değeri aynı öneme sahiptir bu şekilde
     */
    template<typename T>
    std::vector<T> apply_reduction(std::span<const T> losses, ReductionType reduction) {
        switch (reduction) {
            case ReductionType::Mean:
            case ReductionType::BatchMean: {
                if (losses.empty()) return {T(0)};
                T sum = std::accumulate(losses.begin(), losses.end(), T(0));
                return {sum / static_cast<T>(losses.size())};
            }
            case ReductionType::Sum: {
                T sum = std::accumulate(losses.begin(), losses.end(), T(0));
                return {sum};
            }
            case ReductionType::None:
            default:
                return std::vector<T>(losses.begin(), losses.end());
        }
    }

    /**
     * @brief Ağırlıklı (weighted) indirgeme işlemi yapar.
     *
     * @param losses İndirgenecek hata değerleri
     * @param weights Her hata değerine karşılık gelen ağırlıklar
     * @param reduction İndirgeme türü
     * @return std::vector<T> İndirgenmiş sonuçlar.
     */
    template<typename T>
    std::vector<T> apply_reduction(std::span<const T> losses, std::span<const T> weights, ReductionType reduction) {
        if (losses.size() != weights.size()) {
            throw std::invalid_argument("NexusLoss: İndirgeme için losses ve weights boyutları eşit olmalıdır.");
        }

        switch (reduction) {
            case ReductionType::Mean:
            case ReductionType::BatchMean: {
                if (losses.empty()) return {T(0)};
                T weighted_sum = T(0);
                T weight_sum = T(0);
                for (size_t i = 0; i < losses.size(); ++i) {
                    weighted_sum += losses[i] * weights[i];
                    weight_sum += weights[i];
                }
                // Paydanın sıfır olmasını engellemek için utils.hpp'deki epsilon kullanılıyor
                T denom = std::max(weight_sum, epsilon<T>);
                return {weighted_sum / denom};
            }
            case ReductionType::Sum: {
                T weighted_sum = T(0);
                for (size_t i = 0; i < losses.size(); ++i) {
                    weighted_sum += losses[i] * weights[i];
                }
                return {weighted_sum};
            }
            case ReductionType::None:
            default: {
                std::vector<T> result(losses.size());
                for (size_t i = 0; i < losses.size(); ++i) {
                    result[i] = losses[i] * weights[i];
                }
                return result;
            }
        }
    }
}

#endif //NEXUSLOSS_REDUCTION_HPP
