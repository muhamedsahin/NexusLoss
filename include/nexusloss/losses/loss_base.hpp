//
// Created by muham on 24.09.2026.
//

/**
 * @file loss_base.hpp
 * @brief Tüm loss sınıflarının ortak arayüzü ve indirgeme mantığı.
 *
 * NexusLoss tasarımında her loss sınıfı aynı davranışı gösterir:
 *  - element-wise hata üretir,
 *  - ardından reduction uygular,
 *  - istenirse ağırlıklı reduction da destekler.
 *
 * Bu yapı, yeni loss fonksiyonları eklerken tekrar eden kod miktarını azaltır ve API
 * tutarlılığını korur.
 */

#ifndef NEXUSLOSS_LOSS_BASE_HPP
#define NEXUSLOSS_LOSS_BASE_HPP

#include <vector>
#include <span>
#include <stdexcept>
#include "../core/reduction.hpp"
#include "../core/utils.hpp"

namespace nexusloss {

    /**
     * @brief Tüm loss fonksiyonları için temel soyut sınıf.
     *
     * @tparam T Veri tipi (float, double, vb.)
     */
    template<typename T>
    class LossBase {
    protected:
        core::ReductionType reduction_;

    public:
        /**
         * @brief Loss sınıfını belirtilen indirgeme tipiyle oluşturur.
         *
         * @param reduction İndirgeme tipi (Mean, Sum, None)
         */
        //explicit nesneyi tanımlarken LossBase loss(core::ReductionType::Sum) gibi parantezler ile tanımlamamızı sağlar
        //direk classın adını vermemiz class oluşturulurken otonom çalışacak kod olmasını sağlamktır
        // : operatör ile reductiona atama yapmamızın nedeni nesneyi oluşturmadan halletmek bu sayede çok daha hızlı ama istersek nesneyi olutşruduktan sonrada yaparız
        explicit LossBase(core::ReductionType reduction = core::ReductionType::Mean)
            : reduction_(reduction) {}

        virtual ~LossBase() = default;// Oluşturduğumuz nesne ölürken çalışacak olan fonksyon

        /**
         * @brief Her eleman için hata değerini hesaplar (alt sınıflar implemente edecek).
         *
         * @param pred Model tahminleri
         * @param target Gerçek değerler
         * @return std::vector<T> Eleman bazlı hata değerleri
         *
         * not: virtual kısımı bu fonksyonun baş kısımını belirler
         * istersek kendimiz aynı baş kısımını kullanarak başka fonksyonlar türetebiliriz
         */
        virtual std::vector<T> compute_element_wise(
            std::span<const T> pred,
            std::span<const T> target
        ) = 0;

        /**
         * @brief Ana hesaplama fonksiyonu - ağırlıksız versiyon.
         *
         * @param pred Model tahminleri
         * @param target Gerçek değerler
         * @return std::vector<T> İndirgenmiş hata değerleri (Mean/Sum için 1 elemanlı, None için orijinal boyutta)
         */
        std::vector<T> compute(
            std::span<const T> pred,
            std::span<const T> target
        ) {
            // 1. Boyut kontrolü
            core::check_same_size(pred, target);

            // 2. Eleman bazlı hataları hesapla
            std::vector<T> element_wise_losses = compute_element_wise(pred, target);

            // 3. İndirgeme uygula
            return core::apply_reduction(
                std::span<const T>(element_wise_losses),
                reduction_
            );
        }

        /**
         * @brief Ana hesaplama fonksiyonu - ağırlıklı versiyon.
         *
         * @param pred Model tahminleri
         * @param target Gerçek değerler
         * @param weights Her eleman için ağırlık değerleri
         * @return std::vector<T> İndirgenmiş hata değerleri
         */
        std::vector<T> compute(
            std::span<const T> pred,
            std::span<const T> target,
            std::span<const T> weights
        ) {
            // 1. Boyut kontrolü (weights dahil)
            core::check_same_size(pred, target, weights);

            // 2. Eleman bazlı hataları hesapla
            std::vector<T> element_wise_losses = compute_element_wise(pred, target);

            // 3. Ağırlıklı indirgeme uygula
            return core::apply_reduction(
                std::span<const T>(element_wise_losses),
                weights,
                reduction_
            );
        }

        /**
         * @brief İndirgeme tipini değiştirir.
         *
         * @param reduction Yeni indirgeme tipi
         */
        void set_reduction(core::ReductionType reduction) {
            reduction_ = reduction;
        }

        /**
         * @brief Mevcut indirgeme tipini döndürür.
         *
         * @return ReductionType Mevcut indirgeme tipi
         */
        core::ReductionType get_reduction() const {
            return reduction_;
        }

        /**
         * @brief Eleman bazli loss'un tahmine gore turevi (reduction HARIC).
         *
         * Alt sinif bunu doldurur. Donen vektor, pred ile ayni uzunluktadir ve
         * `d L_i / d pred_i` degerini tasir. Ortalama/toplam olcegi
         * `gradient()` icinde uygulanir; boylece turev formulu reduction'dan
         * bagimsiz kalir.
         *
         * Varsayilan: turevi yazilmamis loss'larda acikca hata verir.
         * Zero-one gibi turevsiz olculer 0 subgradyani dokumante ederek override eder.
         */
        virtual std::vector<T> gradient_element_wise(
            std::span<const T> pred,
            std::span<const T> target
        ) const {
            (void)pred;
            (void)target;
            throw std::logic_error("NexusLoss: bu loss icin analitik gradyan tanimli degil");
        }

        /**
         * @brief Skaler (veya None icin eleman bazli) loss'un dL/d(pred) gradyani.
         *
         * Mean / BatchMean: her eleman `1/N` ile olceklenir, cunku skaler loss ortalamadir.
         * Sum: eleman turevi oldugu gibi kalir.
         * None: her eleman kendi kaybinin turevidir; olcekleme yoktur.
         *
         * @warning forward() cagirmak zorunda degildir. Girdileri dogrudan alir.
         *          forward/backward cifti icin asagidaki `backward()` metoduna bak.
         */
        std::vector<T> gradient(
            std::span<const T> pred,
            std::span<const T> target
        ) const {
            core::check_same_size(pred, target);
            std::vector<T> grad = gradient_element_wise(pred, target);
            if (grad.size() != pred.size()) {
                throw std::logic_error("NexusLoss: gradyan uzunlugu tahmin ile ayni olmali");
            }
            scale_gradient(grad, /*weight*/ {});
            return grad;
        }

        /// @brief Agirlikli gradient. Mean'de payda agirlik toplami, Sum/None'da w_i carpanidir.
        std::vector<T> gradient(
            std::span<const T> pred,
            std::span<const T> target,
            std::span<const T> weights
        ) const {
            core::check_same_size(pred, target, weights);
            std::vector<T> grad = gradient_element_wise(pred, target);
            if (grad.size() != pred.size()) {
                throw std::logic_error("NexusLoss: gradyan uzunlugu tahmin ile ayni olmali");
            }
            scale_gradient(grad, weights);
            return grad;
        }

        /**
         * @brief compute() ile ayni degeri uretir ve backward() icin girdileri saklar.
         *
         * NexusTrain her adimda once forward, sonra backward cagirir. Saklanan sey
         * hesaba giren tensordur; agirlik guncellemesi bu sinifin isi degildir.
         */
        std::vector<T> forward(std::span<const T> pred, std::span<const T> target) {
            remember(pred, target, {});
            return compute(pred, target);
        }

        std::vector<T> forward(
            std::span<const T> pred,
            std::span<const T> target,
            std::span<const T> weights
        ) {
            remember(pred, target, weights);
            return compute(pred, target, weights);
        }

        /**
         * @brief Son forward()'un girdilerine gore dL/d(prediction).
         * @warning forward() cagrilmadan backward() tanimsizdir; burada bilinçli hata atilir.
         */
        std::vector<T> backward() const {
            if (!forward_called_) {
                throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
            }
            if (has_weights_) {
                return gradient(cached_prediction_, cached_target_, cached_weights_);
            }
            return gradient(cached_prediction_, cached_target_);
        }

    private:
        void remember(std::span<const T> pred, std::span<const T> target, std::span<const T> weights) {
            cached_prediction_.assign(pred.begin(), pred.end());
            cached_target_.assign(target.begin(), target.end());
            cached_weights_.assign(weights.begin(), weights.end());
            has_weights_ = !weights.empty();
            forward_called_ = true;
        }

        void scale_gradient(std::vector<T>& grad, std::span<const T> weights) const {
            const bool weighted = !weights.empty();
            if (reduction_ == core::ReductionType::None || reduction_ == core::ReductionType::Sum) {
                if (weighted) {
                    for (size_t i = 0; i < grad.size(); ++i) grad[i] *= weights[i];
                }
                return;
            }
            // Mean ve BatchMean
            if (!weighted) {
                const T inv = T(1) / static_cast<T>(grad.size());
                for (T& value : grad) value *= inv;
                return;
            }
            T weight_sum = T(0);
            for (T weight : weights) weight_sum += weight;
            const T denom = std::max(weight_sum, core::epsilon<T>);
            for (size_t i = 0; i < grad.size(); ++i) grad[i] *= weights[i] / denom;
        }

        std::vector<T> cached_prediction_;
        std::vector<T> cached_target_;
        std::vector<T> cached_weights_;
        bool has_weights_ = false;
        bool forward_called_ = false;
    };

} // namespace nexusloss

#endif //NEXUSLOSS_LOSS_BASE_HPP
