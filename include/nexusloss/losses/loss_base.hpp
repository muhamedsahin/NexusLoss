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

#include <atomic>
#include <vector>
#include <span>
#include <stdexcept>
#include "../core/reduction.hpp"
#include "../core/utils.hpp"

namespace nexusloss {

    namespace core {
        /// `gradient_element_wise_into` varsayilan yolu gecici vektor ayirdiginda artar.
        /// Alt sinif dogrudan yazarsa bu sayac yerinde kalir. Test sifir bekler.
        inline std::atomic<std::uint64_t>& inplace_fallback_allocations() {
            static std::atomic<std::uint64_t> count{0};
            return count;
        }
    }

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
         * @brief Eleman turevini `grad_out` icine yazar. Ayirma yapmaz.
         *
         * Reduction bu metoda girmez. Mean/Sum olcegi `gradient_into` ve
         * `backward_into` icinde, ayni tamponun uzerinde uygulanir.
         * Varsayilan uygulama `gradient_element_wise` cagirir ve bu yuzden
         * gecici vektor ayirir; sicak yol bunu override eder.
         */
        virtual void gradient_element_wise_into(
            std::span<const T> pred,
            std::span<const T> target,
            std::span<T> grad_out
        ) const {
            core::inplace_fallback_allocations().fetch_add(1, std::memory_order_relaxed);
            const std::vector<T> temporary = gradient_element_wise(pred, target);
            if (temporary.size() != grad_out.size()) {
                throw std::logic_error("NexusLoss: gradyan uzunlugu tahmin ile ayni olmali");
            }
            for (size_t i = 0; i < temporary.size(); ++i) {
                grad_out[i] = temporary[i];
            }
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
        /// @brief dL/d(pred) degerini `grad_out` icine yazar. Yeni vektor ayirmaz.
        void gradient_into(
            std::span<const T> pred,
            std::span<const T> target,
            std::span<T> grad_out
        ) const {
            core::check_same_size(pred, target);
            require_gradient_span(pred, grad_out);
            gradient_element_wise_into(pred, target, grad_out);
            scale_gradient(grad_out, /*weight*/ {});
        }

        /// @brief Agirlikli `gradient_into`. Mean'de payda agirlik toplami, Sum/None'da w_i carpanidir.
        void gradient_into(
            std::span<const T> pred,
            std::span<const T> target,
            std::span<const T> weights,
            std::span<T> grad_out
        ) const {
            core::check_same_size(pred, target, weights);
            require_gradient_span(pred, grad_out);
            gradient_element_wise_into(pred, target, grad_out);
            scale_gradient(grad_out, weights);
        }

        std::vector<T> gradient(
            std::span<const T> pred,
            std::span<const T> target
        ) const {
            core::check_same_size(pred, target);
            std::vector<T> grad(pred.size());
            gradient_element_wise_into(pred, target, grad);
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
            std::vector<T> grad(pred.size());
            gradient_element_wise_into(pred, target, grad);
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

        /// @brief Son forward()'un tahmin uzunlugu; `backward_into` tamponu bu boyutta olmalidir.
        [[nodiscard]] size_t gradient_size() const noexcept { return cached_prediction_.size(); }

        /**
         * @brief Son forward()'un girdilerine gore dL/d(prediction).
         * @warning forward() cagrilmadan backward() tanimsizdir; burada bilinçli hata atilir.
         */
        std::vector<T> backward() const {
            if (!forward_called_) {
                throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
            }
            std::vector<T> grad(cached_prediction_.size());
            backward_into(grad);
            return grad;
        }

        /**
         * @brief Son forward'un turevini `grad_out` icine yazar.
         *
         * Basarili cagri yigin disinda bellek ayirmaz: eleman turevi dogrudan
         * cagiranin span'ine yazilir, Mean/Sum olcegi ayni bellege uygulanir.
         * `grad_out` uzunlugu saklanan tahmin ile ayni olmalidir.
         */
        void backward_into(std::span<T> grad_out) const {
            if (!forward_called_) {
                throw std::logic_error("NexusLoss: backward() oncesi forward() cagrilmali");
            }
            if (grad_out.size() != cached_prediction_.size()) {
                throw std::invalid_argument("NexusLoss: grad_out boyutu pred ile ayni olmali");
            }
            gradient_element_wise_into(cached_prediction_, cached_target_, grad_out);
            if (has_weights_) {
                scale_gradient(grad_out, cached_weights_);
            } else {
                scale_gradient(grad_out, {});
            }
        }

    protected:
        void require_gradient_span(std::span<const T> pred, std::span<T> grad_out) const {
            if (grad_out.size() != pred.size()) {
                throw std::invalid_argument("NexusLoss: grad_out boyutu pred ile ayni olmali");
            }
        }

    private:
        void remember(std::span<const T> pred, std::span<const T> target, std::span<const T> weights) {
            cached_prediction_.assign(pred.begin(), pred.end());
            cached_target_.assign(target.begin(), target.end());
            cached_weights_.assign(weights.begin(), weights.end());
            has_weights_ = !weights.empty();
            forward_called_ = true;
        }

        void scale_gradient(std::span<T> grad, std::span<const T> weights) const {
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
