# NexusLoss

Bu proje, Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi **Muhammed Fatih Şahin** tarafından geliştirilmiştir.

NexusLoss, eğitim adımında tahmin ile hedef arasındaki hatayı ve bu hatanın tahmine göre türevini üreten bir C++20 kayıp kütüphanesidir. Sürüm **0.1.0**. Dağıtım biçimi header-only'dir: derlenecek `.cpp` yoktur, CMake hedefi `nexusloss::nexusloss` bir `INTERFACE` kitaplıktır. Çalışma anında üçüncü parti bağımlılık yoktur. Hesap CPU üzerinde, `float` ve `double` için şablonla, analitik gradyanla yürür.

Ağırlık güncellemesi NexusOptim'e, veri NexusData'ya, katman NexusModel'e, adımın sırası NexusTrain'e aittir. MatrixFlash Pro (`matrix_pro`) kendi autograd grafiğini tutar. NexusLoss o grafa bağlanmaz. Üst katman, dönen skaler ve gradyanı isterse `matrix_pro::custom_loss` içine koyar.

Dokümantasyon sitesi aynı depoda, `docs/website` altındadır. Teknik referansın basılı kopyası `docs/NexusLoss.pdf` dosyasıdır. Bu dosyanın kaynak sayfası `docs/NexusLoss.html`dir.

## İçindekiler

- [Ne işe yarar](#ne-işe-yarar)
- [İçinde ne var](#içinde-ne-var)
- [Performans mimarisi](#performans-mimarisi)
- [Klasör ve dosya yapısı](#klasör-ve-dosya-yapısı)
- [Kurulum](#kurulum)
- [Hızlı başlangıç](#hızlı-başlangıç)
- [Sözleşme](#sözleşme)
- [İndirgeme ve ağırlık](#indirgeme-ve-ağırlık)
- [Sayısal kararlılık](#sayısal-kararlılık)
- [Hangi kaybı seçmeli](#hangi-kaybı-seçmeli)
- [Kayıp kataloğu](#kayıp-kataloğu)
- [Şekil ve kodlama kuralları](#şekil-ve-kodlama-kuralları)
- [Test](#test)
- [Dokümantasyon sitesi](#dokümantasyon-sitesi)
- [Sık karşılaşılan hatalar](#sık-karşılaşılan-hatalar)

## Ne işe yarar

Bir eğitim adımı dört parçaya ayrılır. Veri gelir, model bir tahmin üretir, kayıp o tahmini hedefle karşılaştırır, eniyileyici ağırlığı gradyan yönünde günceller. NexusLoss yalnızca üçüncü parçadır.

Çağıran taraf şunları verir:

- tahmin (`std::span<const T>`),
- hedef (aynı uzunlukta span, ya da sınıf kayıplarında bir sınıf indeksi),
- isteğe bağlı örnek ağırlığı.

Kütüphane şunları döndürür:

- skaler ya da eleman bazlı kayıp değeri,
- tahminle aynı uzunlukta `dL / d(prediction)` vektörü.

Dönen vektör bir autograd düğümü değildir. Çağıran bu vektörü modele geri yayar. Kütüphane ağırlık tensörünü görmez, öğrenme hızı uygulamaz, veri yüklemez.

İki hesap şekli vardır.

**Eleman bazlı kayıplar** `LossBase<T>` üzerinden gelir. Regresyon, ikili çapraz entropi, hinge ailesi bu gruptadır. `compute_element_wise` her konum için bir hata üretir, `core::apply_reduction` bunu özetler, `gradient` aynı özete uygun ölçeği gradyana uygular.

**Yapısal kayıplar** `LossBase`tan türemez. Kutu, CTC, nokta bulutu, liste sıralaması ve tek örnekli çok sınıflı çapraz entropi kendi `forward` / `backward` imzasını kullanır. Zorla düz vektöre indirgenmedikleri için şekil bilgisi (zaman, sınıf, kutu köşesi, nokta boyutu) kaybolmaz.

## İçinde ne var

On üç aile, başlık dosyalarında sınıf ve serbest fonksiyon olarak durur. Dokümantasyon sitesi bu kümeyi **88 kayıt** olarak indeksler. Kayıtların bir kısmı sınıf, bir kısmı aynı formülün serbest fonksiyonudur. Asıl kaynak `include/nexusloss/losses/` altındaki başlıklardır.

| Aile | Ad alanı | Başlık | Ne ölçer |
| --- | --- | --- | --- |
| Regresyon | `nexusloss` | `losses/regression.hpp` | Sürekli tahmin ile hedef arasındaki uzaklık |
| Sınıflandırma | `nexusloss` ve `nexusloss::classification` | `losses/classification.hpp` | Olasılık veya logit ile sınıf etiketi |
| Segmentasyon | `nexusloss::segmentation` | `losses/segmentation.hpp` | Piksel maskelerinin örtüşmesi ve sınırı |
| Tespit | `nexusloss::detection` | `losses/detection.hpp` | Kutu örtüşmesi, odak kaybı, kutu regresyonu |
| Metrik öğrenme | `nexusloss::metric` | `losses/metric.hpp` | Gömme uzayında yakınlık ve açısal marj |
| Sıralama | `nexusloss::ranking` | `losses/ranking.hpp` | Çift, liste ve yaklaşık NDCG |
| Üretken modeller | `nexusloss::generative` | `losses/generative.hpp` | Çekişmeli kayıp, ELBO, algısal ve difüzyon |
| Öz-denetimli | `nexusloss::self_supervised` | `losses/self_supervised.hpp` | Görüntü çiftleri ve çapraz korelasyon |
| Damıtım | `nexusloss::distillation` | `losses/distillation.hpp` | Öğretmen-öğrenci logit ve öznitelik |
| Dizi | `nexusloss::sequence` | `losses/sequence.hpp` | CTC, maskeli token, sonraki cümle |
| Nokta bulutu | `nexusloss::point_cloud` | `losses/point_cloud.hpp` | Chamfer ve eşit kardinaliteli EMD |
| Pekiştirmeli öğrenme | `nexusloss::reinforcement` | `losses/reinforcement.hpp` | Politika gradyanı, PPO, değer, entropi |
| Sağkalım | `nexusloss::survival` | `losses/survival.hpp` | Cox kısmi olabilirlik, Weibull NLL |

Çekirdek iki dosyadır.

- `core/reduction.hpp` — `Mean`, `Sum`, `None`, `BatchMean`.
- `core/utils.hpp` — boyut kontrolü, `epsilon` (`1e-7`), `safe_log`, taşmayan sigmoid, `log_sum_exp`, `softmax`, `one_hot`, maske, `safe_sqrt`, `safe_divide`.

Tek giriş noktası `nexusloss/nexusloss.hpp`dir. Bu başlık hesap yapmaz; çekirdeği ve on üç aileyi birlikte açar.

## Performans mimarisi

Hız, ayrı bir iş parçacığı havuzundan veya elle yazılmış SIMD çekirdeğinden gelmez. Döngüler skaler ve taşınabilirdir. `-O3` veya MSVC eşdeğeri altında derleyici bunları otomatik vektörleyebilir; kütüphane buna söz vermez. Tasarımın kazancı bağımlılıksız, satır içi ve tahminsiz bir sıcak yoldur.

1. **Header-only şablon.** `float` ve `double` ayrı derlenir, aralarında gizli dönüşüm yoktur. Sanal çağrı eleman başına değildir. `LossBase` sanal fonksiyonu vektörün tamamı için bir kez çalışır; asıl aritmetik türetilmiş sınıftaki düz döngüdedir.
2. **`std::span` sözleşmesi.** `compute` ve `gradient` çağıranın tamponunu kopyalamadan okur. Sahiplik çağıranda kalır. `vector`, `array` veya yeterli süre yaşayan bir C dizisi span'e örtük dönüşür.
3. **Analitik gradyan.** Eğitim yolunda sonlu fark yoktur. Sonlu fark yalnızca testtedir (`tests/grad_check.hpp`, adım `1e-6`, tolerans `1e-5`). Analitik formül, her koordinat için iki ek ileri geçişin maliyetini eğitimden çıkarır.
4. **Autograd şeridi yok.** Ara tensör, düğüm ve topoloji sıralaması tutulmaz. Bellek, giriş boyutu ve çıkan gradyan kadardır. Bunun bedeli şudur: kütüphane modeli tanımaz, gradyanı katmanlara kendisi dağıtmaz.
5. **`forward` önbelleği bilinçli bir kopyadır.** `forward` tahmin, hedef ve varsa ağırlığı içeride saklar ki `backward()` argümansız çağrılabilsin. NexusTrain adımı bu çifti kullanır. Kopya istemeyen çağrı `gradient(pred, target)`dir; `forward` şart değildir. `backward()`, `forward` olmadan `std::logic_error` atar.
6. **İki API şekli.** Kutu, CTC ve nokta bulutunu eleman-bazlı tabana zorlamak şekli düzleştirir ve gereksiz bellek trafiği üretir. Bu kayıplar skaler artı şekilli gradyan döndürür.
7. **Sayısal taşma eğitimi durdurur.** `inf` ve `NaN` bir adımı çöpe çevirir, sonra da eniyileyiciyi. `log_sum_exp` paydadan önce maksimumu çıkarır. `sigmoid` büyük pozitif ve negatif logitlerde ayrı dal kullanır. Log ve bölme `epsilon` tabanına oturur. Bu dallar birkaç karşılaştırma ekler; karşılığında kayıp sonlu kalır.
8. **İndirgeme ölçeği öğrenme hızına bağlıdır.** `Mean` gradyanı `1/N` ile böler, böylece batch büyüdükçe adım kendiliğinden şişmez. `Sum` bölmez; aynı eniyileyici ayarıyla adım yaklaşık `N` kat büyür. Batch boyutunu değiştirirken indirgeme tipini sabit tutun.
9. **Test bağımlılığı sıcak yola girmez.** GoogleTest yalnızca `NEXUSLOSS_BUILD_TESTS=ON` iken `FetchContent` ile alınır. Kütüphaneyi kullanan uygulama gtest'e bağlanmaz.

Derleyici uyarıları arayüzle birlikte gelir: MSVC'de `/W4 /utf-8`, diğer derleyicilerde `-Wall -Wextra -Wpedantic`. Standart uzantıları kapalıdır (`CMAKE_CXX_EXTENSIONS OFF`).

## Klasör ve dosya yapısı

```
NexusLoss/
├── CMakeLists.txt                 sürüm 0.1.0, INTERFACE hedef, test ve örnek anahtarları
├── README.md                      bu dosya
├── include/nexusloss/
│   ├── nexusloss.hpp              tek include
│   ├── details.txt                klasörün rolü
│   ├── core/
│   │   ├── reduction.hpp          Mean, Sum, None, BatchMean
│   │   ├── utils.hpp              kararlılık yardımcıları
│   │   └── details.txt
│   └── losses/
│       ├── loss_base.hpp          eleman bazlı sözleşme, forward önbelleği
│       ├── regression.hpp
│       ├── classification.hpp     sınıflar nexusloss'ta, çok sınıflı API classification'da
│       ├── segmentation.hpp
│       ├── detection.hpp
│       ├── metric.hpp
│       ├── ranking.hpp
│       ├── generative.hpp
│       ├── self_supervised.hpp
│       ├── distillation.hpp
│       ├── sequence.hpp
│       ├── point_cloud.hpp
│       ├── reinforcement.hpp
│       ├── survival.hpp
│       └── details.txt
├── examples/
│   └── main.cpp                   Huber, BCE-with-logits, çok sınıflı çapraz entropi
├── tests/
│   ├── grad_check.hpp             merkezi sonlu fark
│   ├── test_smoke.cpp
│   ├── test_regression.cpp
│   ├── test_classification.cpp
│   ├── test_segmentation.cpp
│   ├── test_detection.cpp
│   ├── test_metric.cpp
│   ├── test_ranking.cpp
│   ├── test_generative.cpp
│   ├── test_self_supervised.cpp
│   ├── test_distillation.cpp
│   ├── test_sequence.cpp
│   ├── test_point_cloud.cpp
│   ├── test_reinforcement.cpp
│   ├── test_survival.cpp
│   └── details.txt
└── docs/
    ├── NexusLoss.pdf              teknik referans
    ├── NexusLoss.html             PDF'in basım kaynağı
    └── website/                   Next.js dokümantasyon sitesi
```

`build/` ve `cmake-build-debug/` CMake üretimidir. Kaynak ağacının parçası sayılmaz. `docs/website` ayrı bir Node projesidir; C++ kütüphanesinin derlemesine girmez.

## Kurulum

İsterler: CMake 3.20 veya üzeri, C++20 derleyicisi. Kütüphanenin kendisi için ağ bağlantısı gerekmez. Testler açıkken GoogleTest v1.14.0 yapılandırma sırasında indirilir.

Başka bir CMake hedefine bağlamak:

```cmake
add_subdirectory(path/to/NexusLoss)
target_link_libraries(uygulama PRIVATE nexusloss::nexusloss)
target_compile_features(uygulama PRIVATE cxx_std_20)
```

Yalnızca başlığı derlemek (CMake olmadan):

```text
cl /std:c++20 /utf-8 /W4 /I include examples\main.cpp
```

```text
c++ -std=c++20 -Wall -Wextra -Wpedantic -I include examples/main.cpp -o basic_example
```

Örnek ve testleri birlikte üretmek (PowerShell, depo kökünden):

```powershell
cmake -S . -B build -DNEXUSLOSS_BUILD_TESTS=ON -DNEXUSLOSS_BUILD_EXAMPLES=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\Release\basic_example.exe
```

Tek yapılandırmalı üreteçlerde (Ninja, Makefiles) `Release` alt klasörü olmaz; `basic_example` doğrudan `build` içindedir. Testi kapatmak için `-DNEXUSLOSS_BUILD_TESTS=OFF`. Örneği kapatmak için `-DNEXUSLOSS_BUILD_EXAMPLES=OFF`. İkisi de öntanımlı açıktır.

Kurulum hedefi (`install`) tanımlı değildir. Tüketici ya `add_subdirectory` kullanır ya da `include/` yolunu derleyiciye verir.

## Hızlı başlangıç

Depodaki `examples/main.cpp` üç aileyi aynı programda çalıştırır.

```cpp
#include <nexusloss/nexusloss.hpp>
#include <iostream>
#include <vector>

int main() {
    const std::vector<double> prediction{2.0, 4.0, 3.0};
    const std::vector<double> target{1.0, 5.0, 2.5};

    nexusloss::HuberLoss<double> huber(1.0);
    const double huber_loss = huber.forward(prediction, target).front();
    const std::vector<double> huber_grad = huber.backward();

    const std::vector<double> logits{0.0, 2.0};
    const std::vector<double> binary_target{1.0, 0.0};
    nexusloss::BCEWithLogitsLoss<double> bce;
    const double bce_loss = bce.forward(logits, binary_target).front();

    const std::vector<double> class_logits{1.0, 2.0, 3.0};
    nexusloss::classification::CrossEntropyLoss<double> cross_entropy;
    const double ce = cross_entropy.forward(class_logits, 2);
    const std::vector<double> ce_grad = cross_entropy.backward();

    std::cout << huber_loss << ' ' << bce_loss << ' ' << ce << '\n';
    return 0;
}
```

`forward(...).front()` eleman bazlı kayıplarda geçerlidir çünkü `Mean` ve `Sum` tek elemanlı vektör döndürür. `None` indirgemesinde vektör girişle aynı uzunluktadır; `.front()` o zaman yalnızca ilk elemandır. Çok sınıflı `CrossEntropyLoss::forward` doğrudan skaler `T` döndürür, `.front()` gerekmez.

Önbelleksiz gradyan:

```cpp
nexusloss::MSELoss<float> mse;
std::vector<float> pred{0.5f, 1.5f};
std::vector<float> y{1.0f, 1.0f};
const std::vector<float> value = mse.compute(pred, y);
const std::vector<float> grad = mse.gradient(pred, y);
```

Ağırlıklı ortalama:

```cpp
nexusloss::MAELoss<double> mae(nexusloss::core::ReductionType::Mean);
const std::vector<double> w{1.0, 3.0, 1.0};
const std::vector<double> value = mae.forward(prediction, target, w);
const std::vector<double> grad = mae.backward();
```

`Mean` paydası ağırlık toplamıdır. `Sum` ve `None` ağırlığı eleman eleman çarpar.

## Sözleşme

`LossBase<T>` için çağrı sırası:

| Çağrı | Girdi | Çıktı | Önbellek |
| --- | --- | --- | --- |
| `compute` | pred, target, isteğe bağlı weight | indirgenmiş kayıp | yazmaz |
| `gradient` | pred, target, isteğe bağlı weight | `dL/d(pred)` | yazmaz |
| `forward` | pred, target, isteğe bağlı weight | `compute` ile aynı değer | girdileri kopyalar |
| `backward` | yok | son `forward`un gradyanı | okur |

`compute_element_wise` ve `gradient_element_wise` indirgeme uygulamaz. Ölçek `gradient` içindeki `scale_gradient` adımındadır. Böylece formül, `Mean` ile `Sum` arasında kopyalanmaz.

Boş span `check_same_size` içinde `std::invalid_argument` olur. `pred` ve `target` uzunlukları, verildiyse `weight` uzunluğu eşit olmalıdır. Gradyan uzunluğu tahminle aynı değilse `std::logic_error` olur. Türevi yazılmamış bir kayıp `gradient_element_wise` içinde aynı hatayı verir. `ZeroOneLoss` türevsizdir; sıfır altgradyan döndürür, istisna atmaz.

`ReductionType::BatchMean` düz vektörde `Mean` ile aynıdır. Satır bazlı kayıplarda (KL gibi) batch'e bölmeyi çağıran yapar. Enum değeri, o niyetin adını korumak için durur.

Çok sınıflı fonksiyonlar (`classification::CrossEntropyLoss` ve serbest `sparse_categorical_cross_entropy` ailesi) **tek örnek** içindir. Batch için örnekleri siz döner, indirgemeyi siz seçersiniz. Logitler ham skordur; içeride `log_sum_exp` vardır, dışarıda softmax uygulayıp tekrar çapraz entropi vermeyin.

Yapısal kayıplar kendi önbelleğini tutar. Ortak kural değişmez: `backward`, o nesnenin son `forward`una bakar. Serbest fonksiyonlar (`ciou_loss`, `info_nce_loss`, `cox_ph_loss`) nesne tutmaz; değeri hemen döndürür. Gradyan gerekiyorsa eşlenik `*_gradient` fonksiyonunu veya sınıfın `backward`unu kullanın.

## İndirgeme ve ağırlık

`apply_reduction` üç özet üretir.

- **Mean.** Toplam bölü eleman sayısı. Ağırlıklıysa toplam `w_i * L_i` bölü `sum(w)`. Payda sıfıra inmesin diye `max(sum(w), epsilon)`.
- **Sum.** Ağırlıksız toplam, ya da `sum(w_i * L_i)`.
- **None.** Aynı uzunlukta vektör. Ağırlık varsa her terim `w_i` ile çarpılır.

Gradyan ölçeği kayıp ölçeğiyle aynıdır. MSE'de eleman türevi `2(pred - target)`dir. `Mean` bunu ayrıca `1/N` ile çarpar. Optimizatöre `Sum` verip öğrenme hızını `Mean` için ayarladıysanız adım büyür.

Sınıf dengesinde eleman ağırlığı yetmeyebilir. `classification::effective_number_class_weights` etkin sayı formülüyle sınıf ağırlığı üretir; `weighted_categorical_cross_entropy` ve `weighted_binary_cross_entropy` bu ağırlığı kayba taşır.

## Sayısal kararlılık

`core::epsilon<T>` şablon sabitidir ve `1e-7`dir. `float` ile `double` aynı eşiği kullanır; `double` için daha dar bir eşik tanımlanmamıştır.

| Yardımcı | Nerede gerekir |
| --- | --- |
| `safe_log` | `log(0)` ve negatif log |
| `clamp_probability` | olasılığı `(eps, 1-eps)` içine alır |
| `sigmoid` / `stable_sigmoid` | logit → olasılık, taşmayan `exp` |
| `log_sum_exp` | softmax paydası, çapraz entropi |
| `softmax` | dağılım; toplam 1 olacak şekilde |
| `safe_sqrt` | negatif girişte kök yok |
| `safe_divide` | Dice paydası |
| `one_hot` | sınıf indeksi → düz satır, örnek-major |
| `apply_mask` | pad konumunu sıfırla çarpar |

Poisson kaybı tahmin olarak **log-oran** bekler. Tweedie kaybı `1 < power < 2` ve pozitif ortalama ile birim dağılımlı sapma hesaplar. Aralık dışındaki güç tanımsızdır; çağıran gücü bu aralıkta vermelidir.

## Hangi kaybı seçmeli

| Eldeki problem | Başlangıç kaybı | Sapınca |
| --- | --- | --- |
| Gerçek değerli regresyon, aykırı az | `MSELoss` | Aykırı varsa `HuberLoss` (delta öntanımlı 1) veya `MAELoss` |
| Pozitif ve çarpık hedef | `MSLELoss` | Sayım verisiyse `PoissonLoss` (log-oran) |
| Asimetrik hata maliyeti | `QuantileLoss` | |
| İkili etiket, olasılık hazır | `BCELoss` | Model logit üretiyorsa `BCEWithLogitsLoss` |
| Tek doğru sınıf, logit vektörü | `classification::CrossEntropyLoss` | Etiket gürültülüyse `LabelSmoothingCrossEntropyLoss` |
| Sınıf çok dengesiz | `FocalLoss` veya `categorical_focal_loss` | Ağırlık için `effective_number_class_weights` |
| Etiket ±1 | `HingeLoss`, `SquaredHingeLoss` | `BCE` bu kodlamayı kabul etmez |
| Piksel maskesi, olasılık | `segmentation::DiceLoss` | Dengesiz sınırda `TverskyLoss` / `FocalTverskyLoss` |
| Kutu `[x_min, y_min, x_max, y_max]` | `detection::CIoULoss` | Yalnızca örtüşme yetiyorsa `IoULoss` |
| Yüz / gömme, normalize kosinüs | `ArcFaceLoss` veya `CosFaceLoss` | Üçlü örnek varsa `TripletLoss` |
| Sıralı liste | `ListNetLoss` veya `ListMLELoss` | Çift bazlıysa `RankNetLoss` |
| Üretici–ayırt edici | `WassersteinLoss` veya `LSGANLoss` | Varyasyonel alt sınır için `VAEELBOLoss` |
| İki görünüm, öz-denetim | `InfoNCELoss` (sıcaklık öntanımlı 0.1) | Çapraz korelasyon için `BarlowTwinsLoss` |
| Öğretmen logitleri | `KnowledgeDistillationLoss` | Öznitelik haritası için `FeatureDistillationLoss` |
| Hizalanmamış dizi | `CTCLoss` | Maskeli token için `MaskedSequenceCrossEntropyLoss` |
| İki nokta bulutu | `ChamferDistanceLoss` | Eşit sayıda nokta ve tam eşleme için `EarthMoverDistanceLoss` |
| Politika gradyanı | `PolicyGradientLoss` | Kırpılmış vekil için `PPOClipLoss` (epsilon 0.2) |
| Sağkalım, sansürlü süre | `CoxPHLoss` | Parametrik süre için `WeibullNLLLoss` |

Bu tablo başlangıç seçimidir. Her kaybın kabul ettiği tensör şekli bir sonraki bölümdedir. Yanlış şekil derlenmeyebilir veya sessizce yanlış ölçek üretir.

## Kayıp kataloğu

### Regresyon — `nexusloss`

Hepsi `LossBase<T>`tır. Kurucu indirgeme alır; Huber, Quantile, Poisson, Tweedie, Cauchy, Charbonnier ve Tukey ek parametre taşır.

| Sınıf | Eleman formülü (özet) | Not |
| --- | --- | --- |
| `MSELoss` | `(pred - target)²` | Gradyan `2(pred - target)`, sonra indirgeme ölçeği |
| `MAELoss` | `\|pred - target\|` | L1 |
| `HuberLoss` | küçük hatada kare, büyük hatada doğrusal | `delta` öntanımlı `1` |
| `LogCoshLoss` | `log(cosh(pred - target))` | |
| `MSLELoss` | log uzayında kare | girişler loga uygun olmalı |
| `QuantileLoss` | pinball | `quantile` eğimi belirler |
| `PoissonLoss` | Poisson negatif log olabilirlik | tahmin log-oran. `PoissonNLLLoss<T>` takma addır |
| `TweedieLoss` | birim dağılımlı sapma | `1 < power < 2`, pozitif ortalama |
| `CauchyLoss` | `log(1 + ((pred-target)/scale)²)` | |
| `CharbonnierLoss` | yumuşatılmış L1 | `sqrt(diff² + eps²)` ailesi |
| `TukeyBiweightLoss` | eşik dışında gradyan sıfır | |

### Sınıflandırma

`nexusloss` içindeki eleman bazlı sınıflar `LossBase`tır: `BCELoss`, `BCEWithLogitsLoss`, `HingeLoss`, `SquaredHingeLoss`, `ExponentialLoss`, `PerceptronLoss`, `ZeroOneLoss`, `FocalLoss`.

`nexusloss::classification` içindeki çok sınıflı tipler tek örnektir: `CrossEntropyLoss` (kurucuda label smoothing, öntanımlı 0), `LabelSmoothingCrossEntropyLoss`, `KLDivLoss`.

Serbest fonksiyonlar: `binary_cross_entropy`, `binary_cross_entropy_with_logits`, `categorical_cross_entropy`, `sparse_categorical_cross_entropy`, `label_smoothing_cross_entropy`, `focal_loss`, `categorical_focal_loss`, `generalized_cross_entropy`, `symmetric_cross_entropy`, `weighted_binary_cross_entropy`, `weighted_categorical_cross_entropy`, `effective_number_class_weights`. Çapraz entropi ve KL için eşlenik `*_gradient` fonksiyonları vardır.

`HingeLoss`, `SquaredHingeLoss`, `PerceptronLoss`, `ExponentialLoss` ve `ZeroOneLoss` hedefi **−1 ve +1** bekler. `BCELoss` hedefi **[0, 1]** bekler. İkisini karıştırmak kaybı sayısal olarak üretir ve yanlış yönde gradyan verir.

### Segmentasyon — `nexusloss::segmentation`

Sınıflar: `DiceLoss`, `IoULoss`, `TverskyLoss`, `FocalTverskyLoss`, `LovaszSoftmaxLoss`.

Serbest fonksiyonlar: `dice_loss`, `iou_loss`, `tversky_loss`, `focal_tversky_loss`, `combo_loss` (Dice + BCE), `boundary_loss`, `hausdorff_distance_loss`, `lovasz_hinge_loss`, `lovasz_softmax_loss`.

Yumuşak maskeler olasılıktır. İstisna: Lovász-hinge logit ister, Lovász-softmax piksel-major olasılık ve sınıf indeksi ister. `hausdorff_distance_loss` uzaklık dönüşümü çalıştırmaz; çağıran sınır mesafe haritasını verir.

### Tespit — `nexusloss::detection`

Örtüşme: `iou`, `iou_loss`, `giou_loss`, `diou_loss`, `ciou_loss` ve `*_loss_gradient`. Sınıflar: `IoULoss`, `GIoULoss`, `DIoULoss`, `CIoULoss`.

Kutu sırası `[x_min, y_min, x_max, y_max]`dir. `DetectionFocalLoss` logit alır; alfa öntanımlı `0.25`, gama `2`. `SmoothL1BBoxLoss` kodlanmış kutu ofsetleri içindir; beta öntanımlı `1`. Bu sınıf regresyon başlığındaki `HuberLoss` ile aynı tensörü beklemez.

### Metrik öğrenme — `nexusloss::metric`

Sınıflar: `ContrastiveLoss`, `TripletLoss`, `ArcFaceLoss` (marj `0.5`, ölçek `64`), `CosFaceLoss` (marj `0.35`, ölçek `64`).

Serbest fonksiyonlar: `contrastive_loss`, `cosine_embedding_loss`, `triplet_loss`, `center_loss`, `n_pair_loss`, `arcface_loss`, `cosface_loss`, `sphereface_loss`. Açısal kayıplar normalize edilmiş kosinüs logiti bekler; normalizasyonu kütüphane içeride her çağrıda yeniden keşfetmez. Gradyanlar giriş sırasıyla art arda durur.

`margin_ranking_loss` bu başlıkta da vardır; sıralama ailesindeki `MarginRankingLoss` ile aynı işin sınıf karşılığı `ranking` içindedir.

### Sıralama — `nexusloss::ranking`

Sınıflar: `MarginRankingLoss`, `RankNetLoss`, `ListNetLoss`, `ListMLELoss`.

Serbest fonksiyonlar: `bpr_loss`, `approx_ndcg_loss`, `soft_rank_loss` (`approx_ndcg_loss` çağrısıdır), `lambda_rank_loss`, `pairwise_logistic`, `listnet_loss`, `listmle_loss`. Yüksek skor daha iyidir. Liste fonksiyonları bir sorgunun aday skorlarını birlikte görür; eleman bazlı `LossBase`a bölünmez.

### Üretken modeller — `nexusloss::generative`

Sınıflar: `AdversarialLoss`, `WassersteinLoss`, `LSGANLoss`, `VAEELBOLoss`.

Serbest fonksiyonlar: `wasserstein_discriminator_loss`, `wasserstein_generator_loss`, `least_squares_gan_loss`, `hinge_discriminator_loss`, `hinge_generator_loss`, `elbo_loss`, `feature_matching_loss`, `cycle_consistency_loss`, `identity_loss`, `perceptual_loss`, `style_loss`, `total_variation_loss`, `diffusion_epsilon_loss`, `l1_loss`, `l2_loss`.

`perceptual_loss` ve `style_loss` VGG çalıştırmaz. Çağıran öznitelik vektörünü verir. `total_variation_loss` görüntüsü satır-major ve kanal-sondadır: indeks `(y * W + x) * C + c`. `diffusion_epsilon_loss` gürültü ile tahminin kare farkının ortalamasıdır (`l2_loss` üzerinden). Ayırt edici ve üretici gradyanları ayrı fonksiyonlardır; tek vektörde karışmaz.

### Öz-denetimli — `nexusloss::self_supervised`

Sınıflar: `InfoNCELoss` (sıcaklık öntanımlı `0.1`), onun alt sınıfı `NTXentLoss`, `BYOLLoss`, `BarlowTwinsLoss`.

Serbest fonksiyonlar: `info_nce_loss`, `nt_xent_loss`, `byol_loss`, `dino_loss`, `barlow_twins_loss`, `vicreg_loss`. InfoNCE pozitif skoru ve negatif skorların listesini alır. DINO ve VICReg öznitelik istatistiğini çağıranın verdiği vektörler üzerinde hesaplar; artırma hattı bu depoda yoktur.

### Damıtım — `nexusloss::distillation`

`KnowledgeDistillationLoss` sıcaklığın karesi ile ölçeklenmiş KL kullanır. `FeatureDistillationLoss` öğrenci ve öğretmen özniteliklerini karşılaştırır. `AttentionTransferLoss` dikkat haritalarını karşılaştırır. Üçü de öğretmeni eğitmez; öğretmen çıktısı sabittir ve çağıran tarafından verilir.

### Dizi — `nexusloss::sequence`

`CTCLoss` zaman-major logit bekler: düzen `[zaman, sınıf]`. Boşluk sınıfı öntanımlı `0`. İleri-geri olasılıkları log uzayında toplanır (`log_add`). `MaskedSequenceCrossEntropyLoss` maskeli dil modeli içindir; pad veya yok sayılan konum maskeyle düşer. `next_sentence_prediction_loss` ikili karardır.

### Nokta bulutu — `nexusloss::point_cloud`

`ChamferDistanceLoss` iki bulutun birbirine en yakın nokta uzaklıklarını toplar. Noktalar düz bellekten okunur; öntanımlı boyut `3` (`[x, y, z, ...]`). `EarthMoverDistanceLoss` eşit sayıda nokta ister, Macar algoritmasıyla birebir eşler ve `O(n³)` çalışır. Büyük bulutta Chamfer öntanımlı seçimdir; EMD tam taşıma maliyeti istendiğinde ve kardinalite küçükken kullanılır.

### Pekiştirmeli öğrenme — `nexusloss::reinforcement`

| Sınıf | Anlamı |
| --- | --- |
| `PolicyGradientLoss` | `−mean(log π · A)` |
| `PPOClipLoss` | kırpılmış vekil, epsilon öntanımlı `0.2` |
| `ValueLoss` | değer fonksiyonu hatası |
| `EntropyBonusLoss` | `−H` döndürür; giriş olasılıktır, logit değildir |

Ortam, avantaj tahmini ve klip istatistiği bu kütüphanede yoktur. Fonksiyonlar verilen log-olasılık, avantaj ve oran üzerinde kapanır.

### Sağkalım — `nexusloss::survival`

`CoxPHLoss` Breslow risk kümesi kullanır. Olay göstergesi `bool`dur (`observed`). Sansürlü gözlem paya girmez, risk kümesinde kalır. `WeibullNLLLoss` log-ölçek ve log-şekil parametreleriyle Weibull negatif log olabilirlik hesaplar. Süre modeli ve sansür kodlaması çağıranın verisindedir.

## Şekil ve kodlama kuralları

Aşağıdaki kurallar testlerle aynı kabuldür. Bir kayıp yanlış kodlamada da sayı üretebilir; hata o zaman derlemede değil eğitimde görünür.

- BCE hedefleri `[0, 1]`. Hinge ailesi hedefleri `−1` ve `+1`.
- Çok sınıflı API tek örnektir: logit vektörü + `size_t` sınıf indeksi.
- Poisson tahminleri log-orandır, ham sayım değildir.
- Tweedie gücü `(1, 2)` açık aralığındadır ve ortalama pozitiftir.
- Tespit kutuları köşe sırasıdır: minimum x, minimum y, maksimum x, maksimum y.
- `SmoothL1BBoxLoss` kodlanmış ofset içindir, köşe kutusu için değildir.
- CTC zaman-major `[zaman, sınıf]`, boşluk indeksi öntanımlı `0`.
- Nokta bulutu düz bellek, öntanımlı nokta boyutu `3`.
- Açısal marj kayıpları normalize kosinüs logiti bekler.
- Algısal, stil ve Hausdorff kayıpları hazır öznitelik veya mesafe haritası ister. Ağ veya uzaklık dönüşümü çalıştırmazlar.
- Toplam değişim görüntüsü satır-major, kanal-sondadır.
- `EntropyBonusLoss` olasılık dağılımı alır.

## Test

Her aile için ayrı bir `tests/test_*.cpp` vardır. `test_smoke.cpp` başlıkların birlikte derlendiğini ve temel çağrıların sonlu kaldığını yoklar. Analitik gradyan, `expect_matches_finite_difference` ile merkezi farka bağlanır:

```text
sayısal ≈ (L(x + h) − L(x − h)) / (2h)
h = 1e-6 ,  tolerans = 1e-5
```

`expect_finite` her bileşenin sonlu olduğunu ister. Yeni bir kayıp eklerken aynı kalıp yeterlidir: değeri bilinen bir noktada elle hesaplanan skalerle karşılaştırın, gradyanı sonlu farkla karşılaştırın, boş ve boyu tutmayan span'in istisna attığını yazın.

GoogleTest keşfi `gtest_discover_tests` ile yapılır. `ctest` tek tek vaka listeler. Başarısız iddiada `--output-on-failure` iddia satırını basar.

## Dokümantasyon sitesi

Site `docs/website` içindedir ve Next.js uygulamasıdır. Kütüphane başlıklarından otomatik üretilmez; metin `docs/website/lib/docs/` altındaki katalog dosyalarındadır. Sayfalar:

- `/` — genel bakış, hızlı başlangıç, kıyas bölümü.
- `/docs` — 13 aile, 88 kayıt, formül, argüman, gradyan ve kod örneği.
- `/api/locale` — ülke ipucuna göre öntanımlı dil. Türkiye için Türkçe, aksi halde İngilizce. Ziyaretçi `TR` / `EN` seçerse tercih `localStorage` anahtarı `nexusloss-lang` altında kalır ve iki sayfanın gezinmesi ortaktır.

Siteyi yerelde açmak:

```powershell
cd docs\website
npm install
npm run dev
```

PDF ile sitenin rolü ayrıdır. Site gezinme ve örnek içindir. `docs/NexusLoss.pdf` tek başına okunan teknik referanstır; bu README ile aynı sözleşmeyi, mimariyi ve kataloğu taşır.

## Sık karşılaşılan hatalar

**`backward()` istisna atıyor.** Aynı nesnede önce `forward` çağrılmamış. Ya sırayı düzeltin ya da `gradient(pred, target)` kullanın.

**Gradyan beklenenden N kat büyük.** İndirgeme `Sum`. `Mean`e geçin veya öğrenme hızını bölün.

**Çapraz entropi bir batch'te tek sayı üretmiyor.** `classification::CrossEntropyLoss` tek örnektir. Döngü ve indirgeme çağırandadır.

**BCE loss'u hedge edilmiş etiketle anlamsız.** `+1/−1` hedefler hinge ailesine gider. BCE için `0/1`.

**Softmax uyguladıktan sonra `CrossEntropyLoss`.** Fonksiyon logit ister ve log-sum-exp'i kendisi yapar. Çifte softmax dağılımı ezer.

**CTC boyutu tutmuyor.** Dizi zaman × sınıf, zaman dışta. Boşluk sınıfı `0` değilse kurucuda söyleyin.

**EMD çok yavaş veya hata veriyor.** Eşit kardinalite ve `O(n³)` gerekir. Büyük bulutta `ChamferDistanceLoss` kullanın.

**Hausdorff veya perceptual hep aynı.** Bu fonksiyonlar ağ çalıştırmaz. Mesafe haritası veya öznitelik hazır değilse kayıp o hazır tensörün farkıdır.

**Boş vektör.** `check_same_size` boş girişi reddeder. Maske bütün elemanları atıyorsa çağrıdan önce o örneği düşürün; sıfır uzunluk göndermeyin.

**`float` modeli `double` kaybına bağlamak.** Şablon tipi tahminin tipiyle aynı olmalıdır. Kütüphane ikisini karıştırıp dönüştürmez.
