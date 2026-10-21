# NexusLoss kılavuzu

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

Bu dosya, Nexus’un kayıp adımını nasıl çağırdığını anlatır. Sürüm 0.1.0. Lisans Apache-2.0 (`../LICENSE`).

## PDF üretilmedi

Pandoc bu makinede kurulu değildir. `docs/NexusLoss.pdf` yeniden üretilmedi ve elle değiştirilmedi. `docs/NexusLoss.html` de bu kılavuzun kaynağı olarak güncellenmedi. Güncel anlatım bu dosya, depo kökündeki `README.md` ve `docs/website` sitesidir. PDF’i bu metinle aynı kabul etmeyin.

## Ne hesaplanır

Bir eğitim adımı tahmin üretir, kayıp o tahmini hedefle karşılaştırır, eniyileyici ağırlığı günceller. NexusLoss yalnızca ortadaki parçadır.

Üst depodaki cephe `include/nexus/loss/losses.hpp` içindedir.

| Sınıf | Kurucu | Çağrı | Dönüş |
| --- | --- | --- | --- |
| `nexus::loss::CrossEntropy` | `float label_smoothing = 0` | `forward_backward(logits, targets, batch, classes, grad)` | ortalama kayıp (`float`) |
| `nexus::loss::MSE` | yok | `forward_backward(prediction, target, n, grad)` | ortalama kare hata (`float`) |

İki çağrı da gradyanı `grad` tamponuna yazar. Tamponu kütüphane ayırmaz. Dönüş bir tensör düğümü değildir.

`label_smoothing` `[0, 1)` dışında ise kurucu `std::invalid_argument` atar. `1` yasal değildir.

## Çapraz entropi

Logitler ham skordur, satır-major bellektedir. `i`. satır `logits + i * classes` adresindedir ve `classes` elemandır. `targets`, uzunluğu `batch` olan `int` sınıf indeksleridir.

Koşullar:

- `batch > 0`
- `classes > 1`
- her `targets[i]`, `0` ile `classes - 1` arasındadır

Aksi halde `std::invalid_argument` atılır.

Her satır `nexusloss::classification::CrossEntropyLoss<float>` ile işlenir. Satır kaybı log-sum-exp ile hesaplanır; dışarıda softmax uygulayıp sonucu tekrar vermeyin. Satır gradyanı `softmax(z) − yumuşatılmış hedef`tir.

Yumuşatma `s`, sınıf sayısı `C` olsun:

- her sınıfa `s / C` eklenir
- doğru sınıfa ayrıca `1 − s` eklenir

`s = 0` iken hedef tek-sıcaktır. Bütün satırların kaybı toplanır ve `batch`e bölünür. Gradyan tamponunun her elemanı aynı `1/batch` çarpanını alır. Böylece dönen skaler ile tampon aynı ortalamanın iki yüzüdür.

```cpp
#include "nexus/loss/losses.hpp"

#include <vector>

const std::vector<float> logits{2.0f, 0.0f, 0.0f, 2.0f};
const std::vector<int> targets{0, 1};
std::vector<float> grad(logits.size());

const float loss = nexus::loss::CrossEntropy(0.0f).forward_backward(
    logits.data(), targets.data(), 2, 2, grad.data());
```

Burada iki satır ve iki sınıf vardır. Birinci satırın doğru sınıfı `0`, ikincinin `1`dir. `grad` dört elemanlıdır.

## MSE

`prediction` ve `target` aynı `n` elemanı paylaşır. `n > 0` değilse `std::invalid_argument` atılır.

```text
kayıp  = mean_i (prediction[i] − target[i])²
grad[i] = 2 * (prediction[i] − target[i]) / n
```

```cpp
const std::vector<float> prediction{1.0f, 2.0f};
const std::vector<float> target{1.0f, 0.0f};
std::vector<float> grad(prediction.size());

const float loss = nexus::loss::MSE().forward_backward(
    prediction.data(), target.data(), 2, grad.data());
```

Bu cephe indirgeme seçmez. Ortalama sabittir. `Sum` veya `None` isteyen kod `nexusloss::MSELoss<T>` kullanır; o sınıf `LossBase`tir, `forward` girdiyi saklar, `backward` önbelleği okur. İkisi aynı imza değildir.

## Python

Kabuk üst deponun `nexus` modülüdür. `nexus.loss.CrossEntropy` kurucusu `label_smoothing` alır, varsayılan `0.0`.

```python
import nexus

ce = nexus.loss.CrossEntropy(0.0).forward_backward(
    [2.0, 0.0, 0.0, 2.0],
    [0, 1],
    2,
)
```

Dönüş sözlüktür.

- `ce["loss"]` — ortalama kayıp
- `ce["grad"]` — logitlerle aynı uzunlukta `float` listesi

Bağlama, `classes <= 1`, boş hedef veya `len(logits) != len(targets) * classes` durumunda `ValueError`a dönüşen `std::invalid_argument` atar.

MSE sözlük döndürmez. İki elemanlı demettir:

```python
loss, grad = nexus.loss.MSE().forward_backward([1.0, 2.0], [1.0, 0.0])
```

Uzunluklar farklıysa veya dizi boşsa çağrı reddedilir.

Modül bu klasörden `import` edilemez. Üst depo kökünde `-DNEXUS_BUILD_PYTHON=ON` ile `_core` hedefi derlenir; çıktı `python/src/nexus` içine yazılır. `PYTHONPATH` bu `src` klasörünü göstermelidir. Yorumlayıcı Python 3.11’dir. Adımlar `python/README.md` içindedir.

## C++ kurulumu

CMake 3.20 ve C++20 gerekir. Bu deponun hedefi statik `nexusloss::nexusloss`tir.

```cmake
add_subdirectory(path/to/NexusLoss)
target_link_libraries(uygulama PRIVATE nexusloss::nexusloss)
target_compile_features(uygulama PRIVATE cxx_std_20)
```

Cephe başlığı NexusLoss’un `include/` ağacında durmaz. Üst deponun `include/nexus/loss/losses.hpp` yoludur. `nexus::nexus`e bağlanan bir hedef her iki include yolunu da alır. Yalnızca `nexusloss::nexusloss` bağlanırsa aile başlıkları görünür; cephe başlığı için üst `include/` ayrıca verilmelidir.

`src/fused_simd.cpp` float için `nexusloss::fused::mse`, `mae`, `huber` ve `bce_with_logits` sembollerini tanımlar. `reduction` değeri `0` ortalama, diğer değerler toplamdır. Cephedeki `MSE::forward_backward` bu fonksiyonu çağırmaz.

Yerel test ve örnek:

```powershell
cmake -S . -B build -DNEXUSLOSS_BUILD_TESTS=ON -DNEXUSLOSS_BUILD_EXAMPLES=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

İki seçenek de varsayılan olarak açıktır. Test açıkken GoogleTest v1.14.0 yapılandırma sırasında indirilir. `examples/main.cpp` Huber, logitli ikili çapraz entropi ve tek örnek `classification::CrossEntropyLoss` çalıştırır.

## Aile başlıkları

`include/nexusloss/nexusloss.hpp` aileleri birlikte açar. Katalog sitede `/docs` altındadır. Kısa ayrım:

- Eleman bazlı sınıflar `LossBase<T>`ten türer. `compute` ve `gradient` önbellek yazmaz. `forward` girdiyi kopyalar. `backward`, `forward` olmadan `std::logic_error` atar.
- `classification::CrossEntropyLoss<T>` tek örnektir: logit span’i ve `size_t` sınıf indeksi. Batch ortalamasını ve `1/N` ölçeğini cephe ekler.
- Yapısal kayıplar (kutu, CTC, nokta bulutu, liste) kendi `forward` / `backward` imzalarını kullanır.

Bu kılavuzdaki çalışan örnekler cephedir. Aile örneği yazmadan önce ilgili başlığı okuyun; sitedeki parça o başlıkla aynı imzayı taşımalıdır.

## Sık hatalar

**Gradyan beklenenin N katı.** Cephe her zaman ortalamadır. `MSELoss` veya başka bir `LossBase` `Sum` ile kurulduysa ölçek büyüktür.

**Çapraz entropiye softmax verilmiş logit.** Cephe ve `CrossEntropyLoss` log-sum-exp’i kendisi yapar. İkinci softmax dağılımı ezer.

**Python MSE’yi sözlük sanmak.** `forward_backward` `(loss, grad)` döndürür. Sözlük yalnızca `CrossEntropy` içindir.

**Tek başına bu depoyu derleyip `nexus.loss` beklemek.** Python hedefi üst CMake’dedir (`NEXUS_BUILD_PYTHON`).

**PDF ile bu sayfayı karşılaştırmak.** Pandoc olmadığı için PDF yenilenmedi.
