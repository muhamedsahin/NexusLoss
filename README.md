# NexusLoss

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

NexusLoss, bir eğitim adımında tahmin ile hedef arasındaki kaybı ve bu kaybın tahmine göre analitik gradyanını hesaplar. Sürüm **0.1.0**. Lisans **Apache-2.0** (`LICENSE`).

Nexus (üst depo) bu kitaplığı iki sınıfla çağırır. Başlık `include/nexus/loss/losses.hpp` dosyasındadır; ad alanı `nexus::loss`.

- `CrossEntropy(label_smoothing)` — logitler ve sınıf indeksleri. `label_smoothing` `[0, 1)` aralığındadır, varsayılan `0`.
- `MSE` — aynı uzunluktaki tahmin ve hedef.

İkisi de gradyanı çağıranın tamponuna yazar ve **ortalama** kaybı döndürür. Kütüphane ağırlık güncellemez, veri yüklemez ve autograd grafı tutmaz.

Aile başlıkları `include/nexusloss/` altındadır. Çapraz entropi cephesi her satırda `nexusloss::classification::CrossEntropyLoss` kullanır, sonra gradyanı batch boyutuna böler. `MSE` cephesi kare farkın ortalamasını kendisi yazar: eleman türevi `2 (tahmin − hedef)`, ortalama için `1/n`. `float` için ayrıca `nexusloss::fused` (AVX2, `src/fused_simd.cpp`) vardır; cephe o yolu çağırmaz.

Bu depoda yayımlanmış hız kıyası yoktur.

## Kurulum

İsterler: CMake 3.20 veya üzeri, C++20 derleyicisi. CMake hedefi statik kitaplık `nexusloss` ve takma ad `nexusloss::nexusloss`tir. Başlıkların çoğu şablondur; AVX2 çekirdeği ayrı çeviri birimidir, bu yüzden tüketici hedefe bağlanır.

```cmake
add_subdirectory(path/to/NexusLoss)
target_link_libraries(uygulama PRIVATE nexusloss::nexusloss)
target_compile_features(uygulama PRIVATE cxx_std_20)
```

Nexus ağacında üst `CMakeLists.txt` zaten `add_subdirectory(lib/NexusLoss)` çağırır. `nexus::nexus` bu hedefe herkese açık bağlanır ve hem `include/` (cephe) hem `lib/NexusLoss/include` (aileler) yolunu taşır.

Yerel derleme, bu klasörden:

```powershell
cmake -S . -B build -DNEXUSLOSS_BUILD_TESTS=ON -DNEXUSLOSS_BUILD_EXAMPLES=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

`NEXUSLOSS_BUILD_TESTS` ve `NEXUSLOSS_BUILD_EXAMPLES` varsayılan olarak açıktır. Testler GoogleTest v1.14.0 indirir; kapalıyken ağ gerekmez. `examples/main.cpp` aile API’sini (`forward` / `backward`) kullanır, yukarıdaki cepheyi değil.

MSVC arayüz uyarıları `/W4 /utf-8`, `src/fused_simd.cpp` için `/arch:AVX2`. Diğer derleyicilerde `-Wall -Wextra -Wpedantic` ve o dosya için `-mavx2 -mfma`.

### Python

`nexus.loss` bu klasörün içinde değildir. Üst depo kökünden, Python 3.11 ile:

```text
cmake -S . -B build -DNEXUS_BUILD_PYTHON=ON
cmake --build build --target _core
```

Modül `python/src/nexus` altına yazılır. İçe aktarmadan önce `PYTHONPATH` olarak `python/src` verilir. Ayrıntı üst depodaki `python/README.md` dosyasındadır.

## C++

Logitler satır-major’dur: uzunluk `batch * classes`. `targets[i]`, `i`. satırın sınıf indeksidir. `grad` ile `logits` aynı uzunluktadır. `batch > 0` ve `classes > 1` olmalıdır. Sınıf indeksi `[0, classes)` dışındaysa `std::invalid_argument` atılır.

```cpp
#include "nexus/loss/losses.hpp"

#include <vector>

int main() {
    const std::vector<float> logits{2.0f, 0.0f, 0.0f, 2.0f};
    const std::vector<int> targets{0, 1};
    std::vector<float> ce_grad(logits.size());

    const float ce = nexus::loss::CrossEntropy(0.0f).forward_backward(
        logits.data(), targets.data(), /*batch*/ 2, /*classes*/ 2, ce_grad.data());

    const std::vector<float> prediction{1.0f, 2.0f};
    const std::vector<float> target{1.0f, 0.0f};
    std::vector<float> mse_grad(prediction.size());

    const float mse = nexus::loss::MSE().forward_backward(
        prediction.data(), target.data(), static_cast<int>(prediction.size()), mse_grad.data());

    return ce == ce && mse == mse ? 0 : 1;
}
```

`CrossEntropy::forward_backward` ortalama kaybı döndürür ve `grad` içindeki her elemanı `1/batch` ile çarpar. Satır türevi `softmax(logit) − yumuşatılmış hedef`tir. Yumuşatma `0` iken hedef tek-sıcaktır. Logitlere önceden softmax uygulanmaz.

`MSE::forward_backward` `mean((tahmin − hedef)²)` döndürür. `n > 0` olmalıdır. `grad[i] = 2 * (prediction[i] − target[i]) / n`.

## Python

`CrossEntropy.forward_backward(logits, targets, classes)` bir sözlük döndürür: `loss` ortalama kayıp, `grad` logitlerle aynı uzunlukta liste.

```python
import nexus

ce = nexus.loss.CrossEntropy(0.0).forward_backward(
    [2.0, 0.0, 0.0, 2.0],
    [0, 1],
    2,
)
mean_loss = ce["loss"]
gradient = ce["grad"]  # len == 4
```

`MSE.forward_backward(prediction, target)` sözlük değil, `(loss, grad)` çifti döndürür. Uzunluklar eşit ve boş olmayan olmalıdır.

```python
loss, grad = nexus.loss.MSE().forward_backward([1.0, 2.0], [1.0, 0.0])
```

## Dokümantasyon

- Türkçe anlatım: `docs/kilavuz.md`
- Site: `docs/website` (`npm install`, ardından `npm run dev`)
- `docs/NexusLoss.pdf` bu turda üretilmedi. Pandoc kurulu değildir; PDF dosyasına dokunulmadı.

Aile kataloğu (regresyon, sınıflandırma, segmentasyon ve diğerleri) başlıklardadır ve sitedeki `/docs` sayfasında durur. O sınıfların çoğu `forward` ile girdiyi saklar, `backward` ile türevi okur. Nexus eğitim adımının çağırdığı yüzey yukarıdaki `forward_backward` cephesidir.
