<div align="center">

# 📉 NexusLoss

### Tahmin ile hedef arasındaki kaybı ve analitik gradyanını hesaplayan C++20 kütüphanesi

*Regresyon, sınıflandırma, segmentasyon ve daha fazlası — Nexus eğitim adımının kayıp katmanı.*

[![Version](https://img.shields.io/badge/s%C3%BCr%C3%BCm-0.1.0-blueviolet?style=for-the-badge)](CMakeLists.txt)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![AVX2](https://img.shields.io/badge/fused-AVX2-FF6F00?style=for-the-badge)](src/fused_simd.cpp)
[![License](https://img.shields.io/badge/Lisans-Apache--2.0-D22128?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

[Cephe](#-nexusun-çağırdığı-cephe) • [Kurulum](#-kurulum) • [C++](#-c) • [Python](#-python) • [Belgeler](#-dokümantasyon) • [Nexus](https://github.com/muhamedsahin/Nexus)

</div>

> Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

---

## 🎯 Nexus'un çağırdığı cephe

Başlık `include/nexus/loss/losses.hpp`, ad alanı `nexus::loss`:

| Sınıf | Girdi | Dönüş | Gradyan |
| --- | --- | --- | --- |
| `CrossEntropy(label_smoothing)` | logitler + sınıf indeksleri | ortalama kayıp | `softmax − yumuşatılmış hedef`, `1/batch` ölçekli |
| `MSE` | aynı uzunlukta tahmin + hedef | `mean((t − h)²)` | `2 (t − h) / n` |

İkisi de gradyanı **çağıranın tamponuna** yazar. Kütüphane ağırlık güncellemez, veri yüklemez, autograd grafı tutmaz.

- Aile başlıkları `include/nexusloss/` altındadır. Çapraz entropi cephesi her satırda `nexusloss::classification::CrossEntropyLoss` kullanır.
- `float` için ayrıca `nexusloss::fused` (AVX2, `src/fused_simd.cpp`) vardır; cephe o yolu çağırmaz.
- Nexus'un dil modeli eğitmeni artık kaydırmalı kayıp için Nexus'un kendi füzyonlu `fused::cross_entropy_shifted` çekirdeğini kullanır (kopyasız); sınıflandırma `Trainer`'ı bu cepheyi çağırmaya devam eder.

---

## 🛠 Kurulum

CMake 3.20+ ve C++20. Hedef: statik `nexusloss`, takma ad `nexusloss::nexusloss`.

```cmake
add_subdirectory(path/to/NexusLoss)
target_link_libraries(uygulama PRIVATE nexusloss::nexusloss)
target_compile_features(uygulama PRIVATE cxx_std_20)
```

Nexus ağacında üst `CMakeLists.txt` zaten `add_subdirectory(lib/NexusLoss)` çağırır.

```powershell
cmake -S . -B build -DNEXUSLOSS_BUILD_TESTS=ON -DNEXUSLOSS_BUILD_EXAMPLES=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

<details>
<summary><b>Derleme ayrıntıları</b></summary>

- `NEXUSLOSS_BUILD_TESTS` ve `NEXUSLOSS_BUILD_EXAMPLES` varsayılan açıktır. Testler GoogleTest v1.14.0 indirir; kapalıyken ağ gerekmez.
- `examples/main.cpp` aile API'sini (`forward` / `backward`) kullanır, cepheyi değil.
- MSVC: `/W4 /utf-8`, `src/fused_simd.cpp` için `/arch:AVX2`. Diğer derleyiciler: `-Wall -Wextra -Wpedantic`, o dosya için `-mavx2 -mfma`.

</details>

---

## 💻 C++

Logitler satır-major'dur (`batch * classes`). `batch > 0`, `classes > 1`; sınıf indeksi `[0, classes)` dışındaysa `std::invalid_argument`.

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

Logitlere önceden softmax uygulanmaz. Yumuşatma `0` iken hedef tek-sıcaktır.

---

## 🐍 Python

`nexus.loss` üst depodan (Python 3.11) derlenir: `cmake -S . -B build -DNEXUS_BUILD_PYTHON=ON` → `cmake --build build --target _core`, ardından `PYTHONPATH=python/src`.

```python
import nexus

ce = nexus.loss.CrossEntropy(0.0).forward_backward([2.0, 0.0, 0.0, 2.0], [0, 1], 2)
mean_loss, gradient = ce["loss"], ce["grad"]   # len(gradient) == 4

loss, grad = nexus.loss.MSE().forward_backward([1.0, 2.0], [1.0, 0.0])   # (loss, grad) çifti
```

---

## 📚 Dokümantasyon

| Kaynak | Not |
| --- | --- |
| `docs/kilavuz.md` | Türkçe anlatım |
| `docs/website` | Site (`npm install`, ardından `npm run dev`); aile kataloğu `/docs` sayfasında |
| `docs/NexusLoss.pdf` | Bu turda üretilmedi (Pandoc kurulu değil) |

Aile sınıflarının çoğu `forward` ile girdiyi saklar, `backward` ile türevi okur. Bu depoda yayımlanmış hız kıyası **yoktur**.

## 📄 Lisans

Apache-2.0 — [`LICENSE`](LICENSE).

---

<div align="center">

**🎓 Bu proje BTÜ (Bursa Teknik Üniversitesi) Bilgisayar Mühendisliği 1. sınıf öğrencisi tarafından yapay zeka destekli olarak yapıldı.**

<sub>Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.</sub>

<sub>⚡ [Nexus](https://github.com/muhamedsahin/Nexus) ekosisteminin bir parçasıdır.</sub>

</div>
