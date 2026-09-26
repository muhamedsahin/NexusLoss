import { bi, type CategoryDoc } from "./types";

export const regression: CategoryDoc = {
  id: "regression",
  index: "01",
  title: bi("Regression", "Regresyon"),
  lede: bi(
    "Element-wise objectives for continuous targets. Every class inherits LossBase, so forward() caches the inputs and backward() returns dL/d(prediction), already scaled by Mean, Sum or None.",
    "Sürekli hedefler için eleman bazlı amaçlar. Her sınıf LossBase'den türer: forward() girdileri saklar, backward() tahmine göre türevi Mean, Sum veya None ölçeğiyle döndürür.",
  ),
  losses: [
    {
      id: "mse",
      name: "MSE",
      api: "nexusloss::MSELoss<T>",
      formula: "L = (x − y)²",
      kind: bi("Element-wise · LossBase", "Eleman bazlı · LossBase"),
      when: bi(
        "Use it when the target is a clean continuous value and large mistakes should dominate the update. It is the default for regression heads, value functions and reconstruction terms whose noise is roughly Gaussian.",
        "Hedef temiz bir sürekli değerse ve büyük hataların güncellemeye hükmetmesini istiyorsan bunu kullan. Gürültüsü kabaca Gauss olan regresyon başları, değer fonksiyonları ve rekonstrüksiyon terimleri için varsayılan seçimdir.",
      ),
      logic: bi(
        "Squaring the residual makes the loss smooth at zero and grows without a bound. A point that is twice as wrong contributes four times as much, so the optimizer spends its budget on the worst predictions. That is useful when outliers are real signal, and harmful when they are sensor glitches.",
        "Artığın karesi, kaybı sıfırda pürüzsüz kılar ve üstten sınır koymaz. İki kat yanlış bir nokta dört kat katkı verir; optimize edici bütçesini en kötü tahminlere harcar. Aykırı değer gerçek sinyalse bu iyidir, sensör hatasıysa zararlıdır.",
      ),
      gradient: bi(
        "∂L/∂x = 2(x − y) before reduction. Mean divides every component by N. The gradient grows linearly with the error, which is why a single wild target can destabilize a step.",
        "İndirgemeden önce ∂L/∂x = 2(x − y). Mean her bileşeni N'ye böler. Gradyan hatayla doğrusal büyür; tek bir uç hedef bir adımı bozabilir.",
      ),
      code: `#include <nexusloss/nexusloss.hpp>
#include <vector>

std::vector<double> prediction{2.0, 4.0, 3.0};
std::vector<double> target{1.0, 5.0, 2.5};

nexusloss::MSELoss<double> loss; // Mean
double value = loss.forward(prediction, target).front();
std::vector<double> gradient = loss.backward();`,
      params: [
        { name: "reduction", detail: bi("Mean (default), Sum, None or BatchMean.", "Mean (varsayılan), Sum, None veya BatchMean.") },
      ],
    },
    {
      id: "mae",
      name: "MAE / L1",
      api: "nexusloss::MAELoss<T>",
      formula: "L = |x − y|",
      kind: bi("Element-wise · LossBase", "Eleman bazlı · LossBase"),
      when: bi(
        "Choose MAE when every unit of error should cost the same, regardless of how large it already is. Median regression, robust baselines and targets with heavy tails belong here.",
        "Hata ne kadar büyük olursa olsun her birimin aynı maliyeti olmasını istiyorsan MAE seç. Medyan regresyonu, sağlam taban çizgileri ve kalın kuyruklu hedefler buraya girer.",
      ),
      logic: bi(
        "The absolute residual is the loss of the median: half the mass can sit on either side without the far points taking over. It is not differentiable at a perfect prediction. NexusLoss uses the sign subgradient, which is zero only in the trivial equal case handled as a non-positive branch of the sign.",
        "Mutlak artık, medyanın kaybıdır: kütlenin yarısı iki yana oturabilir, uzak noktalar ele geçirmez. Tam isabette türev yoktur. NexusLoss işaret alt-gradyanını kullanır.",
      ),
      gradient: bi(
        "∂L/∂x = sign(x − y). The magnitude never exceeds 1 before reduction, so steps stay bounded even when a label is nonsense.",
        "∂L/∂x = sign(x − y). İndirgemeden önce büyüklük 1'i geçmez; etiket saçma olsa bile adımlar sınırlı kalır.",
      ),
      code: `nexusloss::MAELoss<double> loss(nexusloss::core::ReductionType::Sum);
auto value = loss.forward(prediction, target);
auto gradient = loss.backward();`,
      params: [
        { name: "reduction", detail: bi("Same contract as every LossBase class.", "Tüm LossBase sınıflarıyla aynı sözleşme.") },
      ],
    },
    {
      id: "huber",
      name: "Huber / Smooth L1",
      api: "nexusloss::HuberLoss<T>",
      formula: "½ e²    if |e| ≤ δ\nδ(|e| − ½δ)   otherwise",
      kind: bi("Element-wise · robust", "Eleman bazlı · sağlam"),
      when: bi(
        "This is the regression loss to reach for first when the data is mostly clean but a few points are garbage. Bounding-box offsets in detection use the same shape under the name Smooth L1.",
        "Veri çoğunlukla temiz ama birkaç nokta çöpse ilk uzanılacak regresyon kaybı budur. Tespitte kutu ofsetleri aynı biçimi Smooth L1 adıyla kullanır.",
      ),
      logic: bi(
        "Inside the delta ball the loss is quadratic, so small errors get a precise, vanishing gradient. Outside it, the loss becomes linear and the gradient clips at delta. You keep MSE's fine behavior near the target and MAE's refusal to let one outlier write the update.",
        "Delta topunun içinde kayıp kareseldir; küçük hatalar hassas, sönen bir gradyan alır. Dışarıda kayıp doğrusal olur ve gradyan deltada kırpılır. Hedefin yanında MSE'nin inceliği, tek bir aykırı değerin güncellemeyi yazmasını engelleyen MAE'nin inadı birlikte durur.",
      ),
      gradient: bi(
        "∂L/∂x = (x − y) while |x − y| ≤ δ, and δ · sign(x − y) outside. SmoothL1Loss<T> is an alias of HuberLoss.",
        "|x − y| ≤ δ iken ∂L/∂x = (x − y), dışında δ · sign(x − y). SmoothL1Loss<T>, HuberLoss'un takma adıdır.",
      ),
      code: `nexusloss::HuberLoss<double> loss(1.0); // delta
double value = loss.forward(prediction, target).front();
auto gradient = loss.backward();

// Same type, explicit name used by detection papers.
nexusloss::SmoothL1Loss<double> smooth(1.0);`,
      params: [
        { name: "delta", detail: bi("Positive hinge between the L2 bowl and the L1 tails. Default 1.", "L2 çanağı ile L1 kuyrukları arasındaki pozitif eşik. Varsayılan 1.") },
        { name: "reduction", detail: bi("Mean by default.", "Varsayılan Mean.") },
      ],
    },
    {
      id: "log-cosh",
      name: "Log-Cosh",
      api: "nexusloss::LogCoshLoss<T>",
      formula: "L = log(cosh(x − y))",
      kind: bi("Element-wise · smooth robust", "Eleman bazlı · pürüzsüz sağlam"),
      when: bi(
        "Use it when you want Huber's robustness without a hand-tuned corner. The function is smooth everywhere, which helps second-order intuition and avoids a kink at δ.",
        "El ile ayarlanan bir köşe olmadan Huber'in sağlamlığını istiyorsan bunu kullan. Fonksiyon her yerde pürüzsüzdür; ikinci derece sezgiyi kolaylaştırır ve δ'da bir kırık bırakmaz.",
      ),
      logic: bi(
        "For tiny residuals, cosh(e) ≈ 1 + e²/2, so log(cosh(e)) ≈ e²/2 and the loss behaves like MSE. For large residuals, cosh grows like an exponential and the log turns that into |e| − log 2. The implementation uses a stable form, log1p(exp(−2|e|)) + |e| − log 2, so huge errors do not overflow.",
        "Küçük artıkta cosh(e) ≈ 1 + e²/2 olduğundan log(cosh(e)) ≈ e²/2 ve kayıp MSE gibi davranır. Büyük artıkta cosh üstel büyür, log bunu |e| − log 2'ye çevirir. Uygulama taşmayı önleyen kararlı biçimi kullanır: log1p(exp(−2|e|)) + |e| − log 2.",
      ),
      gradient: bi(
        "∂L/∂x = tanh(x − y). The derivative saturates at ±1, so the step size stays honest no matter how wrong a point is.",
        "∂L/∂x = tanh(x − y). Türev ±1'de doyar; bir nokta ne kadar yanlış olursa olsun adım boyu ölçülü kalır.",
      ),
      code: `nexusloss::LogCoshLoss<double> loss;
double value = loss.forward(prediction, target).front();
auto gradient = loss.backward();`,
      params: [
        { name: "reduction", detail: bi("Mean, Sum or None.", "Mean, Sum veya None.") },
      ],
    },
    {
      id: "msle",
      name: "MSLE",
      api: "nexusloss::MSLELoss<T>",
      formula: "L = (log(1+x) − log(1+y))²",
      kind: bi("Element-wise · relative", "Eleman bazlı · göreli"),
      when: bi(
        "Reach for MSLE when the target is non-negative and a miss of 10 versus 100 should matter more than a miss of 1010 versus 1100. Counts, prices and populations with a wide dynamic range fit.",
        "Hedef negatif değilse ve 10 ile 100 arasındaki sapma, 1010 ile 1100 arasındakinden daha önemliyse MSLE'ye uzan. Geniş dinamik aralıklı sayımlar, fiyatlar ve nüfuslar uyar.",
      ),
      logic: bi(
        "The log compresses scale before the square. Under-predicting and over-predicting the same ratio are treated symmetrically in log space, which is what you want for multiplicative processes. Both prediction and target must be ≥ 0; the +1 inside log1p keeps a true zero legal.",
        "Log, kareden önce ölçeği sıkıştırır. Aynı orandaki eksik ve fazla tahmin log uzayında simetriktir; çarpımsal süreçlerde istenen budur. Tahmin ve hedef ≥ 0 olmalıdır; log1p içindeki +1, gerçek sıfırı yasal kılar.",
      ),
      gradient: bi(
        "∂L/∂x = 2 (log1p(x) − log1p(y)) / (1 + x). Large predictions automatically shrink their own gradient.",
        "∂L/∂x = 2 (log1p(x) − log1p(y)) / (1 + x). Büyük tahminler kendi gradyanlarını kendiliğinden küçültür.",
      ),
      code: `std::vector<double> rate{10.0, 0.0, 40.0};
std::vector<double> truth{12.0, 1.0, 20.0};
nexusloss::MSLELoss<double> loss;
double value = loss.forward(rate, truth).front();`,
      params: [
        { name: "inputs", detail: bi("Prediction and target must be non-negative.", "Tahmin ve hedef negatif olamaz.") },
      ],
    },
    {
      id: "quantile",
      name: "Quantile / Pinball",
      api: "nexusloss::QuantileLoss<T>",
      formula: "q·(y−x)  if y ≥ x\n(q−1)·(y−x)  otherwise",
      kind: bi("Element-wise · asymmetric", "Eleman bazlı · asimetrik"),
      when: bi(
        "Use it when the cost of being low is not the cost of being high: delivery ETAs, inventory, medical risk bands, or any interval you want to learn directly instead of assuming a Gaussian.",
        "Düşük kalmanın maliyeti yüksek kalmanın maliyeti değilse bunu kullan: teslim süreleri, stok, tıbbi risk bantları ya da Gauss varsaymadan doğrudan öğrenmek istediğin herhangi bir aralık.",
      ),
      logic: bi(
        "The pinball loss is the check function of quantile regression. With q = 0.9 the model is punished harder for falling short, so the prediction settles on the 90th percentile of the conditional distribution. q = 0.5 recovers MAE, the median. Train three heads at 0.1, 0.5 and 0.9 and you have an interval without a separate variance network.",
        "Pinball kaybı, kantil regresyonunun kontrol fonksiyonudur. q = 0.9 iken model eksik kalınca daha sert cezalandırılır; tahmin koşullu dağılımın 90. persentiline oturur. q = 0.5, medyan olan MAE'yi geri getirir. 0.1, 0.5 ve 0.9 ile üç baş eğitirsen ayrı bir varyans ağı olmadan aralık elde edersin.",
      ),
      gradient: bi(
        "If the prediction is below the target, ∂L/∂x = −q. If it is above, ∂L/∂x = 1 − q. The push is constant, not proportional to distance.",
        "Tahmin hedefin altındaysa ∂L/∂x = −q. Üstündeyse ∂L/∂x = 1 − q. İtme mesafeyle orantılı değil, sabittir.",
      ),
      code: `nexusloss::QuantileLoss<double> p90(0.9);
double value = p90.forward(prediction, target).front();
auto gradient = p90.backward();`,
      params: [
        { name: "quantile", detail: bi("Open interval (0, 1). There is no default; you must choose the percentile.", "(0, 1) açık aralığı. Varsayılan yok; persentili sen seçersin.") },
      ],
    },
    {
      id: "poisson",
      name: "Poisson NLL",
      api: "nexusloss::PoissonLoss<T>",
      formula: "L = exp(x) − y·x",
      kind: bi("Element-wise · count", "Eleman bazlı · sayım"),
      when: bi(
        "Counts: events per minute, clicks, spikes, photons. The prediction is a log-rate, not the rate itself, matching the canonical Poisson GLM.",
        "Sayımlar: dakikadaki olay, tıklama, diken, foton. Tahmin oranın kendisi değil log-orandır; kanonik Poisson GLM ile aynı sözleşmedir.",
      ),
      logic: bi(
        "A Poisson mean must be positive. Predicting the log keeps every finite x legal and turns the gradient into exp(x) − y, which is exactly predicted rate minus observed count. The factorial term of the full NLL does not depend on x, so it is dropped. PoissonNLLLoss is the same type under a name that says the contract out loud.",
        "Poisson ortalaması pozitif olmalıdır. Log tahmin etmek her sonlu x'i yasal kılar ve gradyanı exp(x) − y yapar: tahmin edilen oran eksi gözlenen sayım. Tam NLL'nin faktöriyel terimi x'e bağlı değildir, düşülür. PoissonNLLLoss aynı tiptir; adı sözleşmeyi yüksek sesle söyler.",
      ),
      gradient: bi(
        "∂L/∂x = exp(x) − y. Targets must be ≥ 0. A too-high log-rate produces a large positive gradient and the rate is pulled down.",
        "∂L/∂x = exp(x) − y. Hedefler ≥ 0 olmalıdır. Fazla yüksek bir log-oran büyük pozitif gradyan üretir ve oran aşağı çekilir.",
      ),
      code: `std::vector<double> log_rate{0.0, 1.2};
std::vector<double> counts{1.0, 4.0};
nexusloss::PoissonNLLLoss<double> loss;
double value = loss.forward(log_rate, counts).front();`,
      params: [
        { name: "prediction", detail: bi("Log-rate. The mean is exp(prediction).", "Log-oran. Ortalama exp(tahmin) olur.") },
        { name: "target", detail: bi("Non-negative count.", "Negatif olmayan sayım.") },
      ],
    },
    {
      id: "tweedie",
      name: "Tweedie",
      api: "nexusloss::TweedieLoss<T>",
      formula: "unit-dispersion deviance, 1 < p < 2",
      kind: bi("Element-wise · compound", "Eleman bazlı · bileşik"),
      when: bi(
        "Insurance severity, rainfall, or any target that is usually zero and occasionally a positive amount. Power between 1 and 2 is the compound Poisson-Gamma regime.",
        "Sigorta şiddeti, yağış ya da çoğu zaman sıfır olup ara sıra pozitif bir miktar olan her hedef. 1 ile 2 arasındaki güç, bileşik Poisson-Gamma rejimidir.",
      ),
      logic: bi(
        "Tweedie deviance generalizes Poisson (power → 1) and Gamma (power → 2). The prediction is the positive mean, not a log. Unit dispersion means the library does not estimate a separate scale; you fold that into the learning rate or a later calibration. The zero-target branch drops the y^(2−p) term so a structural zero stays finite.",
        "Tweedie sapması Poisson'u (güç → 1) ve Gamma'yı (güç → 2) geneller. Tahmin log değil, pozitif ortalamadır. Birim dispersiyon, kütüphanenin ayrı bir ölçek tahmin etmediği anlamına gelir; onu öğrenme hızına veya sonraki kalibrasyona katarsın. Sıfır hedef dalı y^(2−p) terimini düşürür, yapısal sıfır sonlu kalır.",
      ),
      gradient: bi(
        "∂L/∂μ = 2 · μ^(−p) · (μ − y). The mean must be strictly positive and the target non-negative.",
        "∂L/∂μ = 2 · μ^(−p) · (μ − y). Ortalama kesin pozitif, hedef negatif olmamalıdır.",
      ),
      code: `std::vector<double> mean{1.4, 0.2};
std::vector<double> claim{0.0, 3.0};
nexusloss::TweedieLoss<double> loss(1.5); // power
double value = loss.forward(mean, claim).front();`,
      params: [
        { name: "power", detail: bi("Open interval (1, 2). Values outside throw.", "(1, 2) açık aralığı. Dışı hata fırlatır.") },
      ],
    },
    {
      id: "cauchy",
      name: "Cauchy",
      api: "nexusloss::CauchyLoss<T>",
      formula: "L = ½ s² log(1 + (e/s)²)",
      kind: bi("Element-wise · heavy tail", "Eleman bazlı · kalın kuyruk"),
      when: bi(
        "Measurements whose errors really do have no variance: impulsive noise, gross annotation mistakes, or a residual you refuse to let dominate.",
        "Hatalarının gerçekten varyansı olmadığı ölçümler: dürtüsel gürültü, kaba etiket hataları ya da hükmetmesine izin vermediğin bir artık.",
      ),
      logic: bi(
        "The Cauchy negative log-likelihood grows only logarithmically. A point ten scales away is not a hundred times more important than a point one scale away. The scale s sets what 'one scale' means in the units of your target.",
        "Cauchy negatif log-olabilirliği yalnızca logaritmik büyür. On ölçek uzaktaki bir nokta, bir ölçek uzaktakinden yüz kat önemli değildir. Ölçek s, hedefin biriminde 'bir ölçek'in ne demek olduğunu belirler.",
      ),
      gradient: bi(
        "∂L/∂x = e / (1 + (e/s)²). Far away the denominator wins and the gradient falls back toward zero. This is a redescending influence, softer than Tukey's hard cutoff.",
        "∂L/∂x = e / (1 + (e/s)²). Uzakta payda kazanır ve gradyan sıfıra geri düşer. Bu, Tukey'nin sert kesiminden daha yumuşak, yeniden alçalan bir etkidir.",
      ),
      code: `nexusloss::CauchyLoss<double> loss(1.0); // scale
double value = loss.forward(prediction, target).front();
auto gradient = loss.backward();`,
      params: [
        { name: "scale", detail: bi("Positive. Default 1.", "Pozitif. Varsayılan 1.") },
      ],
    },
    {
      id: "charbonnier",
      name: "Charbonnier",
      api: "nexusloss::CharbonnierLoss<T>",
      formula: "L = √(e² + ε²)",
      kind: bi("Element-wise · smooth L1", "Eleman bazlı · pürüzsüz L1"),
      when: bi(
        "Image restoration and optical flow, where you want an L1 penalty that is still differentiable at a perfect pixel. ε is a small smoothing constant, not a robustness threshold.",
        "Görüntü onarımı ve optik akış: mükemmel bir pikselde hâlâ türevlenebilir bir L1 cezası istediğinde. ε bir sağlamlık eşiği değil, küçük bir yumuşatma sabitidir.",
      ),
      logic: bi(
        "This is the hypotenuse of the residual and ε. As ε → 0 it becomes MAE. For errors much larger than ε it is almost |e|, so outliers stay linear. Near zero the curvature is 1/ε, which is why a tiny ε makes the bowl very sharp.",
        "Bu, artık ile ε'nin hipotenüsüdür. ε → 0 iken MAE olur. ε'dan çok büyük hatalarda neredeyse |e|'dir; aykırı değerler doğrusal kalır. Sıfıra yakın eğrilik 1/ε'dur, bu yüzden çok küçük ε çanağı keskinleştirir.",
      ),
      gradient: bi(
        "∂L/∂x = e / √(e² + ε²), a smooth sign function. It never exceeds 1 in magnitude.",
        "∂L/∂x = e / √(e² + ε²), pürüzsüz bir işaret fonksiyonu. Büyüklüğü 1'i geçmez.",
      ),
      code: `nexusloss::CharbonnierLoss<float> loss(1e-3f);
auto value = loss.forward(prediction_f, target_f);`,
      params: [
        { name: "epsilon", detail: bi("Positive smoother. Default 1e-3.", "Pozitif yumuşatıcı. Varsayılan 1e-3.") },
      ],
    },
    {
      id: "tukey",
      name: "Tukey Biweight",
      api: "nexusloss::TukeyBiweightLoss<T>",
      formula: "inside c: (c²/6)(1 − (1−r²)³)\noutside: c²/6",
      kind: bi("Element-wise · redescending", "Eleman bazlı · yeniden alçalan"),
      when: bi(
        "The last resort for contamination: once a residual exceeds the scale, it should stop teaching the model entirely. Robust statistics and RANSAC-like regression use this on purpose.",
        "Kirlenme için son çare: artık ölçeği aşınca modele bir şey öğretmeyi tamamen bırakmalıdır. Sağlam istatistik ve RANSAC benzeri regresyon bunu bilinçli kullanır.",
      ),
      logic: bi(
        "Tukey's rho rises smoothly and then becomes perfectly flat. Past the cutoff the loss is a constant, so that point cannot pull the parameters. The classical tuning constant 4.685 keeps about 95% efficiency on clean Gaussian data. If you set the scale too small, good points are ignored and the fit collapses.",
        "Tukey'nin rho'su pürüzsüz yükselir, sonra tamamen düzleşir. Kesimin ötesinde kayıp sabittir; o nokta parametreleri çekemez. Klasik 4.685 sabiti temiz Gauss veride yaklaşık %95 verimlilik korur. Ölçeği fazla küçük seçersen iyi noktalar yok sayılır ve uyum çöker.",
      ),
      gradient: bi(
        "Inside the scale, ∂L/∂x = e (1 − (|e|/c)²)². Outside, the gradient is exactly 0. Mean reduction still divides that zero by N.",
        "Ölçeğin içinde ∂L/∂x = e (1 − (|e|/c)²)². Dışarıda gradyan tam 0'dır. Mean indirgemesi bu sıfırı yine N'ye böler.",
      ),
      code: `nexusloss::TukeyBiweightLoss<double> loss(4.685);
double value = loss.forward(prediction, target).front();
auto gradient = loss.backward(); // far points contribute 0`,
      params: [
        { name: "scale", detail: bi("Positive cutoff in target units. Default 4.685.", "Hedef biriminde pozitif kesim. Varsayılan 4.685.") },
      ],
    },
  ],
};
