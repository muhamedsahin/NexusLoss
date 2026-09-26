import { bi, type CategoryDoc } from "./types";

export const classification: CategoryDoc = {
  id: "classification",
  index: "02",
  title: bi("Classification", "Sınıflandırma"),
  lede: bi(
    "Binary losses that live on LossBase take a vector of scores and targets in lockstep. Multi-class losses live in nexusloss::classification and consume one sample of logits plus a class index. Loop the batch yourself, then reduce.",
    "LossBase üzerindeki ikili kayıplar skor ve hedef vektörlerini eleman eleman alır. Çok sınıflı kayıplar nexusloss::classification içindedir; bir örneğin logitlerini ve sınıf indeksini tüketir. Batch'i sen döngüye al, sonra indirge.",
  ),
  losses: [
    {
      id: "bce",
      name: "Binary cross-entropy",
      api: "nexusloss::BCELoss<T>",
      formula: "L = −[y log p + (1−y) log(1−p)]",
      kind: bi("Probability · LossBase", "Olasılık · LossBase"),
      when: bi(
        "Independent yes/no labels when the network already emitted a probability in (0, 1): a sigmoid head you own, or a calibration layer. If the head emits logits, use BCE with logits instead.",
        "Ağ zaten (0, 1) içinde bir olasılık ürettiyse bağımsız evet/hayır etiketleri: senin sahip olduğun bir sigmoid başı ya da bir kalibrasyon katmanı. Baş logit üretiyorsa logits'li BCE kullan.",
      ),
      logic: bi(
        "Cross-entropy is the negative log probability of the observed bit. It is the unique local proper scoring rule that pushes p toward the conditional frequency of y. Near 0 and 1 the log is steep, so a confident wrong answer is expensive. Targets must sit in [0, 1]; soft labels are legal.",
        "Çapraz entropi, gözlenen bitin negatif log olasılığıdır. p'yi y'nin koşullu frekansına iten, yerel olarak tek uygun skor kuralıdır. 0 ve 1'e yakın log dikleşir; kendinden emin yanlış cevap pahalıdır. Hedefler [0, 1] içinde olmalıdır; yumuşak etiketler yasaldır.",
      ),
      gradient: bi(
        "∂L/∂p = (p − y) / (p(1−p)) inside the open interval. At the numerical edges the gradient is defined as 0 so a saturated probability does not explode.",
        "Açık aralıkta ∂L/∂p = (p − y) / (p(1−p)). Sayısal kenarlarda gradyan 0 tanımlanır; doymuş bir olasılık patlamaz.",
      ),
      code: `std::vector<double> probability{0.9, 0.2};
std::vector<double> target{1.0, 0.0};
nexusloss::BCELoss<double> loss;
double value = loss.forward(probability, target).front();
auto gradient = loss.backward();`,
      params: [
        { name: "prediction", detail: bi("Probability in (0, 1), not a logit.", "(0, 1) içinde olasılık, logit değil.") },
        { name: "target", detail: bi("[0, 1], hard or soft.", "[0, 1], sert veya yumuşak.") },
      ],
    },
    {
      id: "bce-with-logits",
      name: "BCE with logits",
      api: "nexusloss::BCEWithLogitsLoss<T>",
      formula: "L = max(x,0) − x y + log(1 + exp(−|x|))",
      kind: bi("Logit · numerically stable", "Logit · sayısal olarak kararlı"),
      when: bi(
        "The default binary loss. The last layer is linear. Combining sigmoid and BCE in one function keeps log(sigmoid(x)) finite even when x is ±100.",
        "Varsayılan ikili kayıp. Son katman doğrusaldır. Sigmoid ile BCE'yi tek fonksiyonda birleştirmek, x ±100 olsa bile log(sigmoid(x))'i sonlu tutar.",
      ),
      logic: bi(
        "The algebra is identical to BCE(sigmoid(x), y). The rewritten form never materializes a 0 or 1 probability, so the log does not hit −inf during the forward pass. The gradient collapses to sigmoid(x) − y, which is why a logit head trained this way behaves like a probability head without you writing the sigmoid.",
        "Cebir, BCE(sigmoid(x), y) ile aynıdır. Yeniden yazılmış biçim 0 veya 1 olasılığı üretmez; ileri geçişte log −inf'e çarpmaz. Gradyan sigmoid(x) − y'ye iner. Bu yüzden bu kayıpla eğitilen bir logit başı, sigmoid yazmadan bir olasılık başı gibi davranır.",
      ),
      gradient: bi(
        "∂L/∂x = σ(x) − y, then scaled by the reduction. This is the form you want in a training step.",
        "∂L/∂x = σ(x) − y, sonra indirgeme ölçeği. Eğitim adımında istediğin biçim budur.",
      ),
      code: `std::vector<double> logits{2.0, -1.5};
std::vector<double> target{1.0, 0.0};
nexusloss::BCEWithLogitsLoss<double> loss;
double value = loss.forward(logits, target).front();
auto gradient = loss.backward();`,
      params: [
        { name: "prediction", detail: bi("Raw logit. Do not sigmoid it first.", "Ham logit. Önce sigmoid uygulama.") },
      ],
    },
    {
      id: "cross-entropy",
      name: "Cross-entropy",
      api: "classification::CrossEntropyLoss<T>",
      formula: "L = log Σ exp(z) − z_c",
      kind: bi("One sample · logits", "Tek örnek · logit"),
      when: bi(
        "Mutually exclusive classes: one image, one digit, one intent. The vector is logits for a single example and the target is a class index. For a batch, call it once per row and average.",
        "Birbirini dışlayan sınıflar: bir görüntü, bir rakam, bir niyet. Vektör tek örneğin logitleridir, hedef sınıf indeksidir. Batch için satır satır çağır ve ortala.",
      ),
      logic: bi(
        "Softmax turns logits into a distribution; negative log-likelihood then charges the model for the probability it assigned to the true class. NexusLoss never builds the softmax explicitly in the forward value: log-sum-exp subtracts the max logit first, so exp(1000) becomes exp(0). Label smoothing is an optional constructor argument in [0, 1). The free functions categorical_cross_entropy and sparse_categorical_cross_entropy are the same math without the cached backward.",
        "Softmax logitleri dağılıma çevirir; negatif log-olabilirlik modeli gerçek sınıfa verdiği olasılık için ücretlendirir. NexusLoss ileri değerde softmax'ı açıkça kurmaz: log-sum-exp önce en büyük logiti çıkarır, exp(1000) exp(0) olur. Etiket yumuşatma, [0, 1) aralığında isteğe bağlı kurucu argümanıdır. categorical_cross_entropy ve sparse_categorical_cross_entropy aynı matematiğin önbelleksiz halidir.",
      ),
      gradient: bi(
        "∂L/∂z = softmax(z) − one_hot(c), with the smoothed target if you passed a smoothing value. Calling backward() before forward() throws.",
        "∂L/∂z = softmax(z) − one_hot(c); yumuşatma verdiysen hedef yumuşatılmış halidir. forward() olmadan backward() hata fırlatır.",
      ),
      code: `std::vector<double> logits{1.0, 2.0, 3.0};
nexusloss::classification::CrossEntropyLoss<double> loss; // smoothing 0
double value = loss.forward(logits, /*class*/ 2);
auto gradient = loss.backward(); // softmax - [0, 0, 1]

double bare = nexusloss::classification::sparse_categorical_cross_entropy<double>(logits, 2);`,
      params: [
        { name: "label_smoothing", detail: bi("Default 0. Must be in [0, 1).", "Varsayılan 0. [0, 1) içinde olmalı.") },
        { name: "target", detail: bi("Class index, not a one-hot vector.", "Tek-sıcak vektör değil, sınıf indeksi.") },
      ],
    },
    {
      id: "label-smoothing",
      name: "Label smoothing",
      api: "classification::LabelSmoothingCrossEntropyLoss<T>",
      formula: "target = (1−ε) e_c + ε / C",
      kind: bi("One sample · regularized CE", "Tek örnek · düzenlenmiş CE"),
      when: bi(
        "When the model is over-confident and you would rather it keep a little probability on the other classes. Fine-tuning classifiers, distillation students and noisy single-label data.",
        "Model fazla kendinden eminse ve diğer sınıflarda biraz olasılık bırakmasını tercih ediyorsan. Sınıflandırıcı ince ayarı, damıtma öğrencileri ve gürültülü tek etiketli veri.",
      ),
      logic: bi(
        "A hard one-hot asks the softmax to drive the true logit to infinity. Smoothing replaces that target with a mixture: most of the mass stays on the labeled class, a uniform crumb is shared by everyone. The network is penalized for probabilities near 1, which is a form of entropy regularization. This class is a thin wrapper around CrossEntropyLoss with default ε = 0.1.",
        "Sert tek-sıcak, softmax'tan gerçek logiti sonsuza sürmesini ister. Yumuşatma o hedefi bir karışımla değiştirir: kütlenin çoğu etiketli sınıfta kalır, herkese eşit bir kırıntı dağıtılır. Ağ, 1'e yakın olasılıklar için cezalandırılır; bu bir entropi düzenlemesidir. Bu sınıf, varsayılan ε = 0.1 ile CrossEntropyLoss üzerine ince bir sarmalayıcıdır.",
      ),
      gradient: bi(
        "Same softmax-minus-target gradient, but the target is no longer a pure one-hot, so the true class is not asked to absorb the entire residual.",
        "Aynı softmax-eksi-hedef gradyanı, fakat hedef artık saf tek-sıcak değildir; gerçek sınıftan artığın tamamını emmesi istenmez.",
      ),
      code: `nexusloss::classification::LabelSmoothingCrossEntropyLoss<double> loss(0.1);
double value = loss.forward(logits, 2);
auto gradient = loss.backward();`,
      params: [
        { name: "smoothing", detail: bi("ε in [0, 1). Default 0.1.", "ε, [0, 1). Varsayılan 0.1.") },
      ],
    },
    {
      id: "kl",
      name: "KL divergence",
      api: "classification::KLDivLoss<T>",
      formula: "L = Σ y (log y − log p)",
      kind: bi("Distributions · probabilities", "Dağılımlar · olasılıklar"),
      when: bi(
        "When the target is itself a distribution: soft labels, a frozen teacher at temperature 1, or a prior you want the prediction to match. For temperature-scaled logits, use distillation instead.",
        "Hedefin kendisi bir dağılımsa: yumuşak etiketler, sıcaklığı 1 olan donmuş bir öğretmen ya da tahminin uymasını istediğin bir önsel. Sıcaklıkla ölçeklenmiş logitler için damıtmayı kullan.",
      ),
      logic: bi(
        "KL(y ‖ p) measures how many extra nats you pay for coding samples from y with the model p. It is zero only when p = y, and it is not symmetric. Both tensors are probabilities. A zero in p with a positive y is infinite, which is the mathematical way of saying you assigned no mass to something that happens.",
        "KL(y ‖ p), y'den gelen örnekleri p modeli ile kodlarken kaç nat fazla ödediğini ölçer. Yalnızca p = y iken sıfırdır ve simetrik değildir. İki tensör de olasılıktır. Pozitif y varken p'de sıfır olması sonsuzdur; olan bir şeye hiç kütle vermediğinin matematiksel halidir.",
      ),
      gradient: bi(
        "∂L/∂p_i = −y_i / p_i. This is with respect to the probability, not the logit. If you need a logit gradient, differentiate through softmax yourself or call distillation::kl_divergence.",
        "∂L/∂p_i = −y_i / p_i. Bu, logite değil olasılığa göredir. Logit gradyanı istiyorsan softmax üzerinden kendin türet ya da distillation::kl_divergence çağır.",
      ),
      code: `std::vector<double> prediction{0.7, 0.2, 0.1};
std::vector<double> teacher{0.6, 0.3, 0.1};
nexusloss::classification::KLDivLoss<double> loss;
double value = loss.forward(prediction, teacher);
auto gradient = loss.backward();`,
      params: [
        { name: "prediction", detail: bi("Probability vector. Not logits.", "Olasılık vektörü. Logit değil.") },
        { name: "target", detail: bi("Probability vector on the same simplex.", "Aynı simpleks üzerinde olasılık vektörü.") },
      ],
    },
    {
      id: "hinge",
      name: "Hinge",
      api: "nexusloss::HingeLoss<T>",
      formula: "L = max(0, 1 − y x)",
      kind: bi("Margin · labels ±1", "Marj · etiketler ±1"),
      when: bi(
        "Linear SVMs and any binary head where you care about a margin, not a calibrated probability. The target encoding is −1 and +1, never 0 and 1.",
        "Doğrusal SVM'ler ve kalibre olasılık değil marj istediğin her ikili baş. Etiket kodu −1 ve +1'dir, asla 0 ve 1 değil.",
      ),
      logic: bi(
        "Once the signed score y·x clears 1, the loss is exactly zero and that example stops contributing. The '1' is the margin. Points inside the margin or on the wrong side of zero keep pushing. This is why hinge can be sparse in its gradient: easy examples are ignored.",
        "İşaretli skor y·x, 1'i geçince kayıp tam sıfır olur ve o örnek katkı vermeyi keser. '1' marjdır. Marjın içinde veya sıfırın yanlış tarafındaki noktalar itmeye devam eder. Hinge gradyanının seyrek olmasının nedeni budur: kolay örnekler yok sayılır.",
      ),
      gradient: bi(
        "∂L/∂x = −y when y x < 1, and 0 otherwise. A satisfied margin produces a silent gradient.",
        "y x < 1 iken ∂L/∂x = −y, aksi halde 0. Tatmin edilmiş bir marj sessiz gradyan üretir.",
      ),
      code: `std::vector<double> score{0.2, 1.4, -0.5};
std::vector<double> label{1.0, 1.0, -1.0}; // not {0, 1}
nexusloss::HingeLoss<double> loss;
double value = loss.forward(score, label).front();`,
      params: [
        { name: "target", detail: bi("Must be −1 or +1.", "−1 veya +1 olmalıdır.") },
      ],
    },
    {
      id: "squared-hinge",
      name: "Squared hinge",
      api: "nexusloss::SquaredHingeLoss<T>",
      formula: "L = max(0, 1 − y x)²",
      kind: bi("Margin · smooth inside", "Marj · içeride pürüzsüz"),
      when: bi(
        "Same ±1 labels as hinge, when you want the violating points to be punished more than linearly and the kink at the margin to be milder for optimization.",
        "Hinge ile aynı ±1 etiketler; ihlal eden noktaların doğrusal olandan sert cezalandırılmasını ve marjdaki kırığın optimizasyon için daha yumuşak olmasını istediğinde.",
      ),
      logic: bi(
        "Squaring the hinge residual keeps the loss zero outside the margin and makes the penalty grow faster as an example falls deeper into the wrong region. The gradient is continuous at the margin (it hits zero from both sides), which squared hinge buys and plain hinge does not.",
        "Hinge artığının karesi, marjın dışında kaybı sıfır tutar ve örnek yanlış bölgeye battıkça cezayı daha hızlı büyütür. Gradyan marjda süreklidir (iki yandan da sıfıra değer); bunu karesel hinge satın alır, düz hinge almaz.",
      ),
      gradient: bi(
        "∂L/∂x = −2 y max(0, 1 − y x). Satisfied points still contribute nothing.",
        "∂L/∂x = −2 y max(0, 1 − y x). Tatmin olan noktalar yine hiçbir şey katmaz.",
      ),
      code: `nexusloss::SquaredHingeLoss<double> loss;
double value = loss.forward(score, label).front();
auto gradient = loss.backward();`,
      params: [
        { name: "target", detail: bi("−1 / +1, same as hinge.", "Hinge ile aynı, −1 / +1.") },
      ],
    },
    {
      id: "exponential",
      name: "Exponential",
      api: "nexusloss::ExponentialLoss<T>",
      formula: "L = exp(−y x)",
      kind: bi("Boosting · ±1 labels", "Boosting · ±1 etiket"),
      when: bi(
        "AdaBoost-style objectives, where a wrong answer must become exponentially more urgent and there is no margin at which the example is allowed to rest.",
        "AdaBoost tarzı amaçlar: yanlış cevap üstel olarak daha acil olmalı ve örneğin dinlenmesine izin verilen bir marj yoktur.",
      ),
      logic: bi(
        "Unlike hinge, exponential loss never reaches zero. A more confident correct answer keeps reducing the loss, and a wrong answer explodes. That explosion is the point in boosting (hard examples get the next weak learner) and the hazard in a deep net (one bad logit dominates the batch).",
        "Hinge'in aksine üstel kayıp sıfıra hiç inmez. Daha emin doğru cevap kaybı azaltmaya devam eder, yanlış cevap patlar. Bu patlama boosting'de amaçtır (zor örnekler sonraki zayıf öğreniciyi alır) ve derin ağda tehlikedir (tek kötü logit batch'e hükmeder).",
      ),
      gradient: bi(
        "∂L/∂x = −y exp(−y x). The same exponential that defines the loss multiplies the step.",
        "∂L/∂x = −y exp(−y x). Kaybı tanımlayan aynı üstel, adımı da çarpar.",
      ),
      code: `nexusloss::ExponentialLoss<double> loss;
double value = loss.forward(score, label).front();`,
      params: [
        { name: "target", detail: bi("−1 / +1.", "−1 / +1.") },
      ],
    },
    {
      id: "perceptron",
      name: "Perceptron",
      api: "nexusloss::PerceptronLoss<T>",
      formula: "L = max(0, −y x)",
      kind: bi("Margin 0 · ±1 labels", "Marj 0 · ±1 etiket"),
      when: bi(
        "A historical baseline and a useful ablation: the model is only punished for being on the wrong side of zero. There is no demand for confidence.",
        "Tarihsel bir taban çizgisi ve yararlı bir ablasyon: model yalnızca sıfırın yanlış tarafında olduğu için cezalandırılır. Güven talebi yoktur.",
      ),
      logic: bi(
        "This is hinge with the margin removed. Any positive signed score is perfect. The decision boundary can sit arbitrarily close to the data, which is why the perceptron criterion separates when a separation exists but does not prefer a wide gap.",
        "Bu, marjı çıkarılmış hinge'dir. Pozitif işaretli her skor mükemmeldir. Karar sınırı veriye keyfi yakın oturabilir; bu yüzden perceptron ölçütü ayrım varsa ayırır ama geniş bir aralık tercih etmez.",
      ),
      gradient: bi(
        "∂L/∂x = −y when y x < 0, else 0. Correctly signed points are invisible to the update.",
        "y x < 0 iken ∂L/∂x = −y, değilse 0. Doğru işaretli noktalar güncellemeye görünmez.",
      ),
      code: `nexusloss::PerceptronLoss<double> loss;
double value = loss.forward(score, label).front();`,
      params: [
        { name: "target", detail: bi("−1 / +1.", "−1 / +1.") },
      ],
    },
    {
      id: "zero-one",
      name: "Zero-one",
      api: "nexusloss::ZeroOneLoss<T>",
      formula: "L = 1[sign(x) ≠ y]",
      kind: bi("Metric · not for training", "Metrik · eğitim için değil"),
      when: bi(
        "Report it. Do not train with it. It is the classification error you actually care about at evaluation time, exposed with the same LossBase shape so a loop can log it beside the training loss.",
        "Raporla. Bununla eğitme. Değerlendirmede gerçekten önemsediğin sınıflandırma hatasıdır; bir döngünün onu eğitim kaybının yanında loglayabilmesi için aynı LossBase biçiminde durur.",
      ),
      logic: bi(
        "The 0-1 loss is constant almost everywhere and jumps on the decision boundary. No local gradient points toward a better classifier. NexusLoss documents that fact by returning an explicit zero subgradient instead of pretending a slope exists.",
        "0-1 kaybı hemen her yerde sabittir ve karar sınırında zıplar. Hiçbir yerel gradyan daha iyi bir sınıflandırıcıya işaret etmez. NexusLoss bunu, bir eğim varmış gibi davranmak yerine açık bir sıfır alt-gradyan döndürerek belgeler.",
      ),
      gradient: bi(
        "The analytical gradient is the zero vector. An optimizer step on it changes nothing.",
        "Analitik gradyan sıfır vektörüdür. Üzerinde bir optimize edici adımı hiçbir şeyi değiştirmez.",
      ),
      code: `nexusloss::ZeroOneLoss<double> metric;
double error = metric.forward(score, label).front();
auto silent = metric.backward(); // all zeros`,
      params: [
        { name: "target", detail: bi("−1 / +1, compared with the sign of the score.", "−1 / +1, skorun işaretiyle karşılaştırılır.") },
      ],
    },
    {
      id: "focal",
      name: "Focal",
      api: "nexusloss::FocalLoss<T>",
      formula: "L = −α_t (1 − p_t)^γ log(p_t)",
      kind: bi("Probability · class imbalance", "Olasılık · sınıf dengesizliği"),
      when: bi(
        "Dense binary problems where negatives drown the loss: foreground pixels, rare events, or any sigmoid head whose easy background examples would otherwise own the gradient. For detection logits, prefer DetectionFocalLoss.",
        "Negatiflerin kaybı boğduğu yoğun ikili problemler: ön plan pikselleri, nadir olaylar ya da kolay arka plan örneklerinin gradyanı sahiplenmesin diye sigmoid başı. Tespit logitleri için DetectionFocalLoss'u tercih et.",
      ),
      logic: bi(
        "Lin et al. multiply BCE by (1 − p_t)^γ. When the model is already right, p_t is near 1, the factor vanishes, and that easy example goes quiet. γ = 0 recovers weighted BCE. α_t rebalances the positive and negative class; the positive uses α, the negative uses 1 − α. This class expects probabilities. A positive target is any value ≥ 0.5.",
        "Lin ve arkadaşları BCE'yi (1 − p_t)^γ ile çarpar. Model zaten haklıysa p_t 1'e yakındır, çarpan yok olur ve kolay örnek susar. γ = 0 ağırlıklı BCE'yi geri getirir. α_t pozitif ve negatif sınıfı yeniden dengeler; pozitif α, negatif 1 − α kullanır. Bu sınıf olasılık bekler. Pozitif hedef, ≥ 0.5 olan her değerdir.",
      ),
      gradient: bi(
        "Differentiated through p_t. If γ = 0 the extra modulating term drops. Saturated probabilities (within epsilon of 0 or 1) contribute a zero gradient.",
        "p_t üzerinden türetilir. γ = 0 ise ek modülasyon terimi düşer. Doymuş olasılıklar (0 veya 1'e epsilon kadar yakın) sıfır gradyan verir.",
      ),
      code: `std::vector<double> probability{0.8, 0.05};
std::vector<double> target{1.0, 0.0};
nexusloss::FocalLoss<double> loss(0.25, 2.0); // alpha, gamma
double value = loss.forward(probability, target).front();`,
      params: [
        { name: "alpha", detail: bi("In [0, 1]. Default 0.25. Weight on the positive class.", "[0, 1] içinde. Varsayılan 0.25. Pozitif sınıfın ağırlığı.") },
        { name: "gamma", detail: bi("≥ 0. Default 2. Higher ignores easy examples more.", "≥ 0. Varsayılan 2. Yükseldikçe kolay örnekleri daha çok yok sayar.") },
      ],
    },
    {
      id: "categorical-focal",
      name: "Categorical focal",
      api: "classification::categorical_focal_loss",
      formula: "−α (1 − p_c)^γ log(p_c)",
      kind: bi("Free function · one sample", "Serbest fonksiyon · tek örnek"),
      when: bi(
        "Single-label multi-class problems with a long tail of rare classes, when a softmax focal term is clearer than training C independent sigmoids.",
        "Nadir sınıfların uzun kuyruğu olan tek etiketli çok sınıflı problemler; C tane bağımsız sigmoid eğitmektense bir softmax focal terimi daha açıksa.",
      ),
      logic: bi(
        "The function softmaxes the logits, reads p_c for the true class, and applies the focal modulator only there. Easy classes that the model already ranks first contribute little. It returns a scalar; there is no cached backward on this free function. For a training step with gradients, the binary FocalLoss and DetectionFocalLoss classes are the ones that implement backward().",
        "Fonksiyon logitlere softmax uygular, gerçek sınıfın p_c değerini okur ve focal modülatörü yalnızca orada uygular. Modelin zaten birinci sıraya koyduğu kolay sınıflar az katkı verir. Skaler döner; bu serbest fonksiyonun önbellekli backward'ı yoktur. Gradyanlı bir eğitim adımı için backward() uygulayanlar ikili FocalLoss ve DetectionFocalLoss sınıflarıdır.",
      ),
      gradient: bi(
        "Not cached. Use the value for logging or for a custom derivative. The class-based focal losses are the training path.",
        "Önbelleklenmez. Değeri loglamak veya özel bir türev için kullan. Eğitim yolu, sınıf tabanlı focal kayıplardır.",
      ),
      code: `double value = nexusloss::classification::categorical_focal_loss<double>(
    logits, /*class*/ 2, /*alpha*/ 1.0, /*gamma*/ 2.0);`,
      params: [
        { name: "alpha", detail: bi("Default 1.", "Varsayılan 1.") },
        { name: "gamma", detail: bi("Default 2.", "Varsayılan 2.") },
      ],
    },
    {
      id: "gce",
      name: "Generalized CE",
      api: "classification::generalized_cross_entropy",
      formula: "L = (1 − p_t^q) / q",
      kind: bi("Free function · noisy labels", "Serbest fonksiyon · gürültülü etiket"),
      when: bi(
        "Labels you do not fully trust. q near 0 behaves like cross-entropy; q near 1 behaves like MAE on the probability and stops chasing a wrong label quite so hard.",
        "Tam güvenmediğin etiketler. q 0'a yakınsa çapraz entropi gibi davranır; q 1'e yakınsa olasılık üzerinde MAE gibi davranır ve yanlış etiketi o kadar sert kovalamaz.",
      ),
      logic: bi(
        "Zhang and Sabuncu's GCE is the Box-Cox transform of the true-class probability. Cross-entropy's gradient blows up as p_t → 0, which is exactly when a corrupted label does the most damage. Raising q flattens that spike. The function takes a probability and a target in [0, 1], not logits.",
        "Zhang ve Sabuncu'nun GCE'si, gerçek sınıf olasılığının Box-Cox dönüşümüdür. Çapraz entropinin gradyanı p_t → 0 iken patlar; bozuk bir etiketin en çok zarar verdiği yer tam orasıdır. q'yu yükseltmek o sivriyi yassılaştırır. Fonksiyon logit değil, bir olasılık ve [0, 1] içinde bir hedef alır.",
      ),
      gradient: bi(
        "The free function returns the scalar only. The idea of the derivative is −p_t^(q−1) with respect to p_t, which stays bounded for q > 0.",
        "Serbest fonksiyon yalnızca skaler döner. Türevin fikri p_t'ye göre −p_t^(q−1)'dir; q > 0 için sınırlı kalır.",
      ),
      code: `double value = nexusloss::classification::generalized_cross_entropy<double>(
    /*probability*/ 0.2, /*target*/ 1.0, /*q*/ 0.7);`,
      params: [
        { name: "q", detail: bi("Default 0.7. Must keep the loss defined; small q ≈ CE.", "Varsayılan 0.7. Küçük q ≈ CE.") },
      ],
    },
    {
      id: "symmetric-ce",
      name: "Symmetric CE",
      api: "classification::symmetric_cross_entropy",
      formula: "α CE(p, y) + β CE(y, p)",
      kind: bi("Free function · noisy labels", "Serbest fonksiyon · gürültülü etiket"),
      when: bi(
        "Another noisy-label defense. The reverse cross-entropy term is bounded, so a completely wrong hard label cannot send the loss to infinity by itself.",
        "Bir başka gürültülü etiket savunması. Ters çapraz entropi terimi sınırlıdır; tamamen yanlış sert bir etiket kaybı tek başına sonsuza itemez.",
      ),
      logic: bi(
        "Ordinary CE trusts y completely. Symmetric CE adds the cross-entropy written in the other direction, which for a one-hot y becomes a penalty on how far p sits from that vertex, clipped by the fact that CE(y, p) only reads the true-class term when y is one-hot... wait, the implementation takes probability and target scalars for the binary form: α times BCE plus β times the reverse. The reverse term punishes the label for disagreeing with a confident model, which is what you want if some labels are the thing that is wrong.",
        "Sıradan CE, y'ye tamamen güvenir. Simetrik CE diğer yönde yazılmış çapraz entropiyi ekler. Uygulama ikili biçim için olasılık ve hedef skalerleri alır: α katı BCE artı β katı ters yön. Ters terim, kendinden emin bir modelle uyuşmayan etiketi cezalandırır; yanlış olan şey etiketlerin bir kısmıysa istediğin budur.",
      ),
      gradient: bi(
        "Scalar helper. Combine it with a class loss if you need a backward() cache. α and β default to 1.",
        "Skaler yardımcı. backward() önbelleği istiyorsan bir sınıf kaybıyla birleştir. α ve β varsayılanı 1'dir.",
      ),
      code: `double value = nexusloss::classification::symmetric_cross_entropy<double>(
    0.8, 1.0, /*alpha*/ 1.0, /*beta*/ 1.0);`,
      params: [
        { name: "alpha", detail: bi("Weight on CE(p, y). Default 1.", "CE(p, y) ağırlığı. Varsayılan 1.") },
        { name: "beta", detail: bi("Weight on the reverse term. Default 1.", "Ters terimin ağırlığı. Varsayılan 1.") },
      ],
    },
    {
      id: "weighted-ce",
      name: "Weighted & balanced CE",
      api: "classification::weighted_*  ·  effective_number_class_weights",
      formula: "w_c · CE    or    w₊ y + w₋ (1−y)",
      kind: bi("Free functions · imbalance", "Serbest fonksiyonlar · dengesizlik"),
      when: bi(
        "You know the class counts and want the rare class to punch above its frequency, without switching the whole objective to focal loss.",
        "Sınıf sayılarını biliyorsun ve tüm amacı focal kayba çevirmeden nadir sınıfın frekansının üstünde vurmasını istiyorsun.",
      ),
      logic: bi(
        "weighted_binary_cross_entropy scales the positive and negative log terms separately. weighted_categorical_cross_entropy scales the whole per-example CE by the true class weight, with optional label smoothing. effective_number_class_weights turns raw counts into Cui et al.'s effective-number weights: (1 − β) / (1 − β^{n_c}), then normalizes them to sum to the number of classes. β near 1 treats frequent classes as almost redundant.",
        "weighted_binary_cross_entropy pozitif ve negatif log terimlerini ayrı ölçekler. weighted_categorical_cross_entropy örnek başına CE'yi gerçek sınıf ağırlığıyla ölçekler; etiket yumuşatma isteğe bağlıdır. effective_number_class_weights ham sayıları Cui ve arkadaşlarının etkin sayı ağırlıklarına çevirir: (1 − β) / (1 − β^{n_c}), sonra sınıfların sayısına toplanacak şekilde normalize eder. β 1'e yakınsa sık sınıfları neredeyse tekrar sayar.",
      ),
      gradient: bi(
        "These helpers return values or a weight vector. Feed the weights into a LossBase forward() if you want them inside the cached gradient, or multiply a cross-entropy gradient by w_c yourself.",
        "Bu yardımcılar değer veya bir ağırlık vektörü döner. Önbellekli gradyanın içine girmelerini istiyorsan ağırlıkları bir LossBase forward()'una ver, ya da çapraz entropi gradyanını w_c ile kendin çarp.",
      ),
      code: `double bce = nexusloss::classification::weighted_binary_cross_entropy<double>(
    0.7, 1.0, /*positive*/ 4.0, /*negative*/ 1.0);

std::vector<size_t> counts{1000, 50, 20};
auto weights = nexusloss::classification::effective_number_class_weights<double>(counts, 0.999);
double ce = nexusloss::classification::weighted_categorical_cross_entropy<double>(
    logits, 2, weights);`,
      params: [
        { name: "class weights", detail: bi("Non-negative, one per class, same length as the logits.", "Negatif değil, sınıf başına bir tane, logitlerle aynı uzunlukta.") },
        { name: "beta", detail: bi("Effective-number hyperparameter. Default 0.999. Counts must be positive.", "Etkin sayı hiperparametresi. Varsayılan 0.999. Sayımlar pozitif olmalı.") },
      ],
    },
  ],
};
