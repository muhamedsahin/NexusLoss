import { bi, type CategoryDoc } from "./types";

export const distillation: CategoryDoc = {
  id: "distillation",
  index: "09",
  title: bi("Distillation", "Damıtma"),
  lede: bi(
    "A frozen teacher and a student that should imitate it. Logit distillation uses a temperature. Feature and attention losses assume you already aligned the tensor shapes; the library does not invent a projection matrix.",
    "Donmuş bir öğretmen ve onu taklit etmesi gereken bir öğrenci. Logit damıtması bir sıcaklık kullanır. Özellik ve dikkat kayıpları tensör şekillerini çoktan hizaladığını varsayar; kütüphane bir izdüşüm matrisi uydurmaz.",
  ),
  losses: [
    {
      id: "kd",
      name: "Knowledge distillation",
      api: "distillation::KnowledgeDistillationLoss<T>",
      formula: "T² · KL(softmax(z_t/T) ‖ softmax(z_s/T))",
      kind: bi("Logits · temperature", "Logit · sıcaklık"),
      when: bi(
        "Hinton distillation. The student should match the teacher's dark knowledge, not only the hard label. Combine this scalar with a ground-truth cross-entropy outside the class.",
        "Hinton damıtması. Öğrenci yalnızca sert etiketi değil, öğretmenin karanlık bilgisini eşlemelidir. Bu skaleri sınıfın dışında bir gerçek çapraz entropi ile birleştir.",
      ),
      logic: bi(
        "Dividing logits by T softens both distributions so the small probabilities, the runner-up classes, become visible. Multiplying the KL by T² keeps the gradient magnitude comparable to a hard-label loss as T changes. T = 1 is ordinary KL on softmaxes. The teacher logits are targets.",
        "Logitleri T'ye bölmek iki dağılımı yumuşatır; küçük olasılıklar, ikinci gelen sınıflar görünür olur. KL'yi T² ile çarpmak, T değişirken gradyan büyüklüğünü sert etiket kaybıyla karşılaştırılabilir tutar. T = 1, softmax'lar üzerinde sıradan KL'dir. Öğretmen logitleri hedeftir.",
      ),
      gradient: bi(
        "backward() is with respect to the student logits: T · (softmax(z_s/T) − softmax(z_t/T)). The teacher side is not returned.",
        "backward() öğrenci logitlerine göredir: T · (softmax(z_s/T) − softmax(z_t/T)). Öğretmen tarafı döndürülmez.",
      ),
      code: `nexusloss::distillation::KnowledgeDistillationLoss<double> loss(4.0); // temperature
double value = loss.forward(student_logits, teacher_logits);
auto gradient = loss.backward();`,
      params: [
        { name: "temperature", detail: bi("Default 1. Larger is softer.", "Varsayılan 1. Büyük olan daha yumuşaktır.") },
      ],
    },
    {
      id: "feature-distill",
      name: "Feature distillation",
      api: "distillation::FeatureDistillationLoss<T>",
      formula: "mean (student − teacher)²",
      kind: bi("Matched features", "Eşlenmiş özellikler"),
      when: bi(
        "Hint layers. The student should reproduce an intermediate map, not only the final logits. Fit or Romero-style, once the widths match.",
        "İpucu katmanları. Öğrenci yalnızca son logitleri değil, bir ara haritayı da yeniden üretmelidir. Genişlikler örtüşünce FitNet veya Romero tarzı.",
      ),
      logic: bi(
        "Mean squared error between two flattened tensors. If the student is narrower, learn a 1×1 projection in the model and pass the projected vector. The teacher is a constant. This is the same geometry as perceptual loss, with the roles named student and teacher so a training loop reads clearly.",
        "İki düzleştirilmiş tensör arasında ortalama karesel hata. Öğrenci daha darsa, modelde 1×1 bir izdüşüm öğren ve izdüşümlü vektörü ver. Öğretmen sabittir. Bu, algısal kayıpla aynı geometridir; roller öğrenci ve öğretmen diye adlandırılmıştır, eğitim döngüsü açık okunur.",
      ),
      gradient: bi(
        "∂L/∂student = 2 (student − teacher) / N.",
        "∂L/∂öğrenci = 2 (öğrenci − öğretmen) / N.",
      ),
      code: `nexusloss::distillation::FeatureDistillationLoss<double> loss;
double value = loss.forward(student_features, teacher_features);
auto gradient = loss.backward();`,
      params: [
        { name: "features", detail: bi("Equal length. Align channels before the call.", "Eşit uzunluk. Çağrıdan önce kanalları hizala.") },
      ],
    },
    {
      id: "attention-transfer",
      name: "Attention transfer",
      api: "distillation::AttentionTransferLoss<T>",
      formula: "mean ‖ a_s/‖a_s‖ − a_t/‖a_t‖ ‖²",
      kind: bi("Activation maps · normalized", "Aktivasyon haritaları · normalize"),
      when: bi(
        "Zagoruyko attention transfer. Match where the network looks, after collapsing a map to a spatial attention vector, without forcing the raw magnitudes to agree.",
        "Zagoruyko dikkat aktarımı. Ham büyüklüklerin uyuşmasını zorlamadan, bir haritayı uzamsal bir dikkat vektörüne indirdikten sonra ağın nereye baktığını eşle.",
      ),
      logic: bi(
        "Each vector is divided by its L2 norm, then the MSE of those directions is taken. A student that attends to the same places with a different gain pays nothing. You build the attention vector (sum of squared maps, or whatever the paper's p-norm was) before calling. Empty or zero maps are guarded by the norm.",
        "Her vektör L2 normuna bölünür, sonra bu yönlerin MSE'si alınır. Aynı yerlere farklı bir kazançla bakan öğrenci hiçbir şey ödemez. Dikkat vektörünü (kare haritaların toplamı ya da makaledeki p-norm neyse) çağırmadan önce kurarsın. Boş veya sıfır haritalar norm tarafından korunur.",
      ),
      gradient: bi(
        "The derivative of a normalized vector: the residual minus its projection back onto the student direction, divided by the student norm.",
        "Normalize bir vektörün türevi: artık eksi öğrencinin yönüne geri izdüşümü, öğrenci normuna bölünmüş.",
      ),
      code: `nexusloss::distillation::AttentionTransferLoss<double> loss;
double value = loss.forward(student_attention, teacher_attention);
auto gradient = loss.backward();`,
      params: [
        { name: "maps", detail: bi("Equal-length attention vectors, not raw feature volumes.", "Eşit uzunlukta dikkat vektörleri, ham özellik hacimleri değil.") },
      ],
    },
  ],
};

export const sequence: CategoryDoc = {
  id: "sequence",
  index: "10",
  title: bi("Sequences", "Diziler"),
  lede: bi(
    "CTC aligns an unsegmented input with a short label. Masked cross-entropy trains the tokens you marked and leaves the pads alone. Next-sentence prediction is one logit and one bit, the BERT auxiliary.",
    "CTC, bölütlenmemiş bir girdiyi kısa bir etiketle hizalar. Maskeli çapraz entropi işaretlediğin token'ları eğitir, dolguları rahat bırakır. Sonraki cümle tahmini bir logit ve bir bittir; BERT'in yardımcı kaybı.",
  ),
  losses: [
    {
      id: "ctc",
      name: "CTC",
      api: "sequence::CTCLoss<T>",
      formula: "−log Σ_{alignments} Π_t p_t(a_t)",
      kind: bi("Time-major logits", "Zaman-major logit"),
      when: bi(
        "Speech and handwriting, whenever the input is longer than the label and you do not have frame-level alignment. The blank index absorbs the slack.",
        "Konuşma ve el yazısı: girdi etiketten uzunsa ve kare düzeyinde hizalaman yoksa. Boş indeks gevşekliği emer.",
      ),
      logic: bi(
        "Graves et al. sum the probability of every way to insert blanks and repeat labels so the alignment matches the input length. NexusLoss runs the forward-backward algorithm in log space. Logits are packed time-major, index = time * classes + class. The label vector does not contain blanks. The default blank class is 0. The loss is the negative log total probability of the label.",
        "Graves ve arkadaşları, hizalama girdi uzunluğuna uysun diye boşluk eklemenin ve etiket tekrarının her yolunun olasılığını toplar. NexusLoss ileri-geri algoritmasını log uzayında çalıştırır. Logitler zaman-major paketlenir, indeks = zaman * sınıf + sınıf. Etiket vektörü boşluk içermez. Varsayılan boş sınıf 0'dır. Kayıp, etiketin toplam olasılığının negatif logudur.",
      ),
      gradient: bi(
        "backward() returns ∂L/∂logits for every time and class: the occupancy posterior minus the softmax, the standard CTC gradient.",
        "backward() her zaman ve sınıf için ∂L/∂logit döndürür: doluluk sonsalı eksi softmax, standart CTC gradyanı.",
      ),
      code: `std::vector<double> logits = /* T * C */;
std::vector<size_t> labels{1, 4, 2}; // no blanks
nexusloss::sequence::CTCLoss<double> loss;
double value = loss.forward(logits, /*timesteps*/ 20, /*classes*/ 8, labels, /*blank*/ 0);
auto gradient = loss.backward();`,
      params: [
        { name: "blank", detail: bi("Class index reserved for blank. Default 0.", "Boşluk için ayrılmış sınıf indeksi. Varsayılan 0.") },
        { name: "layout", detail: bi("Time-major. Length must be timesteps × classes.", "Zaman-major. Uzunluk zaman × sınıf olmalıdır.") },
      ],
    },
    {
      id: "mlm",
      name: "Masked language model",
      api: "sequence::MaskedSequenceCrossEntropyLoss<T>",
      formula: "mean CE over masked positions",
      kind: bi("Tokens · ignore the rest", "Token · gerisini yok say"),
      when: bi(
        "BERT-style pretraining. Most positions are context, and only the masked ones have a target. Pads must not contribute a gradient.",
        "BERT tarzı ön eğitim. Konumların çoğu bağlamdır, yalnızca maskelenenlerin hedefi vardır. Dolgular gradyana katkı vermemelidir.",
      ),
      logic: bi(
        "Token logits are packed token-major, token * classes + class. You pass the positions that were masked and the original token ids at those positions. The loss is the mean cross-entropy of that subset. Positions you do not list are structurally zero in the gradient, which is stronger than multiplying by a pad mask after the fact.",
        "Token logitleri token-major paketlenir: token * sınıf + sınıf. Maskelenen konumları ve o konumlardaki özgün token kimliklerini verirsin. Kayıp o alt kümenin ortalama çapraz entropisidir. Listelemediğin konumlar gradyanda yapısal olarak sıfırdır; bu, sonradan bir dolgu maskesiyle çarpmaktan daha güçlüdür.",
      ),
      gradient: bi(
        "softmax − one_hot at each masked token, divided by the number of masks. Everywhere else, 0.",
        "Her maskeli tokenda softmax − one_hot, maske sayısına bölünür. Başka her yerde 0.",
      ),
      code: `std::vector<size_t> positions{2, 5};
std::vector<size_t> targets{17, 4};
nexusloss::sequence::MaskedSequenceCrossEntropyLoss<double> loss;
double value = loss.forward(token_logits, /*tokens*/ 8, /*classes*/ 32, positions, targets);
auto gradient = loss.backward();`,
      params: [
        { name: "masked_targets", detail: bi("Class indices. Out of range throws.", "Sınıf indeksleri. Aralık dışı hata fırlatır.") },
      ],
    },
    {
      id: "nsp",
      name: "Next sentence",
      api: "sequence::next_sentence_prediction_loss",
      formula: "BCE-with-logits(logit, is_next)",
      kind: bi("Free function · one bit", "Serbest fonksiyon · tek bit"),
      when: bi(
        "The BERT auxiliary: did sentence B follow sentence A? One logit from the [CLS] head, one binary target.",
        "BERT yardımcısı: B cümlesi A'nın ardından mı geldi? [CLS] başından bir logit, bir ikili hedef.",
      ),
      logic: bi(
        "It is binary cross-entropy on a logit, isolated so a pretraining step can add it to the masked-token loss with a weight. is_next is 1 when the pair is contiguous and 0 when it was shuffled. There is no class because the state is two scalars.",
        "Bir logit üzerindeki ikili çapraz entropidir; bir ön eğitim adımının onu ağırlıkla maskeli token kaybına ekleyebilmesi için ayrılmıştır. Çift bitişikse is_next 1, karıştırıldıysa 0'dır. Durum iki skaler olduğu için sınıf yoktur.",
      ),
      gradient: bi(
        "∂L/∂logit = σ(logit) − is_next.",
        "∂L/∂logit = σ(logit) − is_next.",
      ),
      code: `double value = nexusloss::sequence::next_sentence_prediction_loss<double>(cls_logit, /*is next*/ 1.0);`,
      params: [
        { name: "is_next", detail: bi("0 or 1.", "0 veya 1.") },
      ],
    },
  ],
};

export const pointCloud: CategoryDoc = {
  id: "point-cloud",
  index: "11",
  title: bi("Point clouds", "Nokta bulutları"),
  lede: bi(
    "Unordered sets of coordinates, packed flat, default dimension 3. Chamfer is the practical training loss. Exact EMD solves a Hungarian assignment and is cubic, so keep it for small clouds.",
    "Düz paketlenmiş, varsayılan boyutu 3 olan sırasız koordinat kümeleri. Chamfer pratik eğitim kaybıdır. Kesin EMD bir Macar ataması çözer ve kübiktir; küçük bulutlara sakla.",
  ),
  losses: [
    {
      id: "chamfer",
      name: "Chamfer",
      api: "point_cloud::ChamferDistanceLoss<T>",
      formula: "mean_a min_b ‖a−b‖² + mean_b min_a ‖b−a‖²",
      kind: bi("Two clouds · nearest", "İki bulut · en yakın"),
      when: bi(
        "Shape completion and generation, when the clouds have different cardinalities and a point only needs a nearby partner, not a unique matching.",
        "Şekil tamamlama ve üretim: bulutların eleman sayıları farklıysa ve bir nokta benzersiz bir eşleşme değil, yakındaki bir ortak istiyorsa.",
      ),
      logic: bi(
        "Each point pays the squared distance to its nearest neighbor in the other cloud, and the two directions are averaged so a predicted cloud cannot win by collapsing onto a subset of the target. Nearest-neighbor ties take the first minimum. The search is exhaustive, which is fine for thousands of points and the wrong tool for millions.",
        "Her nokta diğer buluttaki en yakın komşusuna karesel mesafe öder; iki yön ortalanır, böylece tahmin edilen bir bulut hedefin bir alt kümesine çökerek kazanamaz. En yakın komşu beraberliklerinde ilk minimum alınır. Arama ayrıntılıdır; binlerce nokta için uygundur, milyonlar için yanlış araçtır.",
      ),
      gradient: bi(
        "backward() concatenates ∂L/∂cloud_a and ∂L/∂cloud_b. Mass flows only along the nearest-neighbor edges, with a factor 2(a − b).",
        "backward() ∂L/∂bulut_a ve ∂L/∂bulut_b'yi birleştirir. Kütle yalnızca en yakın komşu kenarları boyunca, 2(a − b) çarpanıyla akar.",
      ),
      code: `std::vector<double> cloud_a{0, 0, 0,  1, 0, 0};
std::vector<double> cloud_b{0.1, 0, 0,  1.2, 0.1, 0};
nexusloss::point_cloud::ChamferDistanceLoss<double> loss(3);
double value = loss.forward(cloud_a, cloud_b);
auto gradient = loss.backward();`,
      params: [
        { name: "dimensions", detail: bi("Default 3. Both lengths must be divisible by it.", "Varsayılan 3. İki uzunluk da buna bölünmelidir.") },
      ],
    },
    {
      id: "emd",
      name: "Earth mover",
      api: "point_cloud::EarthMoverDistanceLoss<T>",
      formula: "mean ‖a_i − b_{π(i)}‖²   over the optimal bijection π",
      kind: bi("Equal size · Hungarian", "Eşit boy · Macar"),
      when: bi(
        "When Chamfer's many-to-one matches are a bug: every predicted point should claim a distinct target point. Small clouds only. The assignment is O(n³).",
        "Chamfer'in çoktan-bire eşleşmeleri bir hataysa: her tahmin noktası ayrı bir hedef nokta sahiplenmelidir. Yalnızca küçük bulutlar. Atama O(n³)'tür.",
      ),
      logic: bi(
        "Exact EMD here is the mean squared distance under the minimum-cost perfect matching, computed with the Hungarian algorithm (Jonker-Volgenant style potentials in the implementation). The two clouds must contain the same number of points. Unlike Chamfer, a point cannot be the neighbor of everyone. For large n, use an approximation outside this library and come back if you only need the loss of a matching you already have.",
        "Buradaki kesin EMD, minimum maliyetli mükemmel eşleşme altındaki ortalama karesel mesafedir; Macar algoritmasıyla hesaplanır. İki bulut aynı sayıda nokta içermelidir. Chamfer'in aksine bir nokta herkesin komşusu olamaz. Büyük n için bu kütüphanenin dışında bir yaklaşım kullan; elinde bir eşleşme varken yalnızca onun kaybını istiyorsan geri gel.",
      ),
      gradient: bi(
        "Once π is chosen it is treated as constant. backward() is 2(a_i − b_π(i)) / n on each matched pair, concatenated over both clouds.",
        "π seçildikten sonra sabit sayılır. backward() her eşleşmiş çiftte 2(a_i − b_π(i)) / n'dir, iki bulut üzerinde birleştirilir.",
      ),
      code: `nexusloss::point_cloud::EarthMoverDistanceLoss<double> loss(3);
double value = loss.forward(cloud_a, cloud_b);
auto gradient = loss.backward();`,
      params: [
        { name: "cardinality", detail: bi("The two clouds must have equal point counts.", "İki bulutun nokta sayıları eşit olmalıdır.") },
      ],
    },
  ],
};

export const reinforcement: CategoryDoc = {
  id: "reinforcement",
  index: "12",
  title: bi("Reinforcement", "Pekiştirmeli"),
  lede: bi(
    "Policy losses differentiate log-probabilities. The advantage is a constant you computed from returns. Value loss is MSE on the critic. Entropy is returned as a loss to minimize, already negated, so adding it increases randomness.",
    "Politika kayıpları log-olasılıkları türevler. Avantaj, getirilerden hesapladığın bir sabittir. Değer kaybı eleştirmen üzerinde MSE'dir. Entropi, minimize edilecek bir kayıp olarak, çoktan negatiflenmiş döner; onu eklemek rastgeleliği artırır.",
  ),
  losses: [
    {
      id: "policy-gradient",
      name: "Policy gradient",
      api: "reinforcement::PolicyGradientLoss<T>",
      formula: "L = −mean(log π · A)",
      kind: bi("Log-probabilities · REINFORCE", "Log-olasılıklar · REINFORCE"),
      when: bi(
        "The on-policy baseline. You sampled actions, stored their log-probabilities, and estimated an advantage. Minimize this and the policy climbs the advantage.",
        "Politika-üstü taban çizgisi. Eylemleri örnekledin, log-olasılıklarını sakladın ve bir avantaj tahmin ettin. Bunu minimize et, politika avantajı tırmanır.",
      ),
      logic: bi(
        "Williams' REINFORCE score function. The advantage is treated as constant, which is why a baseline that does not depend on the action can be subtracted without changing the expected gradient. The leading minus sign means gradient descent increases the probability of actions with positive advantage. NexusLoss does not subtract a baseline for you.",
        "Williams'ın REINFORCE skor fonksiyonu. Avantaj sabit sayılır; eyleme bağlı olmayan bir taban çizgisinin, beklenen gradyanı değiştirmeden çıkarılabilecek olmasının nedeni budur. Baştaki eksi işaret, gradyan inişinin pozitif avantajlı eylemlerin olasılığını artırdığı anlamına gelir. NexusLoss senin yerine bir taban çizgisi çıkarmaz.",
      ),
      gradient: bi(
        "∂L/∂logπ_i = −A_i / N. The advantage tensor is not differentiated.",
        "∂L/∂logπ_i = −A_i / N. Avantaj tensörü türetilmez.",
      ),
      code: `std::vector<double> log_pi{-0.2, -1.4};
std::vector<double> advantage{1.0, -0.5};
nexusloss::reinforcement::PolicyGradientLoss<double> loss;
double value = loss.forward(log_pi, advantage);
auto gradient = loss.backward();`,
      params: [
        { name: "advantage", detail: bi("Same length as log π. Detach it.", "log π ile aynı uzunluk. Ondan kopar.") },
      ],
    },
    {
      id: "ppo",
      name: "PPO clipped surrogate",
      api: "reinforcement::PPOClipLoss<T>",
      formula: "−mean( min(r A, clip(r, 1−ε, 1+ε) A) )",
      kind: bi("New and old log π", "Yeni ve eski log π"),
      when: bi(
        "The default policy loss once you have a trust region to protect. ε = 0.2 is the Schulman default. The old log-probabilities are from the policy that collected the batch.",
        "Koruyacak bir güven bölgen olduktan sonra varsayılan politika kaybı. ε = 0.2 Schulman varsayılanıdır. Eski log-olasılıklar, batch'i toplayan politikadandır.",
      ),
      logic: bi(
        "r = exp(new − old) is the probability ratio. The clip stops the update when a positive advantage would push the ratio above 1+ε, or a negative advantage would push it below 1−ε. The loss takes the pessimistic of the clipped and unclipped objectives, then negates it so you minimize. Multiple epochs on the same batch stay close to the behavior policy.",
        "r = exp(yeni − eski), olasılık oranıdır. Kırpma, pozitif bir avantaj oranı 1+ε üstüne itecekse ya da negatif bir avantaj 1−ε altına itecekse güncellemeyi durdurur. Kayıp, kırpılmış ve kırpılmamış amaçların kötümser olanını alır, sonra minimize edesin diye negatifler. Aynı batch üzerinde birden fazla çağ, davranış politikasına yakın kalır.",
      ),
      gradient: bi(
        "With respect to the new log-probability only. When the clip is active and it is the one selected by the min, the gradient is zero. Old log-probabilities and advantages are constants.",
        "Yalnızca yeni log-olasılığa göredir. Kırpma aktifse ve min onu seçtiyse gradyan sıfırdır. Eski log-olasılıklar ve avantajlar sabittir.",
      ),
      code: `nexusloss::reinforcement::PPOClipLoss<double> loss(0.2);
double value = loss.forward(new_log_pi, old_log_pi, advantage);
auto gradient = loss.backward();`,
      params: [
        { name: "clip", detail: bi("ε ≥ 0. Default 0.2.", "ε ≥ 0. Varsayılan 0.2.") },
      ],
    },
    {
      id: "value",
      name: "Value",
      api: "reinforcement::ValueLoss<T>",
      formula: "mean (V − R)²",
      kind: bi("Critic · returns", "Eleştirmen · getiriler"),
      when: bi(
        "The critic. Predicted state values against returns or against a value target you already bootstrapped. This is MSE with the RL names on the arguments.",
        "Eleştirmen. Tahmin edilen durum değerleri, getirilere veya çoktan önyüklediğin bir değer hedefine karşı. Argümanlarında RL adları olan MSE'dir.",
      ),
      logic: bi(
        "A squared error keeps the value head smooth and makes large TD errors count more, which is usually what you want for a baseline. If returns are noisy, consider Huber in the regression namespace instead and accept that this class is the plain one PPO papers write down.",
        "Karesel hata değer başını pürüzsüz tutar ve büyük TD hatalarının daha çok sayılmasını sağlar; bir taban çizgisi için genellikle istenen budur. Getiriler gürültülüyse regresyon ad alanındaki Huber'i düşün ve bu sınıfın PPO makalelerinin yazdığı sade hali olduğunu kabul et.",
      ),
      gradient: bi(
        "∂L/∂V = 2 (V − R) / N. Returns are constant.",
        "∂L/∂V = 2 (V − R) / N. Getiriler sabittir.",
      ),
      code: `std::vector<double> values{0.4, 1.1};
std::vector<double> returns{0.5, 0.2};
nexusloss::reinforcement::ValueLoss<double> loss;
double value = loss.forward(values, returns);`,
      params: [
        { name: "returns", detail: bi("Same length as values. Already discounted.", "Değerlerle aynı uzunluk. İndirimi yapılmış.") },
      ],
    },
    {
      id: "entropy",
      name: "Entropy bonus",
      api: "reinforcement::EntropyBonusLoss<T>",
      formula: "L = −H(p) = Σ p log p    (mean over the batch layout you pass)",
      kind: bi("Probabilities · exploration", "Olasılıklar · keşif"),
      when: bi(
        "Add it, with a small coefficient, to a policy loss so the action distribution does not collapse to a single atom before the advantage has anything to say.",
        "Küçük bir katsayıyla bir politika kaybına ekle; avantajın söyleyecek bir şeyi olmadan eylem dağılımı tek bir atoma çökmesin.",
      ),
      logic: bi(
        "entropy_bonus returns the Shannon entropy, which you would maximize. EntropyBonusLoss returns its negation, so minimizing the loss maximizes entropy. The input is probabilities, not logits, and they should form a distribution per action (or you pass a flat vector you already decided is one distribution). A uniform categorical has the highest entropy and the lowest loss.",
        "entropy_bonus Shannon entropisini döndürür; onu maksimize ederdin. EntropyBonusLoss onun negatifini döndürür, yani kaybı minimize etmek entropiyi maksimize eder. Girdi logit değil olasılıktır; eylem başına bir dağılım oluşturmalıdır (ya da tek dağılım olduğuna karar verdiğin düz bir vektör verirsin). Düzgün bir kategorik, en yüksek entropiye ve en düşük kayba sahiptir.",
      ),
      gradient: bi(
        "∂L/∂p_i = log(p_i) + 1, up to the averaging convention of the implementation. Stay away from exact zeros; a probability of 0 makes log undefined and the code guards the empty and non-distribution cases it can see.",
        "∂L/∂p_i = log(p_i) + 1, uygulamanın ortalama sözleşmesine kadar. Tam sıfırlardan uzak dur; 0 olasılık log'u tanımsız kılar ve kod görebildiği boş ve dağılım-olmayan durumları korur.",
      ),
      code: `std::vector<double> probabilities{0.7, 0.2, 0.1};
nexusloss::reinforcement::EntropyBonusLoss<double> loss;
double value = loss.forward(probabilities); // negative entropy
auto gradient = loss.backward();`,
      params: [
        { name: "probabilities", detail: bi("Positive masses. Not logits.", "Pozitif kütleler. Logit değil.") },
      ],
    },
  ],
};

export const survival: CategoryDoc = {
  id: "survival",
  index: "13",
  title: bi("Survival", "Sağkalım"),
  lede: bi(
    "Censored time-to-event data. Cox compares relative risk inside a batch. Weibull is a parametric density for one subject, with the scale and shape passed as logs so they stay positive.",
    "Sansürlü olay-süresi verisi. Cox, bir batch içindeki göreli riski karşılaştırır. Weibull, bir denek için parametrik bir yoğunluktur; ölçek ve şekil pozitif kalsın diye log olarak verilir.",
  ),
  losses: [
    {
      id: "cox",
      name: "Cox partial likelihood",
      api: "survival::CoxPHLoss<T>",
      formula: "− Σ_{events} ( η_i − log Σ_{j: t_j ≥ t_i} exp(η_j) )",
      kind: bi("Batch · Breslow risk set", "Batch · Breslow risk kümesi"),
      when: bi(
        "You have durations, a binary event indicator, and a model that emits a log-risk. The baseline hazard is left unspecified, which is the point of Cox.",
        "Sürelerin, ikili bir olay göstergesinin ve log-risk üreten bir modelin var. Taban hazard belirtilmez; Cox'un amacı budur.",
      ),
      logic: bi(
        "The partial likelihood only asks who failed first among the people still at risk. Breslow's risk set is everyone with duration at least the event time, which ties events together instead of breaking them arbitrarily. Censored rows (observed = false) do not add a numerator term, but they stay in the risk set of earlier events and still receive a gradient. Sort order does not matter; the comparisons are by duration.",
        "Kısmi olabilirlik yalnızca hâlâ risk altında olanlar arasında kimin önce başarısız olduğunu sorar. Breslow risk kümesi, süresi en az olay süresi kadar olan herkestir; olayları keyfi kırmak yerine birbirine bağlar. Sansürlü satırlar (observed = false) bir pay terimi eklemez ama daha erken olayların risk kümesinde kalır ve yine bir gradyan alır. Sıralama önemli değildir; karşılaştırmalar süreye göredir.",
      ),
      gradient: bi(
        "For an event i, +1 on η_i and −softmax(η) over the risk set. Censored points only appear in those soft weights. backward() matches forward().",
        "Bir i olayı için η_i üzerinde +1 ve risk kümesi üzerinde −softmax(η). Sansürlü noktalar yalnızca o yumuşak ağırlıklarda görünür. backward(), forward() ile uyumludur.",
      ),
      code: `std::vector<double> log_risk{0.2, 1.1, -0.4};
std::vector<double> duration{5.0, 2.0, 9.0};
std::vector<bool> observed{true, true, false};
nexusloss::survival::CoxPHLoss<double> loss;
double value = loss.forward(log_risk, duration, observed);
auto gradient = loss.backward();`,
      params: [
        { name: "observed", detail: bi("true if the event happened, false if censored.", "Olay olduysa true, sansürlendiyse false.") },
      ],
    },
    {
      id: "weibull",
      name: "Weibull NLL",
      api: "survival::WeibullNLLLoss<T>",
      formula: "event: −log f(t)    censored: −log S(t)",
      kind: bi("One subject · log scale, log shape", "Tek denek · log ölçek, log şekil"),
      when: bi(
        "A parametric survival head when you are willing to assume a Weibull hazard and you want a likelihood for a single (time, event) pair.",
        "Weibull hazard varsaymaya razıysan ve tek bir (süre, olay) çifti için olabilirlik istediğinde parametrik bir sağkalım başı.",
      ),
      logic: bi(
        "The density term is used when the event was observed. The survival term S(t) = P(T > t) is used when the subject was censored: you know they lasted at least this long and nothing more. Scale and shape are passed as logs so the true parameters exp(log_scale) and exp(log_shape) cannot go non-positive. Time must be positive.",
        "Olay gözlendiyse yoğunluk terimi kullanılır. Denek sansürlendiyse sağkalım terimi S(t) = P(T > t) kullanılır: en az bu kadar dayandığını bilirsin, fazlasını değil. Ölçek ve şekil log olarak verilir; gerçek parametreler exp(log_scale) ve exp(log_shape) pozitif olmaktan çıkamaz. Süre pozitif olmalıdır.",
      ),
      gradient: bi(
        "backward() returns two numbers, ∂L/∂log_scale and ∂L/∂log_shape. An event and a censored observation do not share the same expression.",
        "backward() iki sayı döndürür: ∂L/∂log_scale ve ∂L/∂log_shape. Bir olay ile sansürlü bir gözlem aynı ifadeyi paylaşmaz.",
      ),
      code: `nexusloss::survival::WeibullNLLLoss<double> loss;
double value = loss.forward(/*time*/ 3.5, /*event*/ true, /*log scale*/ 0.0, /*log shape*/ 0.2);
auto gradient = loss.backward();`,
      params: [
        { name: "time", detail: bi("Strictly positive duration.", "Kesin pozitif süre.") },
        { name: "event", detail: bi("true uses the density, false uses the survival function.", "true yoğunluğu, false sağkalım fonksiyonunu kullanır.") },
      ],
    },
  ],
};
