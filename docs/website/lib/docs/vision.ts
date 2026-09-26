import { bi, type CategoryDoc } from "./types";

export const segmentation: CategoryDoc = {
  id: "segmentation",
  index: "03",
  title: bi("Segmentation", "Segmentasyon"),
  lede: bi(
    "Soft masks of equal length. These are probabilities, not logits, except Lovász-hinge (logits) and Lovász-softmax (per-pixel class probabilities plus a class index). A smooth constant keeps an empty mask finite. Overlap losses return a scalar from forward() and a full-mask gradient from backward().",
    "Eşit uzunlukta yumuşak maskeler. Lovász-hinge (logit) ve Lovász-softmax (piksel başına sınıf olasılığı artı sınıf indeksi) dışında bunlar olasılıktır, logit değil. Smooth sabiti boş maskeyi sonlu tutar. Örtüşme kayıpları forward()'dan skaler, backward()'dan tam maske gradyanı döndürür.",
  ),
  losses: [
    {
      id: "dice",
      name: "Dice",
      api: "segmentation::DiceLoss<T>",
      formula: "L = 1 − (2 Σ p y + s) / (Σ p² + Σ y² + s)",
      kind: bi("Soft mask · overlap", "Yumuşak maske · örtüşme"),
      when: bi(
        "The standard overlap loss for medical and binary segmentation, especially when the foreground is a small fraction of the image and pixel-wise BCE would be satisfied by predicting all background.",
        "Ön plan görüntünün küçük bir kısmıyken ve piksel BCE'si her yeri arka plan diyerek tatmin olacaksa, tıbbi ve ikili segmentasyon için standart örtüşme kaybı.",
      ),
      logic: bi(
        "Dice is the soft F1 score. Numerator and denominator both see the foreground, so a sea of true negatives does not inflate the score the way accuracy does. The implementation uses Σ p² + Σ y² in the denominator (the squared form), which is differentiable and matches the common soft-Dice used in training. s, default 1, stops a zero-over-zero when both masks are empty.",
        "Dice, yumuşak F1 skorudur. Pay ve payda ön planı görür; doğrulukta olduğu gibi doğru negatif denizi skoru şişirmez. Uygulama paydada Σ p² + Σ y² kullanır (kareli biçim); türevlenebilir ve eğitimde yaygın yumuşak Dice ile uyumludur. Varsayılan s = 1, iki maske de boşken sıfır bölü sıfırı durdurur.",
      ),
      gradient: bi(
        "The quotient rule on the soft coefficient. Every pixel receives a gradient, including background pixels that should have stayed zero. Mean-style 1/N scaling is already inside the scalar loss, not a second LossBase reduction.",
        "Yumuşak katsayı üzerinde bölüm kuralı. Her piksel, sıfır kalması gereken arka plan pikselleri dahil, bir gradyan alır. 1/N ölçeği skaler kaybın içindedir; ikinci bir LossBase indirgemesi yoktur.",
      ),
      code: `std::vector<double> mask{0.9, 0.1, 0.8};
std::vector<double> truth{1.0, 0.0, 1.0};
nexusloss::segmentation::DiceLoss<double> loss(1.0); // smooth
double value = loss.forward(mask, truth);
auto gradient = loss.backward();`,
      params: [
        { name: "smooth", detail: bi("Default 1. Added to numerator and denominator.", "Varsayılan 1. Pay ve paydaya eklenir.") },
      ],
    },
    {
      id: "seg-iou",
      name: "Soft IoU / Jaccard",
      api: "segmentation::IoULoss<T>",
      formula: "L = 1 − (Σ p y + s) / (Σ p + Σ y − Σ p y + s)",
      kind: bi("Soft mask · union", "Yumuşak maske · birleşim"),
      when: bi(
        "When the metric you report is IoU and you want the training loss to be that metric. Dice and IoU rank models similarly; IoU punishes false positives and false negatives against the union, which is a harsher denominator.",
        "Raporladığın metrik IoU ise ve eğitim kaybının o metrik olmasını istiyorsan. Dice ve IoU modelleri benzer sıralar; IoU yanlış pozitif ve yanlış negatifleri birleşime karşı cezalandırır, payda daha serttir.",
      ),
      logic: bi(
        "Jaccard is intersection over union. In the soft version the intersection is the dot product of the two masks and the union is the sum of each minus that intersection, so a pixel is not counted twice. This is not the box IoU in the detection namespace; the tensors here are flattened masks.",
        "Jaccard, kesişim bölü birleşimdir. Yumuşak versiyonda kesişim iki maskenin iç çarpımı, birleşim her birinin toplamı eksi o kesişimdir; bir piksel iki kez sayılmaz. Bu, detection ad alanındaki kutu IoU'su değildir; buradaki tensörler düzleştirilmiş maskelerdir.",
      ),
      gradient: bi(
        "Quotient rule again. A false positive increases the union without increasing the intersection, so its gradient pushes the probability down.",
        "Yine bölüm kuralı. Yanlış pozitif, kesişimi artırmadan birleşimi büyütür; gradyanı olasılığı aşağı iter.",
      ),
      code: `nexusloss::segmentation::IoULoss<double> loss(1.0);
double value = loss.forward(mask, truth);
auto gradient = loss.backward();`,
      params: [
        { name: "smooth", detail: bi("Default 1.", "Varsayılan 1.") },
      ],
    },
    {
      id: "tversky",
      name: "Tversky",
      api: "segmentation::TverskyLoss<T>",
      formula: "L = 1 − (TP + s) / (TP + α FP + β FN + s)",
      kind: bi("Soft mask · asymmetric", "Yumuşak maske · asimetrik"),
      when: bi(
        "When missing foreground is worse than inventing it, or the reverse. Tumor outlines often set β > α so false negatives cost more.",
        "Ön planı kaçırmak uydurmaktan kötüyse ya da tersi. Tümör sınırları çoğu zaman β > α koyar; yanlış negatif daha pahalıdır.",
      ),
      logic: bi(
        "Tversky generalizes Dice. α weights false positives, β weights false negatives. α = β = 0.5 is Dice in the TP/FP/FN parameterization (the 2 in Dice's numerator is the same statement). Turning the knobs does not change the interface: two soft masks in, one scalar out.",
        "Tversky, Dice'i geneller. α yanlış pozitifleri, β yanlış negatifleri ağırlar. α = β = 0.5, TP/FP/FN parametrelemesinde Dice'dir (Dice payındaki 2 aynı ifadedir). Düğmeleri çevirmek arayüzü değiştirmez: iki yumuşak maske girer, bir skaler çıkar.",
      ),
      gradient: bi(
        "∂L flows through TP, FP and FN with the α and β multipliers still attached, so the two error types are not symmetric unless you set them equal.",
        "∂L, α ve β çarpanları üzerinde kalarak TP, FP ve FN'den akar; iki hata tipi eşit ayarlamadıkça simetrik değildir.",
      ),
      code: `nexusloss::segmentation::TverskyLoss<double> loss(0.3, 0.7, 1.0); // alpha, beta, smooth
double value = loss.forward(mask, truth);
auto gradient = loss.backward();`,
      params: [
        { name: "alpha", detail: bi("False-positive weight. Default 0.5.", "Yanlış pozitif ağırlığı. Varsayılan 0.5.") },
        { name: "beta", detail: bi("False-negative weight. Default 0.5.", "Yanlış negatif ağırlığı. Varsayılan 0.5.") },
        { name: "smooth", detail: bi("Default 1.", "Varsayılan 1.") },
      ],
    },
    {
      id: "focal-tversky",
      name: "Focal Tversky",
      api: "segmentation::FocalTverskyLoss<T>",
      formula: "L = (Tversky)^γ",
      kind: bi("Soft mask · hard regions", "Yumuşak maske · zor bölgeler"),
      when: bi(
        "Small structures that Dice already sees but still under-trains, because the overlap looks 'good enough' on the easy slices. γ > 1 stretches the loss when Tversky is not yet small.",
        "Dice'in gördüğü ama hâlâ az eğittiği küçük yapılar; kolay kesitlerde örtüşme 'yeterince iyi' göründüğü için. γ > 1, Tversky henüz küçük değilken kaybı gerer.",
      ),
      logic: bi(
        "Focal Tversky raises the Tversky loss to γ. A mask that is almost right (Tversky loss near 0) is suppressed further. A mask that is still wrong stays loud. The default γ is 4/3, the value used in the original focal-Tversky paper's practical setting. The gradient is γ · Tversky^(γ−1) times the Tversky gradient, so you do not hand-write a second quotient rule.",
        "Focal Tversky, Tversky kaybını γ kuvvetine yükseltir. Neredeyse doğru bir maske (Tversky kaybı 0'a yakın) daha da bastırılır. Hâlâ yanlış olan maske yüksek sesli kalır. Varsayılan γ, 4/3'tür. Gradyan, γ · Tversky^(γ−1) çarpı Tversky gradyanıdır; ikinci bir bölüm kuralını elle yazmazsın.",
      ),
      gradient: bi(
        "Chain rule through the power. If the inner Tversky loss is 0 the focal factor vanishes with it.",
        "Kuvvet üzerinden zincir kuralı. İç Tversky kaybı 0 ise focal çarpan da onunla yok olur.",
      ),
      code: `nexusloss::segmentation::FocalTverskyLoss<double> loss(0.3, 0.7, 4.0 / 3.0);
double value = loss.forward(mask, truth);`,
      params: [
        { name: "gamma", detail: bi("Default 4/3.", "Varsayılan 4/3.") },
      ],
    },
    {
      id: "combo",
      name: "Combo",
      api: "segmentation::combo_loss",
      formula: "w · Dice + (1 − w) · BCE",
      kind: bi("Free function · hybrid", "Serbest fonksiyon · melez"),
      when: bi(
        "When pure overlap ignores pixel-level calibration and pure BCE ignores region shape. The usual first mixture for a segmentation head that already outputs probabilities.",
        "Saf örtüşme piksel kalibrasyonunu, saf BCE bölge şeklini yok saydığında. Zaten olasılık üreten bir segmentasyon başı için alışılmış ilk karışım.",
      ),
      logic: bi(
        "Combo adds a weighted Dice loss to a mean binary cross-entropy on the same soft mask. w = 0.5 gives them equal voice. The BCE term supplies a per-pixel gradient everywhere; Dice supplies a global coupling so the foreground mass has to match. This free function returns the scalar. Pair it with DiceLoss::backward and a BCELoss if you need the two gradients separately.",
        "Combo, aynı yumuşak maske üzerinde ağırlıklı Dice kaybına ortalama ikili çapraz entropi ekler. w = 0.5 onlara eşit söz hakkı verir. BCE terimi her yerde piksel gradyanı sağlar; Dice, ön plan kütlesinin tutması için küresel bir bağ kurar. Bu serbest fonksiyon skaler döner. İki gradyanı ayrı istiyorsan DiceLoss::backward ile bir BCELoss'u yan yana koy.",
      ),
      gradient: bi(
        "Not cached on the free function. The mixture's gradient is w times the Dice gradient plus (1 − w) times the BCE gradient.",
        "Serbest fonksiyonda önbelleklenmez. Karışımın gradyanı, w katı Dice gradyanı artı (1 − w) katı BCE gradyanıdır.",
      ),
      code: `double value = nexusloss::segmentation::combo_loss<double>(mask, truth, /*dice weight*/ 0.5, 1.0);`,
      params: [
        { name: "dice_weight", detail: bi("Default 0.5. The BCE weight is the complement.", "Varsayılan 0.5. BCE ağırlığı tümleyendir.") },
      ],
    },
    {
      id: "boundary",
      name: "Boundary",
      api: "segmentation::boundary_loss",
      formula: "L = mean(p · distance)",
      kind: bi("Free function · contour", "Serbest fonksiyon · kontur"),
      when: bi(
        "When the region overlap is already fine and the remaining error lives on the contour. You bring the signed distance map; NexusLoss does not compute a distance transform.",
        "Bölge örtüşmesi zaten iyiyse ve kalan hata konturdaysa. İşaretli mesafe haritasını sen getirirsin; NexusLoss mesafe dönüşümü hesaplamaz.",
      ),
      logic: bi(
        "Kervadec et al. integrate the probability mass against a signed distance to the ground-truth boundary. Pixels deep inside or far outside, depending on the sign convention of the map you pass, cost more than pixels sitting on the contour. Because the distance map is treated as constant, the gradient with respect to p is just that distance (then averaged).",
        "Kervadec ve arkadaşları olasılık kütlesini, gerçek sınıra işaretli mesafeyle integral eder. Haritanın işaret sözleşmesine göre içeride veya dışarıda derin pikseller, konturda oturanlardan pahalıdır. Mesafe haritası sabit sayıldığı için p'ye göre gradyan yalnızca o mesafedir (sonra ortalanır).",
      ),
      gradient: bi(
        "∂L/∂p_i ∝ signed_distance_i. Build the map outside the library and keep it detached from the graph.",
        "∂L/∂p_i ∝ işaretli_mesafe_i. Haritayı kütüphanenin dışında kur ve graftan kopuk tut.",
      ),
      code: `std::vector<double> signed_distance{-2.0, 0.1, 3.0};
double value = nexusloss::segmentation::boundary_loss<double>(mask, signed_distance);`,
      params: [
        { name: "signed_distance", detail: bi("Same length as the probability mask. Not computed here.", "Olasılık maskesiyle aynı uzunluk. Burada hesaplanmaz.") },
      ],
    },
    {
      id: "hausdorff",
      name: "Hausdorff approximation",
      api: "segmentation::hausdorff_distance_loss",
      formula: "mean( (p − y)² · distance² )",
      kind: bi("Free function · boundary distance", "Serbest fonksiyon · sınır mesafesi"),
      when: bi(
        "When a few distant false blobs should hurt more than a one-pixel wobble, and you already have a distance-to-boundary map for the target.",
        "Birkaç uzak sahte leke, bir piksellik oynamadan daha çok acıtmalıysa ve hedefin sınıra mesafe haritası elindeyse.",
      ),
      logic: bi(
        "The exact Hausdorff distance is a max over the contour and is awkward to differentiate. The approximation weights the squared mask error by the squared distance to the boundary, so a mistake far from the true edge dominates a mistake on the edge. The distance map is an input, same as boundary loss.",
        "Kesin Hausdorff mesafesi kontur üzerinde bir maksimumdur ve türevlemek zordur. Yaklaşım, karesel maske hatasını sınıra olan mesafenin karesiyle ağırlar; gerçek kenardan uzak bir hata, kenardaki bir hataya hükmeder. Mesafe haritası, sınır kaybındaki gibi bir girdidir.",
      ),
      gradient: bi(
        "Scalar helper. The dependence on p is through (p − y)², so the gradient carries 2(p − y) distance².",
        "Skaler yardımcı. p'ye bağımlılık (p − y)² üzerindendir; gradyan 2(p − y) mesafe² taşır.",
      ),
      code: `std::vector<double> distance_to_boundary{0.0, 4.0, 1.0};
double value = nexusloss::segmentation::hausdorff_distance_loss<double>(
    mask, truth, distance_to_boundary);`,
      params: [
        { name: "distance_to_boundary", detail: bi("Non-negative map, same length as the masks.", "Negatif olmayan harita, maskelerle aynı uzunlukta.") },
      ],
    },
    {
      id: "lovasz-hinge",
      name: "Lovász hinge",
      api: "segmentation::lovasz_hinge_loss",
      formula: "Jaccard surrogate on sorted hinge errors",
      kind: bi("Free function · logits", "Serbest fonksiyon · logit"),
      when: bi(
        "Binary segmentation from logits when you want a convex surrogate of the IoU, not a soft overlap of probabilities. One mask, labels in {0, 1} or equivalently mapped to ±1 inside the function.",
        "Olasılıkların yumuşak örtüşmesi değil, IoU'nun dışbükey bir vekilini istediğinde logitlerden ikili segmentasyon. Tek maske; etiketler {0, 1} ya da fonksiyon içinde ±1'e eşlenir.",
      ),
      logic: bi(
        "The Lovász extension turns the set function 'intersection over union' into a tight convex surrogate. Errors are sorted from worst to best and the loss is a weighted sum of those ordered hinge residuals, with weights equal to how much the discrete Jaccard would change if that pixel were added. Sorting is what makes a single bad pixel able to dominate, which is the Hausdorff-like behavior people actually want from IoU.",
        "Lovász genişlemesi, 'kesişim bölü birleşim' küme fonksiyonunu sıkı bir dışbükey vekile çevirir. Hatalar kötüden iyiye sıralanır ve kayıp, o sıralı hinge artıklarının ağırlıklı toplamıdır; ağırlıklar, o piksel eklenirse ayrık Jaccard'ın ne kadar değişeceğidir. Sıralama, tek bir kötü pikselin hükmedebilmesini sağlar; IoU'dan gerçekten istenen Hausdorff benzeri davranış budur.",
      ),
      gradient: bi(
        "The free function returns the scalar. The subgradient follows the sorted permutation and is constant between ties. Use it as a metric-aligned objective beside a differentiable class if you wire the backward yourself.",
        "Serbest fonksiyon skaler döner. Alt-gradyan sıralı permütasyonu izler ve beraberlikler arasında sabittir. Backward'ı kendin bağlarsan, türevlenebilir bir sınıfın yanında metriğe hizalı bir amaç olarak kullan.",
      ),
      code: `std::vector<double> logits{1.2, -0.4, 0.3};
std::vector<double> binary{1.0, 0.0, 1.0};
double value = nexusloss::segmentation::lovasz_hinge_loss<double>(logits, binary);`,
      params: [
        { name: "logits", detail: bi("Raw scores, not probabilities.", "Ham skorlar, olasılık değil.") },
      ],
    },
    {
      id: "lovasz-softmax",
      name: "Lovász softmax",
      api: "segmentation::LovaszSoftmaxLoss<T>",
      formula: "mean over classes of the Lovász Jaccard surrogate",
      kind: bi("Class map · probabilities", "Sınıf haritası · olasılıklar"),
      when: bi(
        "Multi-class semantic segmentation when the number you log at validation is mean IoU and pixel cross-entropy is optimizing a different ranking.",
        "Doğrulamada logladığın sayı ortalama IoU ise ve piksel çapraz entropisi başka bir sıralamayı optimize ediyorsa, çok sınıflı anlamsal segmentasyon.",
      ),
      logic: bi(
        "Each class is treated as a binary foreground against the softmax probability of that class. Errors are sorted, the Jaccard surrogate is applied, and the class losses are averaged. The probability tensor is pixel-major with a fixed class count; targets are class indices, not one-hot masks. forward() caches both so backward() can replay the same permutation.",
        "Her sınıf, o sınıfın softmax olasılığına karşı ikili bir ön plan sayılır. Hatalar sıralanır, Jaccard vekili uygulanır ve sınıf kayıpları ortalanır. Olasılık tensörü sabit sınıf sayısıyla piksel-majordür; hedefler tek-sıcak maske değil sınıf indeksleridir. forward() ikisini de saklar, backward() aynı permütasyonu yeniden oynatabilir.",
      ),
      gradient: bi(
        "Implemented. The subgradient is scattered back onto the probability of each class at each pixel. An out-of-range class index throws.",
        "Uygulanmıştır. Alt-gradyan her pikselde her sınıfın olasılığına geri saçılır. Aralık dışı sınıf indeksi hata fırlatır.",
      ),
      code: `std::vector<double> probabilities{0.7, 0.2, 0.1,  0.1, 0.8, 0.1}; // 2 pixels, 3 classes
std::vector<size_t> classes{0, 1};
nexusloss::segmentation::LovaszSoftmaxLoss<double> loss;
double value = loss.forward(probabilities, classes, /*class count*/ 3);
auto gradient = loss.backward();`,
      params: [
        { name: "class_count", detail: bi("probabilities.size() must be pixels × classes.", "probabilities.size(), piksel × sınıf olmalıdır.") },
      ],
    },
  ],
};

export const detection: CategoryDoc = {
  id: "detection",
  index: "04",
  title: bi("Detection", "Nesne tespiti"),
  lede: bi(
    "Boxes are [x_min, y_min, x_max, y_max]. IoU family losses compare one predicted box with one target box and return a scalar; backward() is four partial derivatives. Smooth-L1 bbox and detection focal cover the other two heads of a detector.",
    "Kutular [x_min, y_min, x_max, y_max] biçimindedir. IoU ailesi bir tahmin kutusunu bir hedef kutuyla karşılaştırır ve skaler döner; backward() dört kısmi türevdir. Smooth-L1 kutu ve tespit focal'i, bir detektörün diğer iki başını kapsar.",
  ),
  losses: [
    {
      id: "det-iou",
      name: "IoU",
      api: "detection::IoULoss<T>",
      formula: "L = 1 − |A ∩ B| / |A ∪ B|",
      kind: bi("One box · 4 coordinates", "Tek kutu · 4 koordinat"),
      when: bi(
        "A pair of boxes that already overlap, as a clean overlap penalty. If they might miss each other completely, IoU saturates at 1 and the gradient dies; move to GIoU.",
        "Zaten örtüşen bir kutu çifti için temiz bir örtüşme cezası. Birbirlerini tamamen kaçırabilirlerse IoU 1'de doyar ve gradyan ölür; GIoU'ya geç.",
      ),
      logic: bi(
        "Intersection over union is scale-free: a 2× box error and a 20× box error with the same relative overlap look the same. The loss is 1 minus that ratio so that perfect overlap is 0. When the intersection is empty the loss is 1 for every disjoint pair, which is a plateau, not a direction.",
        "Kesişim bölü birleşim ölçekten bağımsızdır: aynı göreli örtüşmeye sahip 2 kat ve 20 kat kutu hatası aynı görünür. Mükemmel örtüşme 0 olsun diye kayıp 1 eksi o orandır. Kesişim boşsa her ayrık çift için kayıp 1'dir; bu bir yön değil, bir platodur.",
      ),
      gradient: bi(
        "Analytical in the four edges. The partials come from how each edge moves the intersection rectangle and the predicted area. Disjoint boxes can produce a zero overlap gradient.",
        "Dört kenarda analitiktir. Kısmi türevler, her kenarın kesişim dikdörtgenini ve tahmin alanını nasıl oynattığından gelir. Ayrık kutular sıfır örtüşme gradyanı üretebilir.",
      ),
      code: `std::vector<double> box{0, 0, 2, 2};
std::vector<double> truth{0.5, 0.5, 2.5, 2.5};
nexusloss::detection::IoULoss<double> loss;
double value = loss.forward(box, truth);
auto gradient = loss.backward(); // dL/dx1, dy1, dx2, dy2

double bare = nexusloss::detection::iou_loss<double>(box, truth);`,
      params: [
        { name: "box", detail: bi("Four values, max corner greater than min corner.", "Dört değer; max köşe min köşeden büyük.") },
      ],
    },
    {
      id: "giou",
      name: "GIoU",
      api: "detection::GIoULoss<T>",
      formula: "L = 1 − IoU + |C − U| / |C|",
      kind: bi("One box · enclosure", "Tek kutu · kuşatan kutu"),
      when: bi(
        "The first upgrade when boxes often do not overlap. The enclosing box C gives a gradient even in the disjoint regime.",
        "Kutular sık sık örtüşmediğinde ilk yükseltme. Kuşatan kutu C, ayrık rejimde bile bir gradyan verir.",
      ),
      logic: bi(
        "Rezatofighi et al. subtract the empty fraction of the smallest enclosing box. Two boxes that are far apart enclose a lot of wasted area, so the loss rises above the IoU plateau and points toward closing the gap. Once they overlap, the extra term shrinks and ordinary IoU takes over.",
        "Rezatofighi ve arkadaşları, en küçük kuşatan kutunun boş kesrini çıkarır. Birbirinden uzak iki kutu çok boşa alan kuşatır; kayıp IoU platosunun üstüne çıkar ve aralığı kapatmaya yönelir. Örtüşmeye başlayınca ek terim küçülür ve sıradan IoU devralır.",
      ),
      gradient: bi(
        "IoU gradient plus the partials of the enclosing area. backward() returns four components.",
        "IoU gradyanı artı kuşatan alanın kısmi türevleri. backward() dört bileşen döndürür.",
      ),
      code: `nexusloss::detection::GIoULoss<double> loss;
double value = loss.forward(box, truth);
auto gradient = loss.backward();`,
      params: [
        { name: "box", detail: bi("Same [x1, y1, x2, y2] contract.", "Aynı [x1, y1, x2, y2] sözleşmesi.") },
      ],
    },
    {
      id: "diou",
      name: "DIoU",
      api: "detection::DIoULoss<T>",
      formula: "L = 1 − IoU + ρ² / c²",
      kind: bi("One box · center distance", "Tek kutu · merkez mesafesi"),
      when: bi(
        "When you want the centers to meet directly, not merely to shrink an enclosing box. DIoU converges faster than GIoU on boxes that are already roughly the right size.",
        "Merkezlerin doğrudan buluşmasını, yalnızca kuşatan kutunun küçülmesini değil, istiyorsan. DIoU, boyutu kabaca doğru olan kutularda GIoU'dan daha hızlı yakınsar.",
      ),
      logic: bi(
        "ρ is the Euclidean distance between box centers. c is the diagonal of the smallest enclosing box, so the penalty is scale-free. A predicted box with the right area but the wrong center still pays, and the gradient points straight at the target center instead of wandering through the enclosure term.",
        "ρ, kutu merkezleri arasındaki Öklid mesafesidir. c, en küçük kuşatan kutunun köşegenidir; ceza ölçekten bağımsızdır. Alanı doğru ama merkezi yanlış bir tahmin yine öder ve gradyan kuşatma teriminde dolaşmak yerine dosdoğru hedef merkeze işaret eder.",
      ),
      gradient: bi(
        "IoU partials plus ∂(ρ²/c²)/∂edges. The center penalty is active even when IoU is already high.",
        "IoU kısmi türevleri artı ∂(ρ²/c²)/∂kenarlar. IoU zaten yüksekken bile merkez cezası aktiftir.",
      ),
      code: `nexusloss::detection::DIoULoss<double> loss;
double value = loss.forward(box, truth);`,
      params: [
        { name: "box", detail: bi("Four coordinates.", "Dört koordinat.") },
      ],
    },
    {
      id: "ciou",
      name: "CIoU",
      api: "detection::CIoULoss<T>",
      formula: "L = 1 − IoU + ρ²/c² + α v",
      kind: bi("One box · aspect ratio", "Tek kutu · en-boy"),
      when: bi(
        "The usual box regression loss when aspect ratio matters: a tall box should not be an equally good match for a wide box just because the centers agree.",
        "En-boy önemliyken alışılmış kutu regresyon kaybı: merkezler uyuyor diye uzun bir kutu, geniş bir kutuyla eşit iyi eşleşme sayılmamalı.",
      ),
      logic: bi(
        "Zheng et al. add v, a normalized difference of arctangents of the aspect ratios, weighted by α which grows as IoU improves. Early in training the center term moves the box; later the aspect term refines the shape. v compares w/h, so a uniform scale change that preserves aspect does not pay the extra penalty.",
        "Zheng ve arkadaşları, en-boy oranlarının arktanjant farkının normalize hali olan v'yi ekler; ağırlık α, IoU iyileştikçe büyür. Eğitimin başında merkez terimi kutuyu taşır; sonra en-boy terimi şekli inceltir. v, w/h karşılaştırır; en-boyu koruyan düzgün bir ölçek değişimi ek cezayı ödemez.",
      ),
      gradient: bi(
        "DIoU gradient plus the aspect partials on width and height. The four returned components are still dL/d(x1,y1,x2,y2).",
        "DIoU gradyanı artı genişlik ve yükseklik üzerindeki en-boy kısmi türevleri. Dönen dört bileşen hâlâ dL/d(x1,y1,x2,y2)'dir.",
      ),
      code: `nexusloss::detection::CIoULoss<double> loss;
double value = loss.forward(box, truth);
auto gradient = loss.backward();`,
      params: [
        { name: "box", detail: bi("Positive width and height so the aspect ratio is defined.", "En-boy tanımlı olsun diye pozitif genişlik ve yükseklik.") },
      ],
    },
    {
      id: "det-focal",
      name: "Detection focal",
      api: "detection::DetectionFocalLoss<T>",
      formula: "mean −α_t (1−p_t)^γ log(p_t),  p = σ(x)",
      kind: bi("Logits · dense classifier", "Logit · yoğun sınıflandırıcı"),
      when: bi(
        "The classification head of a one-stage detector. Tens of thousands of anchors are background. This is RetinaNet's focal loss on logits, stable like BCE-with-logits, not the probability FocalLoss.",
        "Tek aşamalı bir detektörün sınıflandırma başı. On binlerce çapa arka plandır. Bu, olasılık FocalLoss'u değil, logitler üzerinde RetinaNet focal kaybıdır; BCE-with-logits gibi kararlıdır.",
      ),
      logic: bi(
        "Each logit is an independent sigmoid, not a softmax across classes, which is what you want when an anchor can be background on every class at once. The modulating factor quiets anchors the model has already rejected. The returned value is the mean over the vector, and the gradient is scaled by 1/N to match.",
        "Her logit bağımsız bir sigmoiddir, sınıflar arası softmax değildir; bir çapanın her sınıfta birden arka plan olabildiği durumda istenen budur. Modülasyon çarpanı, modelin çoktan reddettiği çapaları susturur. Dönen değer vektörün ortalamasıdır ve gradyan buna uyacak şekilde 1/N ile ölçeklenir.",
      ),
      gradient: bi(
        "Analytical in the logit, combining the focal modulator with the sigmoid Jacobian. α defaults to 0.25, γ to 2.",
        "Logitte analitiktir; focal modülatörü sigmoid Jakobiyeni ile birleştirir. α varsayılanı 0.25, γ varsayılanı 2'dir.",
      ),
      code: `std::vector<double> logits{2.0, -3.0, 0.1};
std::vector<double> objectness{1.0, 0.0, 0.0};
nexusloss::detection::DetectionFocalLoss<double> loss(0.25, 2.0);
double value = loss.forward(logits, objectness);
auto gradient = loss.backward();`,
      params: [
        { name: "alpha", detail: bi("[0, 1]. Default 0.25.", "[0, 1]. Varsayılan 0.25.") },
        { name: "gamma", detail: bi("≥ 0. Default 2.", "≥ 0. Varsayılan 2.") },
      ],
    },
    {
      id: "smooth-l1-bbox",
      name: "Smooth L1 bbox",
      api: "detection::SmoothL1BBoxLoss<T>",
      formula: "Huber on encoded box deltas",
      kind: bi("Offset vector · mean", "Ofset vektörü · ortalama"),
      when: bi(
        "The regression head that predicts encoded deltas (dx, dy, dw, dh), not raw corners. Faster R-CNN and its children train this against the target encoding, while IoU losses train the decoded box.",
        "Ham köşeleri değil, kodlanmış deltaları (dx, dy, dw, dh) tahmin eden regresyon başı. Faster R-CNN ve çocukları bunu hedef kodlamaya karşı eğitir; IoU kayıpları çözülmüş kutuyu eğitir.",
      ),
      logic: bi(
        "Same Huber shape as nexusloss::HuberLoss: quadratic inside beta, linear outside. The class is separate because a detector step wants one scalar for a delta vector and a gradient of the same length, averaged, without going through LossBase's element-wise reduction enum. beta is the corner, default 1.",
        "nexusloss::HuberLoss ile aynı Huber biçimi: beta içinde karesel, dışında doğrusal. Sınıf ayrıdır çünkü bir detektör adımı, LossBase'in eleman indirgeme enum'una girmeden, bir delta vektörü için bir skaler ve aynı uzunlukta ortalanmış bir gradyan ister. beta köşedir, varsayılan 1.",
      ),
      gradient: bi(
        "Per-component Huber derivative, divided by the vector length so forward()'s mean and backward() agree.",
        "Bileşen başına Huber türevi, forward()'un ortalaması ile backward() uyuşsun diye vektör uzunluğuna bölünür.",
      ),
      code: `std::vector<double> delta{0.1, -0.2, 0.05, 0.4};
std::vector<double> target{0.0, 0.0, 0.0, 0.0};
nexusloss::detection::SmoothL1BBoxLoss<double> loss(1.0); // beta
double value = loss.forward(delta, target);
auto gradient = loss.backward();`,
      params: [
        { name: "beta", detail: bi("Positive. Default 1. The Huber corner.", "Pozitif. Varsayılan 1. Huber köşesi.") },
      ],
    },
  ],
};
