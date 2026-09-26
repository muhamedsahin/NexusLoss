import { bi, type CategoryDoc } from "./types";

export const metric: CategoryDoc = {
  id: "metric",
  index: "05",
  title: bi("Metric learning", "Metrik öğrenme"),
  lede: bi(
    "Embeddings and angular classifiers. Contrastive and triplet compare Euclidean neighbors. ArcFace, CosFace and SphereFace expect cosine logits that you already normalized — they do not L2-normalize a raw feature vector for you. Gradients come back concatenated in the same order as the inputs.",
    "Gömüler ve açısal sınıflandırıcılar. Contrastive ve triplet Öklid komşularını karşılaştırır. ArcFace, CosFace ve SphereFace, senin çoktan normalize ettiğin kosinüs logitleri bekler; ham özellik vektörünü senin yerine L2-normalize etmezler. Gradyanlar girdilerle aynı sırada birleşik döner.",
  ),
  losses: [
    {
      id: "contrastive",
      name: "Contrastive",
      api: "metric::ContrastiveLoss<T>",
      formula: "same: ‖a−b‖²\ndifferent: max(0, m − ‖a−b‖)²",
      kind: bi("Pair · Euclidean", "Çift · Öklid"),
      when: bi(
        "Siamese nets and verification: two embeddings, a boolean that says whether they are the same identity. Pull matches together, push non-matches out to a margin and then stop.",
        "Siyam ağları ve doğrulama: iki gömü, aynı kimlik olup olmadıklarını söyleyen bir boolean. Eşleşmeleri birbirine çek, eşleşmeyenleri bir marja kadar it, sonra dur.",
      ),
      logic: bi(
        "A positive pair pays the squared distance, so the only rest point is a = b. A negative pair pays only while it is inside the margin; beyond m the loss is zero and that pair is done. The margin is in distance units of your embedding, default 1, which only makes sense if the vectors are on a comparable scale.",
        "Pozitif çift karesel mesafe öder; tek dinlenme noktası a = b'dir. Negatif çift yalnızca marjın içindeyken öder; m'nin ötesinde kayıp sıfırdır ve o çift bitmiştir. Marj gömünün mesafe birimindedir, varsayılan 1; vektörler karşılaştırılabilir ölçekteyse anlamlıdır.",
      ),
      gradient: bi(
        "backward() returns 2·dim values: ∂L/∂a followed by ∂L/∂b. Negatives outside the margin contribute zeros.",
        "backward() 2·boyut değer döndürür: önce ∂L/∂a, sonra ∂L/∂b. Marj dışındaki negatifler sıfır katar.",
      ),
      code: `std::vector<double> a{1, 0}, b{0.2, 0};
nexusloss::metric::ContrastiveLoss<double> loss(1.0); // margin
double value = loss.forward(a, b, /*same class*/ true);
auto gradient = loss.backward();`,
      params: [
        { name: "margin", detail: bi("Default 1. Distance at which negatives go silent.", "Varsayılan 1. Negatiflerin sustuğu mesafe.") },
        { name: "same_class", detail: bi("true pulls, false pushes.", "true çeker, false iter.") },
      ],
    },
    {
      id: "triplet",
      name: "Triplet",
      api: "metric::TripletLoss<T>",
      formula: "L = max(0, ‖a−p‖² − ‖a−n‖² + m)",
      kind: bi("Triple · relative", "Üçlü · göreli"),
      when: bi(
        "Retrieval and face verification when you have an anchor, a positive and a negative in the same batch. The absolute distances do not matter; the gap between them does.",
        "Aynı batch'te bir çapa, bir pozitif ve bir negatif varken erişim ve yüz doğrulama. Mutlak mesafeler önemli değildir; aralarındaki boşluk önemlidir.",
      ),
      logic: bi(
        "The positive must be closer than the negative by at least m. If the gap is already wide enough the loss is zero: easy triplets teach nothing, which is why miners exist. NexusLoss computes the loss and the gradient of the triple you hand it. Mining — picking the hard negative — stays in your data pipeline.",
        "Pozitif, negatiften en az m kadar yakın olmalıdır. Boşluk zaten yeterince genişse kayıp sıfırdır: kolay üçlüler hiçbir şey öğretmez, madencilerin var olma nedeni budur. NexusLoss senin verdiğin üçlünün kaybını ve gradyanını hesaplar. Madencilik — zor negatifi seçmek — veri hattında kalır.",
      ),
      gradient: bi(
        "Concatenated ∂L/∂anchor, ∂L/∂positive, ∂L/∂negative. A satisfied triplet is all zeros.",
        "Birleşik ∂L/∂çapa, ∂L/∂pozitif, ∂L/∂negatif. Tatmin olmuş bir üçlü tamamen sıfırdır.",
      ),
      code: `std::vector<double> anchor{0, 0}, positive{0.1, 0}, negative{1, 0};
nexusloss::metric::TripletLoss<double> loss(0.2);
double value = loss.forward(anchor, positive, negative);
auto gradient = loss.backward();`,
      params: [
        { name: "margin", detail: bi("Default 1. Added to the positive distance.", "Varsayılan 1. Pozitif mesafeye eklenir.") },
      ],
    },
    {
      id: "cosine-embedding",
      name: "Cosine embedding",
      api: "metric::cosine_embedding_loss",
      formula: "same: 1 − cos\ndifferent: max(0, cos − m)",
      kind: bi("Free function · angle", "Serbest fonksiyon · açı"),
      when: bi(
        "When magnitude should not matter and you want to attract or repel directions. Sentence embeddings and normalized image towers.",
        "Büyüklük önemli olmasın, yönleri çekmek veya itmek istiyorsan. Cümle gömüleri ve normalize görüntü kuleleri.",
      ),
      logic: bi(
        "Cosine similarity is the dot product of the two L2 directions. Matches want it at 1. Non-matches are punished only while the cosine stays above the margin (default 0, so orthogonal is already enough). Unlike contrastive loss, a long vector and a short vector that point the same way are a perfect match.",
        "Kosinüs benzerliği, iki L2 yönünün iç çarpımıdır. Eşleşmeler onun 1 olmasını ister. Eşleşmeyenler yalnızca kosinüs marjın üstündeyken cezalandırılır (varsayılan 0, yani dik olmak yeter). Contrastive kaybın aksine, aynı yöne bakan uzun ve kısa bir vektör mükemmel eşleşmedir.",
      ),
      gradient: bi(
        "Scalar helper. The derivative hits both vectors through the normalized dot product; zero vectors are undefined and should be guarded upstream.",
        "Skaler yardımcı. Türev normalize iç çarpım üzerinden iki vektöre de değer. Sıfır vektörler tanımsızdır; yukarı akışta korunmalıdır.",
      ),
      code: `double value = nexusloss::metric::cosine_embedding_loss<double>(a, b, true, /*margin*/ 0.0);`,
      params: [
        { name: "margin", detail: bi("Default 0. Negatives with cosine below this are free.", "Varsayılan 0. Kosinüsü bunun altındaki negatifler bedavadır.") },
      ],
    },
    {
      id: "center",
      name: "Center",
      api: "metric::center_loss",
      formula: "L = ½ ‖x − c_y‖²",
      kind: bi("Free function · class mean", "Serbest fonksiyon · sınıf ortalaması"),
      when: bi(
        "Beside softmax, to pull each embedding toward a learned class center so within-class scatter shrinks without a mined triplet.",
        "Softmax'ın yanında, her gömüyü öğrenilmiş bir sınıf merkezine çekmek için; böylece sınıf içi saçılma, kazılmış bir üçlü olmadan küçülür.",
      ),
      logic: bi(
        "Wen et al. keep a center per class and penalize Euclidean distance to it. The center is not updated by this function — you own that moving average. The loss only says how far this sample sits from the center you pass in. It is usually added to cross-entropy with a small weight, because alone it can collapse every class to one point.",
        "Wen ve arkadaşları sınıf başına bir merkez tutar ve ona Öklid mesafesini cezalandırır. Merkezi bu fonksiyon güncellemez; o hareketli ortalamaya sen sahipsin. Kayıp yalnızca bu örneğin verdiğin merkeze ne kadar uzak olduğunu söyler. Genellikle küçük bir ağırlıkla çapraz entropiye eklenir; tek başına her sınıfı tek noktaya çökertebilir.",
      ),
      gradient: bi(
        "∂L/∂x = x − c. The center's own update is outside the library.",
        "∂L/∂x = x − c. Merkezin kendi güncellemesi kütüphanenin dışındadır.",
      ),
      code: `std::vector<double> embedding{0.4, -0.1};
std::vector<double> center{0.0, 0.0};
double value = nexusloss::metric::center_loss<double>(embedding, center);`,
      params: [
        { name: "class_center", detail: bi("Same dimension as the embedding. Maintained by you.", "Gömüyle aynı boyut. Senin tarafında tutulur.") },
      ],
    },
    {
      id: "n-pair",
      name: "N-pair",
      api: "metric::n_pair_loss",
      formula: "−log  exp(a·p) / (exp(a·p) + Σ exp(a·n))",
      kind: bi("Free function · multi-negative", "Serbest fonksiyon · çok negatif"),
      when: bi(
        "One positive and several negatives in a single softmax, the multi-class version of a triplet. Efficient when the batch already holds one example per class.",
        "Tek bir softmax içinde bir pozitif ve birkaç negatif; üçlünün çok sınıflı hali. Batch zaten sınıf başına bir örnek tutuyorsa verimlidir.",
      ),
      logic: bi(
        "Sohn's N-pair loss is InfoNCE with dot-product similarities and temperature 1. The positive must beat every negative at once, which is a harder and less noisy signal than one random triplet. negative_embeddings is packed as count × dimension, contiguous.",
        "Sohn'un N-pair kaybı, iç çarpım benzerlikleri ve sıcaklık 1 ile InfoNCE'dir. Pozitif, her negatifi aynı anda yenmelidir; bu, rastgele bir üçlüden daha sert ve daha az gürültülü bir sinyaldir. negative_embeddings, sayı × boyut olarak bitişik paketlenir.",
      ),
      gradient: bi(
        "Scalar helper. The softmax over pair scores is the gradient with respect to the dots, then the dots carry it to the embeddings.",
        "Skaler yardımcı. Çift skorları üzerindeki softmax, iç çarpımlara göre gradyandır; iç çarpımlar onu gömülere taşır.",
      ),
      code: `std::vector<double> negatives{1, 0,  0, 1}; // 2 negatives, dim 2
double value = nexusloss::metric::n_pair_loss<double>(a, b, negatives, /*count*/ 2);`,
      params: [
        { name: "negative_count", detail: bi("negatives.size() must equal count × anchor.size().", "negatives.size(), sayı × çapa boyutu olmalıdır.") },
      ],
    },
    {
      id: "arcface",
      name: "ArcFace",
      api: "metric::ArcFaceLoss<T>",
      formula: "CE( s · cos(θ_c + m), … )",
      kind: bi("Cosine logits · additive angle", "Kosinüs logit · eklemeli açı"),
      when: bi(
        "Face recognition and any closed-set identity head where the decision should happen on the angle, with a fixed margin on that angle. The most common angular loss.",
        "Kararın açı üzerinde olması ve o açıda sabit bir marj istenen yüz tanıma ve her kapalı küme kimlik başı. En yaygın açısal kayıp.",
      ),
      logic: bi(
        "You pass cosine logits, one per class, already equal to the normalized weight-feature dot. ArcFace adds the margin m to the target angle before scaling by s and running cross-entropy. Geometrically the target class's acceptance cone gets narrower. Other classes are untouched. Defaults are m = 0.5 radians and s = 64, the ArcFace paper's operating point. The loss does not normalize for you; a cosine outside [−1, 1] is a caller bug.",
        "Sınıf başına bir tane, çoktan normalize ağırlık-özellik iç çarpımına eşit kosinüs logitleri verirsin. ArcFace, ölçek s ve çapraz entropiden önce hedef açıya marj m ekler. Geometrik olarak hedef sınıfın kabul konisi daralır. Diğer sınıflara dokunulmaz. Varsayılanlar m = 0.5 radyan ve s = 64'tür; ArcFace makalesinin çalışma noktası. Kayıp senin yerine normalize etmez; [−1, 1] dışı bir kosinüs çağıran tarafın hatasıdır.",
      ),
      gradient: bi(
        "backward() returns one partial per cosine logit. The target slot is scaled by the derivative of cos(θ+m) with respect to cos θ.",
        "backward() kosinüs logit başına bir kısmi türev döndürür. Hedef yuvası, cos(θ+m)'nin cos θ'ya göre türeviyle ölçeklenir.",
      ),
      code: `std::vector<double> cosine{0.2, 0.9, 0.1}; // already normalized
nexusloss::metric::ArcFaceLoss<double> loss(0.5, 64);
double value = loss.forward(cosine, /*class*/ 1);
auto gradient = loss.backward();`,
      params: [
        { name: "margin", detail: bi("Radians added to the target angle. Default 0.5.", "Hedef açıya eklenen radyan. Varsayılan 0.5.") },
        { name: "scale", detail: bi("Logit multiplier. Default 64.", "Logit çarpanı. Varsayılan 64.") },
      ],
    },
    {
      id: "cosface",
      name: "CosFace",
      api: "metric::CosFaceLoss<T>",
      formula: "CE( s · (cos θ_c − m), … )",
      kind: bi("Cosine logits · additive cosine", "Kosinüs logit · eklemeli kosinüs"),
      when: bi(
        "Same normalized classifier as ArcFace, when you would rather subtract a constant from the target cosine than add a constant to the angle. Slightly simpler geometry, similar accuracy.",
        "ArcFace ile aynı normalize sınıflandırıcı; açıya sabit eklemek yerine hedef kosinüsten sabit çıkarmayı tercih ettiğinde. Geometri biraz daha sade, doğruluk benzer.",
      ),
      logic: bi(
        "CosFace (large-margin cosine loss) shifts the target logit by −m in cosine space, then scales. The margin is a cosine gap, default 0.35, not an angle. Because the shift is linear in the cosine, the gradient factor on the target logit is constant, unlike ArcFace's trigonometric chain.",
        "CosFace (büyük marjlı kosinüs kaybı) hedef logiti kosinüs uzayında −m kaydırır, sonra ölçekler. Marj bir açı değil, kosinüs boşluğudur; varsayılan 0.35. Kaydırma kosinüste doğrusal olduğu için hedef logit üzerindeki gradyan çarpanı sabittir; ArcFace'in trigonometrik zinciri gibi değil.",
      ),
      gradient: bi(
        "Softmax-minus-target on the modified logits, multiplied by s, with the target cosine's shift treated as constant.",
        "Değiştirilmiş logitler üzerinde softmax-eksi-hedef, s ile çarpılır; hedef kosinüsün kaydırması sabit sayılır.",
      ),
      code: `nexusloss::metric::CosFaceLoss<double> loss(0.35, 64);
double value = loss.forward(cosine, 1);`,
      params: [
        { name: "margin", detail: bi("Subtracted from the target cosine. Default 0.35.", "Hedef kosinüsten çıkarılır. Varsayılan 0.35.") },
        { name: "scale", detail: bi("Default 64.", "Varsayılan 64.") },
      ],
    },
    {
      id: "sphereface",
      name: "SphereFace",
      api: "metric::sphereface_loss",
      formula: "CE( s · cos(m θ_c), … )",
      kind: bi("Free function · multiplicative angle", "Serbest fonksiyon · çarpımsal açı"),
      when: bi(
        "The earlier angular-margin idea: multiply the target angle by m instead of adding to it. Useful as an ablation next to ArcFace. Integer m is the classical setting.",
        "Daha eski açısal marj fikri: hedef açıyı eklemek yerine m ile çarpmak. ArcFace'in yanında bir ablasyon olarak yararlıdır. Tam sayı m klasik ayardır.",
      ),
      logic: bi(
        "SphereFace requires cos(mθ) while the network only produced cos θ. The implementation folds that multiplicative margin into the angular-margin cross-entropy (kind = multiplicative). It is stricter near the decision boundary and undefined in spirit once mθ leaves the range where the cosine polynomial is a faithful extension. Prefer ArcFace unless you are reproducing the 2017 paper.",
        "SphereFace cos(mθ) ister, ağ ise yalnızca cos θ üretmiştir. Uygulama bu çarpımsal marjı açısal marj çapraz entropisine katlar. Karar sınırına yakın daha serttir; mθ, kosinüs polinomunun sadık bir genişleme olduğu aralığı terk edince ruhen tanımsızlaşır. 2017 makalesini yeniden üretmiyorsan ArcFace'i tercih et.",
      ),
      gradient: bi(
        "Scalar on the free function. The class path for angular margins is ArcFaceLoss and CosFaceLoss, both of which cache backward().",
        "Serbest fonksiyon üzerinde skaler. Açısal marjların sınıf yolu ArcFaceLoss ve CosFaceLoss'tur; ikisi de backward() önbellekler.",
      ),
      code: `double value = nexusloss::metric::sphereface_loss<double>(cosine, 1, /*margin*/ 1.0, /*scale*/ 64.0);`,
      params: [
        { name: "margin", detail: bi("Multiplies the target angle. Default 1, which disables the extra margin.", "Hedef açıyı çarpar. Varsayılan 1, ek marjı kapatır.") },
      ],
    },
  ],
};

export const ranking: CategoryDoc = {
  id: "ranking",
  index: "06",
  title: bi("Ranking", "Sıralama"),
  lede: bi(
    "Scores are higher-is-better. Pair losses compare two items. List losses take a score vector and a relevance vector of the same length and want the sort order of the scores to match the sort order of the relevance.",
    "Skorlar yüksek-olan-iyidir. Çift kayıpları iki öğeyi karşılaştırır. Liste kayıpları aynı uzunlukta bir skor vektörü ve bir ilgililik vektörü alır; skorların sıralamasının ilgililiğin sıralamasıyla örtüşmesini ister.",
  ),
  losses: [
    {
      id: "margin-ranking",
      name: "Margin ranking",
      api: "ranking::MarginRankingLoss<T>",
      formula: "L = max(0, −y (x1 − x2) + m)",
      kind: bi("Pair · signed", "Çift · işaretli"),
      when: bi(
        "You know which of two scores should win, encoded as y = +1 if x1 should be larger and y = −1 otherwise. Preference data with a gap, not a full list.",
        "İki skordan hangisinin kazanması gerektiğini biliyorsun; x1 büyük olmalıysa y = +1, değilse y = −1. Tam bir liste değil, boşluklu tercih verisi.",
      ),
      logic: bi(
        "This is hinge on the difference. If the preferred score already leads by m, the pair is finished. The same function exists in the metric namespace for a single comparison; the ranking class caches forward/backward for the training step.",
        "Bu, fark üzerindeki hinge'dir. Tercih edilen skor zaten m kadar öndeyse çift bitmiştir. Aynı fonksiyon tek bir karşılaştırma için metric ad alanında da durur; sıralama sınıfı eğitim adımı için forward/backward önbellekler.",
      ),
      gradient: bi(
        "backward() returns two numbers, ∂L/∂x1 and ∂L/∂x2. They are opposites, and both zero when the margin holds.",
        "backward() iki sayı döndürür: ∂L/∂x1 ve ∂L/∂x2. Birbirinin tersidir ve marj tutuyorsa ikisi de sıfırdır.",
      ),
      code: `nexusloss::ranking::MarginRankingLoss<double> loss(0.0);
double value = loss.forward(/*x1*/ 0.2, /*x2*/ 0.8, /*y*/ 1.0);
auto gradient = loss.backward();`,
      params: [
        { name: "margin", detail: bi("Default 0.", "Varsayılan 0.") },
        { name: "target", detail: bi("+1 if x1 should outrank x2, else −1.", "x1, x2'yi geçmeliyse +1, değilse −1.") },
      ],
    },
    {
      id: "ranknet",
      name: "RankNet",
      api: "ranking::RankNetLoss<T>",
      formula: "L = log(1 + exp(−(s⁺ − s⁻)))",
      kind: bi("Pair · logistic", "Çift · lojistik"),
      when: bi(
        "Click or preference pairs when you want a smooth probability that the preferred item outranks the other, with no hard margin.",
        "Sert bir marj olmadan, tercih edilen öğenin diğerini geçme olasılığının pürüzsüz olmasını istediğin tıklama veya tercih çiftleri.",
      ),
      logic: bi(
        "Burges et al. model P(i beats j) as a sigmoid of the score gap. The loss is the logistic negative log-likelihood of the observed order. Unlike hinge, a pair that is already correct still produces a small gradient, so the scores keep separating. ranknet_loss is an alias of pairwise_logistic.",
        "Burges ve arkadaşları P(i, j'yi yener)'i skor farkının sigmoidi olarak modeller. Kayıp, gözlenen sıranın lojistik negatif log-olabilirliğidir. Hinge'in aksine zaten doğru olan bir çift küçük bir gradyan üretmeye devam eder; skorlar ayrılmayı sürdürür. ranknet_loss, pairwise_logistic'in takma adıdır.",
      ),
      gradient: bi(
        "∂L/∂s⁺ = σ(s⁻ − s⁺) − 1 and ∂L/∂s⁻ = −that. The preferred score is always nudged up.",
        "∂L/∂s⁺ = σ(s⁻ − s⁺) − 1 ve ∂L/∂s⁻ = −o. Tercih edilen skor her zaman yukarı dürtülür.",
      ),
      code: `nexusloss::ranking::RankNetLoss<double> loss;
double value = loss.forward(/*preferred*/ 0.4, /*other*/ 1.1);
auto gradient = loss.backward();`,
      params: [
        { name: "preferred", detail: bi("Score that should be higher.", "Daha yüksek olması gereken skor.") },
        { name: "other", detail: bi("Score that should be lower.", "Daha düşük olması gereken skor.") },
      ],
    },
    {
      id: "bpr",
      name: "BPR",
      api: "ranking::bpr_loss",
      formula: "L = −log σ(s⁺ − s⁻)",
      kind: bi("Free function · implicit", "Serbest fonksiyon · örtük"),
      when: bi(
        "Implicit feedback recommenders: one item the user touched, one item they did not. Bayesian Personalized Ranking is RankNet with the sign fixed so the positive always comes first.",
        "Örtük geri bildirim önericileri: kullanıcının dokunduğu bir öğe, dokunmadığı bir öğe. Bayesian Personalized Ranking, pozitifin her zaman önce gelmesi için işareti sabitlenmiş RankNet'tir.",
      ),
      logic: bi(
        "Rendle et al. maximize the probability that an observed item outranks a sampled unobserved one. There is no relevance grade, only the pair. Sampling the negative is your job; the function scores the pair you sampled.",
        "Rendle ve arkadaşları, gözlenen bir öğenin örneklenen gözlenmemiş bir öğeyi geçme olasılığını maksimize eder. İlgililik derecesi yoktur, yalnızca çift vardır. Negatifi örneklemek senin işin; fonksiyon örneklediğin çifti skorlar.",
      ),
      gradient: bi(
        "Same logistic pair gradient as RankNet. The free function returns the scalar; RankNetLoss::backward is the cached form if you store the two scores.",
        "RankNet ile aynı lojistik çift gradyanı. Serbest fonksiyon skaler döner; iki skoru saklıyorsan önbellekli biçim RankNetLoss::backward'tır.",
      ),
      code: `double value = nexusloss::ranking::bpr_loss<double>(/*positive*/ 1.2, /*negative*/ 0.3);`,
      params: [
        { name: "scores", detail: bi("Higher means more likely to be preferred.", "Yüksek, tercih edilmeye daha yakın demektir.") },
      ],
    },
    {
      id: "listnet",
      name: "ListNet",
      api: "ranking::ListNetLoss<T>",
      formula: "KL( softmax(relevance) ‖ softmax(scores) )",
      kind: bi("List · top-1", "Liste · top-1"),
      when: bi(
        "A whole slate with graded relevance, when you care about the probability of each item being the top pick, not about every pairwise argument.",
        "Her ikili tartışmayı değil, her öğenin zirve seçim olma olasılığını önemsediğin, dereceli ilgililiği olan bütün bir liste.",
      ),
      logic: bi(
        "Cao et al. put a softmax on the relevance grades and a softmax on the scores, then take the cross-entropy between those distributions. Items with high relevance must receive high score mass. Ties in relevance become a soft target. The top-1 form does not see the full permutation, which keeps it O(n).",
        "Cao ve arkadaşları ilgililik derecelerine ve skorlara softmax koyar, sonra bu dağılımlar arasında çapraz entropi alır. Yüksek ilgililiği olan öğeler yüksek skor kütlesi almalıdır. İlgililik beraberlikleri yumuşak hedef olur. Top-1 biçimi tam permütasyonu görmez; bu onu O(n) tutar.",
      ),
      gradient: bi(
        "∂L/∂scores = softmax(scores) − softmax(relevance). backward() returns one entry per item.",
        "∂L/∂skorlar = softmax(skorlar) − softmax(ilgililik). backward() öğe başına bir girdi döndürür.",
      ),
      code: `std::vector<double> scores{0.2, 1.4, 0.0};
std::vector<double> relevance{0.0, 2.0, 1.0};
nexusloss::ranking::ListNetLoss<double> loss;
double value = loss.forward(scores, relevance);
auto gradient = loss.backward();`,
      params: [
        { name: "relevance", detail: bi("Same length as scores. Larger means should rank earlier.", "Skorlarla aynı uzunluk. Büyük olan daha önce sıralanmalı.") },
      ],
    },
    {
      id: "listmle",
      name: "ListMLE",
      api: "ranking::ListMLELoss<T>",
      formula: "− Σ log P(next item | remaining)",
      kind: bi("List · Plackett-Luce", "Liste · Plackett-Luce"),
      when: bi(
        "You have a total order, or a relevance vector that induces one, and you want the likelihood of that exact permutation rather than only the top-1 distribution.",
        "Tam bir sıran, ya da onu doğuran bir ilgililik vektörün var ve yalnızca top-1 dağılımını değil o permütasyonun olabilirliğini istiyorsun.",
      ),
      logic: bi(
        "Plackett-Luce says the probability of a ranking is the product of softmaxes over the items still remaining. ListMLE sorts your items by relevance (stable, so ties keep input order) and scores that permutation. It uses the full list, so it is sharper than ListNet and more expensive only by the sort.",
        "Plackett-Luce, bir sıralamanın olasılığının, hâlâ kalan öğeler üzerindeki softmax'ların çarpımı olduğunu söyler. ListMLE öğeleri ilgililiğe göre sıralar (kararlıdır, beraberlikler girdi sırasını korur) ve o permütasyonu skorlar. Tam listeyi kullanır; ListNet'ten daha keskindir ve ek maliyeti yalnızca sıralamadır.",
      ),
      gradient: bi(
        "Each position's gradient is softmax-over-the-suffix minus one on the chosen item. backward() follows the cached scores and relevance.",
        "Her konumun gradyanı, sonek üzerindeki softmax eksi seçilen öğedeki bir'dir. backward() önbellekteki skor ve ilgililiği izler.",
      ),
      code: `nexusloss::ranking::ListMLELoss<double> loss;
double value = loss.forward(scores, relevance);
auto gradient = loss.backward();`,
      params: [
        { name: "relevance", detail: bi("Defines the target permutation via a stable descending sort.", "Kararlı azalan sıralama ile hedef permütasyonu tanımlar.") },
      ],
    },
    {
      id: "approx-ndcg",
      name: "ApproxNDCG",
      api: "ranking::approx_ndcg_loss",
      formula: "L = 1 − softDCG / idealDCG",
      kind: bi("Free function · metric surrogate", "Serbest fonksiyon · metrik vekili"),
      when: bi(
        "Learning to rank when the leaderboard number is NDCG and a pairwise logistic is only a proxy you no longer trust.",
        "Liderlik tablosundaki sayı NDCG ise ve ikili lojistiğe artık güvenmediğin öğrenerek sıralama.",
      ),
      logic: bi(
        "NDCG needs a hard rank, and hard rank is a sort. ApproxNDCG replaces the rank with a soft rank: each item's position is 1 plus a sum of sigmoids of score gaps, tempered by temperature. The ideal DCG sorts the relevance itself. Temperature 1 is the default; lower temperatures approach the true sort and the gradient gets peakier.",
        "NDCG sert bir sıra ister, sert sıra bir sıralamadır. ApproxNDCG sırayı yumuşak sırayla değiştirir: her öğenin konumu, skor farklarının sigmoidlerinin toplamı artı 1'dir; sıcaklık bunu yumuşatır. İdeal DCG ilgililiğin kendisini sıralar. Sıcaklık varsayılanı 1'dir; düşük sıcaklıklar gerçek sıralamaya yaklaşır ve gradyan sivrileşir.",
      ),
      gradient: bi(
        "Scalar helper. The soft ranks are differentiable in the scores; the ideal DCG is constant with respect to the model.",
        "Skaler yardımcı. Yumuşak sıralar skorlarda türevlenebilir; ideal DCG modele göre sabittir.",
      ),
      code: `double value = nexusloss::ranking::approx_ndcg_loss<double>(scores, relevance, /*temperature*/ 1.0);`,
      params: [
        { name: "temperature", detail: bi("Default 1. Lower is closer to a hard sort.", "Varsayılan 1. Düşük olan sert sıralamaya daha yakındır.") },
      ],
    },
    {
      id: "soft-rank",
      name: "SoftRank",
      api: "ranking::soft_rank_loss",
      formula: "same surrogate as ApproxNDCG",
      kind: bi("Free function · alias", "Serbest fonksiyon · takma ad"),
      when: bi(
        "Same objective as ApproxNDCG, under the name used by the SoftRank literature. Call whichever reads clearly next to your paper.",
        "ApproxNDCG ile aynı amaç, SoftRank literatürünün kullandığı adla. Makalenin yanında hangisi açık okunuyorsa onu çağır.",
      ),
      logic: bi(
        "soft_rank_loss forwards to the same temperature-smoothed DCG ratio. It exists so a call site can say SoftRank without a comment explaining the alias. Behavior, temperature and the ideal-DCG normalization match approx_ndcg_loss.",
        "soft_rank_loss, aynı sıcaklıkla yumuşatılmış DCG oranına iletir. Bir çağrı yerinin takma adı açıklayan bir yorum olmadan SoftRank diyebilmesi için durur. Davranış, sıcaklık ve ideal DCG normalizasyonu approx_ndcg_loss ile aynıdır.",
      ),
      gradient: bi(
        "Identical to ApproxNDCG.",
        "ApproxNDCG ile aynı.",
      ),
      code: `double value = nexusloss::ranking::soft_rank_loss<double>(scores, relevance, 1.0);`,
      params: [
        { name: "temperature", detail: bi("Default 1.", "Varsayılan 1.") },
      ],
    },
    {
      id: "lambda-rank",
      name: "LambdaRank",
      api: "ranking::lambda_rank_loss",
      formula: "pairwise logistic, weighted by |ΔNDCG|",
      kind: bi("Free function · NDCG-aware pairs", "Serbest fonksiyon · NDCG farkında çiftler"),
      when: bi(
        "You like RankNet's pairs but want a swap that changes NDCG a lot to count more than a swap near the bottom of the list.",
        "RankNet çiftlerini seviyorsun ama NDCG'yi çok değiştiren bir takasın, listenin dibine yakın bir takastan daha çok sayılmasını istiyorsun.",
      ),
      logic: bi(
        "Burges' LambdaRank multiplies each pair's logistic by the absolute NDCG delta of swapping those two items. NexusLoss approximates that weight with soft ranks and the ideal DCG, then returns a weighted mean of the pair losses. σ scales the logistic, default 1. High-relevance items near the top dominate, which is the entire point of NDCG.",
        "Burges'in LambdaRank'i her çiftin lojistiğini, o iki öğeyi takas etmenin mutlak NDCG deltasıyla çarpar. NexusLoss bu ağırlığı yumuşak sıralar ve ideal DCG ile yaklaştırır, sonra çift kayıplarının ağırlıklı ortalamasını döndürür. σ lojistiği ölçekler, varsayılan 1. Tepedeki yüksek ilgililikli öğeler hükmeder; NDCG'nin bütün amacı budur.",
      ),
      gradient: bi(
        "Scalar helper. The dominant term is the RankNet gradient of each pair, scaled by that pair's NDCG weight.",
        "Skaler yardımcı. Baskın terim, her çiftin NDCG ağırlığıyla ölçeklenmiş RankNet gradyanıdır.",
      ),
      code: `double value = nexusloss::ranking::lambda_rank_loss<double>(scores, relevance, /*sigma*/ 1.0);`,
      params: [
        { name: "sigma", detail: bi("Default 1. Steepness of the pairwise sigmoid.", "Varsayılan 1. İkili sigmoidin dikliği.") },
      ],
    },
  ],
};
