import { bi, type CategoryDoc } from "./types";

export const generative: CategoryDoc = {
  id: "generative",
  index: "07",
  title: bi("Generative", "Üretken"),
  lede: bi(
    "GAN scores, a VAE evidence bound, and image penalties that compare tensors you already computed. Perceptual and style losses do not run VGG. They compare feature vectors and Gram statistics you pass in. Diffusion compares predicted noise with the noise you added.",
    "GAN skorları, bir VAE kanıt sınırı ve senin çoktan hesapladığın tensörleri karşılaştıran görüntü cezaları. Algısal ve stil kayıpları VGG çalıştırmaz. Verdiğin özellik vektörlerini ve Gram istatistiklerini karşılaştırır. Difüzyon, tahmin edilen gürültüyü eklediğin gürültüyle karşılaştırır.",
  ),
  losses: [
    {
      id: "adversarial",
      name: "Adversarial (vanilla GAN)",
      api: "generative::AdversarialLoss<T>",
      formula: "D: −log σ(real) − log(1−σ(fake))\nG: −log σ(fake)",
      kind: bi("Logits · two players", "Logit · iki oyuncu"),
      when: bi(
        "The original Goodfellow minimax game, written on logits so the logs stay finite. Use the discriminator and generator methods on alternate steps.",
        "Loglar sonlu kalsın diye logitler üzerinde yazılmış özgün Goodfellow minimax oyunu. Ayırıcı ve üretici metotlarını ardışık adımlarda kullan.",
      ),
      logic: bi(
        "The discriminator wants real logits high and fake logits low, which is BCE-with-logits against 1 and 0. The non-saturating generator wants the discriminator to call fakes real, so it minimizes −log σ(fake) instead of log(1−σ(fake)). That is the form that still has a gradient when the discriminator is winning.",
        "Ayırıcı gerçek logitlerin yüksek, sahte logitlerin düşük olmasını ister; bu, 1 ve 0'a karşı logits'li BCE'dir. Doymayan üretici, ayırıcının sahteleri gerçek saymasını ister; bu yüzden log(1−σ(fake)) yerine −log σ(fake) minimize eder. Ayırıcı kazanırken hâlâ gradyanı olan biçim budur.",
      ),
      gradient: bi(
        "discriminator_backward returns real partials followed by fake partials. generator_backward returns one partial per fake logit.",
        "discriminator_backward önce gerçek, sonra sahte kısmi türevleri döndürür. generator_backward sahte logit başına bir kısmi türev döndürür.",
      ),
      code: `std::vector<double> real{1.2, 0.4}, fake{-0.5, -1.1};
nexusloss::generative::AdversarialLoss<double> loss;
double d = loss.discriminator(real, fake);
auto d_grad = loss.discriminator_backward(real, fake);
double g = loss.generator(fake);
auto g_grad = loss.generator_backward(fake);`,
      params: [
        { name: "logits", detail: bi("Raw scores. The sigmoid lives inside the loss.", "Ham skorlar. Sigmoid kaybın içindedir.") },
      ],
    },
    {
      id: "wasserstein",
      name: "Wasserstein",
      api: "generative::WassersteinLoss<T>",
      formula: "D: mean(fake) − mean(real)\nG: −mean(fake)",
      kind: bi("Unbounded scores", "Sınırlanmamış skorlar"),
      when: bi(
        "WGAN, where the critic is a Lipschitz function and the scores are not probabilities. You enforce the Lipschitz constraint outside this loss, with weight clipping or a gradient penalty you add yourself.",
        "Eleştirmenin Lipschitz bir fonksiyon olduğu ve skorların olasılık olmadığı WGAN. Lipschitz kısıtını bu kaybın dışında, ağırlık kırpma ya da kendin eklediğin bir gradyan cezasıyla uygularsın.",
      ),
      logic: bi(
        "The Wasserstein critic maximizes the gap between real and fake scores. Writing the discriminator loss as mean(fake) − mean(real) means a minimizer does the right thing. The generator minimizes the negation of the fake scores, pushing them up. There is no log, so a confident critic does not saturate.",
        "Wasserstein eleştirmeni gerçek ve sahte skorlar arasındaki boşluğu maksimize eder. Ayırıcı kaybını mean(fake) − mean(real) yazmak, bir minimize edicinin doğru işi yapması demektir. Üretici sahte skorların negatifini minimize eder, onları yukarı iter. Log yoktur; kendinden emin bir eleştirmen doymaz.",
      ),
      gradient: bi(
        "Each real score receives −1/N, each fake score +1/N on the discriminator step. Generator gradients are −1/N on the fake scores.",
        "Ayırıcı adımında her gerçek skor −1/N, her sahte skor +1/N alır. Üretici gradyanları sahte skorlarda −1/N'dir.",
      ),
      code: `nexusloss::generative::WassersteinLoss<double> loss;
double d = loss.discriminator(real, fake);
auto d_grad = loss.discriminator_backward(real, fake);
double g = loss.generator(fake);`,
      params: [
        { name: "scores", detail: bi("Not logits and not probabilities. Keep the critic 1-Lipschitz yourself.", "Logit değil, olasılık değil. Eleştirmeni 1-Lipschitz sen tut.") },
      ],
    },
    {
      id: "lsgan",
      name: "Least squares GAN",
      api: "generative::LSGANLoss<T>",
      formula: "L = mean( (score − target)² )",
      kind: bi("Scores · Pearson χ²", "Skor · Pearson χ²"),
      when: bi(
        "When vanilla GAN saturates and WGAN's constraint is more machinery than you want. Mao et al. replace the sigmoid cross-entropy with a square.",
        "Vanilla GAN doyuyor ve WGAN kısıtı istediğinden fazla makineyse. Mao ve arkadaşları sigmoid çapraz entropinin yerine kare koyar.",
      ),
      logic: bi(
        "The discriminator is trained toward target 1 on real scores and target 0 on fake scores. The generator is trained so fake scores move toward 1. The square gives a gradient proportional to the error even when the score is on the correct side of 0.5, which is the non-saturating behavior people adopt LSGAN for.",
        "Ayırıcı, gerçek skorlarda hedef 1'e, sahte skorlarda hedef 0'a eğitilir. Üretici, sahte skorlar 1'e gitsin diye eğitilir. Kare, skor 0.5'in doğru tarafında olsa bile hatayla orantılı bir gradyan verir; insanların LSGAN'ı benimsemesinin nedeni bu doymayan davranıştır.",
      ),
      gradient: bi(
        "∂L/∂score = 2 (score − target) / N. backward() takes the same target you used in forward().",
        "∂L/∂skor = 2 (skor − hedef) / N. backward(), forward()'da kullandığın hedefi alır.",
      ),
      code: `nexusloss::generative::LSGANLoss<double> loss;
double d_real = loss.forward(real, /*target*/ 1.0);
auto grad = loss.backward(real, 1.0);
double g = loss.forward(fake, 1.0);`,
      params: [
        { name: "target", detail: bi("Conventionally 1 for real and for the generator, 0 for fake.", "Alışıldığı üzere gerçek ve üretici için 1, sahte için 0.") },
      ],
    },
    {
      id: "hinge-gan",
      name: "Hinge GAN",
      api: "generative::hinge_discriminator_loss",
      formula: "D: mean(relu(1−real) + relu(1+fake))\nG: −mean(fake)",
      kind: bi("Free functions · SAGAN / BigGAN", "Serbest fonksiyonlar · SAGAN / BigGAN"),
      when: bi(
        "Spectral-norm generators in the SAGAN and BigGAN line, where the discriminator is a margin classifier and the generator is a linear score.",
        "Ayırıcının bir marj sınıflandırıcısı, üreticinin doğrusal bir skor olduğu SAGAN ve BigGAN çizgisindeki spektral-norm üreticiler.",
      ),
      logic: bi(
        "Real scores above 1 and fake scores below −1 stop teaching the discriminator. The generator never sees that hinge; it just pushes fake scores up. The two free functions are the training objectives. There is no combined class, because the two steps do not share a cache.",
        "1'in üstündeki gerçek skorlar ve −1'in altındaki sahte skorlar ayırıcıya öğretmeyi bırakır. Üretici o hinge'i hiç görmez; sahte skorları yukarı iter. İki serbest fonksiyon eğitim amaçlarıdır. Birleşik sınıf yoktur çünkü iki adım bir önbellek paylaşmaz.",
      ),
      gradient: bi(
        "Discriminator: −1 on real scores that are below 1, +1 on fake scores that are above −1, then averaged. Generator: −1/N.",
        "Ayırıcı: 1'in altındaki gerçek skorlarda −1, −1'in üstündeki sahte skorlarda +1, sonra ortalama. Üretici: −1/N.",
      ),
      code: `double d = nexusloss::generative::hinge_discriminator_loss<double>(real, fake);
double g = nexusloss::generative::hinge_generator_loss<double>(fake);`,
      params: [
        { name: "scores", detail: bi("Unbounded critic outputs.", "Sınırlanmamış eleştirmen çıktıları.") },
      ],
    },
    {
      id: "elbo",
      name: "VAE ELBO",
      api: "generative::VAEELBOLoss<T>",
      formula: "L = recon + β · ½ Σ (μ² + exp(log σ²) − log σ² − 1)",
      kind: bi("Latent · KL to N(0, I)", "Gizli · N(0, I)'ye KL"),
      when: bi(
        "A variational autoencoder whose encoder emits a mean and a log-variance per latent dimension. The reconstruction term is a scalar you computed with MSE, BCE or whatever the likelihood is.",
        "Kodlayıcısı gizli boyut başına bir ortalama ve bir log-varyans üreten varyasyonel otomatik kodlayıcı. Rekonstrüksiyon terimi, MSE, BCE ya da olabilirlik neyse onunla hesapladığın bir skalerdir.",
      ),
      logic: bi(
        "The ELBO is reconstruction plus the KL from the encoder's diagonal Gaussian to the standard normal prior. β = 1 is the true bound. β > 1 is a β-VAE: you spend more of the loss on a factorized latent and usually get worse pixels and cleaner axes. The class differentiates only the KL. Add your reconstruction gradient in the decoder yourself.",
        "ELBO, rekonstrüksiyon artı kodlayıcının köşegen Gauss'undan standart normal önsele olan KL'dir. β = 1 gerçek sınırdır. β > 1 bir β-VAE'dir: kaybın daha fazlasını çarpanlara ayrılmış bir gizlide harcarsın, genellikle daha kötü pikseller ve daha temiz eksenler alırsın. Sınıf yalnızca KL'yi türevler. Rekonstrüksiyon gradyanını kod çözücüde sen eklersin.",
      ),
      gradient: bi(
        "backward() returns ∂KL/∂μ followed by ∂KL/∂logσ², each scaled by β. Length is 2 × latent.",
        "backward() önce ∂KL/∂μ, sonra ∂KL/∂logσ² döndürür; her biri β ile ölçeklenir. Uzunluk 2 × gizlidir.",
      ),
      code: `std::vector<double> mean{0.2, -0.4}, log_var{-0.1, 0.3};
nexusloss::generative::VAEELBOLoss<double> loss(1.0); // beta
double value = loss.forward(/*reconstruction*/ 0.8, mean, log_var);
auto gradient = loss.backward();`,
      params: [
        { name: "beta", detail: bi("Default 1. KL weight.", "Varsayılan 1. KL ağırlığı.") },
        { name: "reconstruction", detail: bi("A scalar you supply. Not differentiated here.", "Senin verdiğin bir skaler. Burada türetilmez.") },
      ],
    },
    {
      id: "feature-matching",
      name: "Feature matching",
      api: "generative::feature_matching_loss",
      formula: "mean |real features − fake features|",
      kind: bi("Free function · L1", "Serbest fonksiyon · L1"),
      when: bi(
        "GAN stabilization. Match the discriminator's intermediate statistics instead of only its final bit, so the generator gets a denser learning signal.",
        "GAN kararlılığı. Üretici daha yoğun bir öğrenme sinyali alsın diye ayırıcının yalnızca son bitini değil, ara istatistiklerini eşle.",
      ),
      logic: bi(
        "Salimans et al. ask the generator to match the expected features of real data. You extract both tensors and pass them here. The loss is an L1 mean, which is robust to a few wild activations. It does not know which layer you picked.",
        "Salimans ve arkadaşları üreticiden gerçek verinin beklenen özelliklerini eşlemesini ister. İki tensörü çıkarıp buraya verirsin. Kayıp bir L1 ortalamasıdır; birkaç vahşi aktivasyona karşı sağlamdır. Hangi katmanı seçtiğini bilmez.",
      ),
      gradient: bi(
        "∂L/∂fake = sign(fake − real) / N. Real features are targets.",
        "∂L/∂sahte = sign(sahte − gerçek) / N. Gerçek özellikler hedeftir.",
      ),
      code: `double value = nexusloss::generative::feature_matching_loss<double>(real_features, fake_features);`,
      params: [
        { name: "features", detail: bi("Equal length. You choose the layer.", "Eşit uzunluk. Katmanı sen seçersin.") },
      ],
    },
    {
      id: "cycle",
      name: "Cycle consistency",
      api: "generative::cycle_consistency_loss",
      formula: "mean |F(G(x)) − x|",
      kind: bi("Free function · unpaired translation", "Serbest fonksiyon · eşlenmemiş çeviri"),
      when: bi(
        "CycleGAN and any unpaired translator. After a round trip the image should come home, because nothing else ties the two domains together.",
        "CycleGAN ve her eşlenmemiş çevirmen. Gidiş-dönüşten sonra görüntü eve dönmelidir; iki alanı birbirine bağlayan başka bir şey yoktur.",
      ),
      logic: bi(
        "An L1 between the reconstruction and the original. L1 keeps color outliers from dominating, which is why the CycleGAN paper prefers it to L2. Identity loss is the same distance with a different story: G(y) should stay y when y is already in the target domain.",
        "Rekonstrüksiyon ile özgün arasındaki bir L1. L1, renk aykırı değerlerinin hükmetmesini engeller; CycleGAN makalesinin onu L2'ye tercih etmesinin nedeni budur. Kimlik kaybı aynı mesafe, farklı hikâyedir: y zaten hedef alandaysa G(y), y olarak kalmalıdır.",
      ),
      gradient: bi(
        "Sign of the residual, averaged. Apply it to whichever tensor you treat as the prediction.",
        "Artığın işareti, ortalanmış. Tahmin saydığın tensöre uygula.",
      ),
      code: `double cycle = nexusloss::generative::cycle_consistency_loss<double>(reconstructed, original);
double identity = nexusloss::generative::identity_loss<double>(identity_output, input);`,
      params: [
        { name: "tensors", detail: bi("Flattened images or features, equal length.", "Düzleştirilmiş görüntüler veya özellikler, eşit uzunluk.") },
      ],
    },
    {
      id: "perceptual",
      name: "Perceptual",
      api: "generative::perceptual_loss",
      formula: "mean (φ(x) − φ(y))²",
      kind: bi("Free function · features in", "Serbest fonksiyon · özellik girer"),
      when: bi(
        "Super-resolution and synthesis, when pixel MSE looks blurry and you would rather match activations of a network trained on recognition.",
        "Piksel MSE bulanık göründüğünde ve tanıma için eğitilmiş bir ağın aktivasyonlarını eşlemeyi tercih ettiğinde; süper çözünürlük ve sentez.",
      ),
      logic: bi(
        "Johnson et al. measure L2 in feature space. NexusLoss does not own φ. You run the frozen extractor, flatten the maps, and pass the two vectors. The loss is mean squared error, so a few channels with large activations dominate unless you scaled them.",
        "Johnson ve arkadaşları özellik uzayında L2 ölçer. NexusLoss φ'ye sahip değildir. Donmuş çıkarıcıyı çalıştırır, haritaları düzleştirir ve iki vektörü verirsin. Kayıp ortalama karesel hatadır; ölçeklemediysen büyük aktivasyonlu birkaç kanal hükmeder.",
      ),
      gradient: bi(
        "2 (φ_pred − φ_target) / N, in feature space. Scatter it back through φ only if you choose to.",
        "Özellik uzayında 2 (φ_tahmin − φ_hedef) / N. Yalnızca istersen φ üzerinden geri saç.",
      ),
      code: `double value = nexusloss::generative::perceptual_loss<double>(predicted_features, target_features);`,
      params: [
        { name: "features", detail: bi("Already extracted. Equal length.", "Çoktan çıkarılmış. Eşit uzunluk.") },
      ],
    },
    {
      id: "style",
      name: "Style",
      api: "generative::style_loss",
      formula: "mean squared difference of Gram matrices",
      kind: bi("Free function · correlations", "Serbest fonksiyon · korelasyonlar"),
      when: bi(
        "Neural style transfer. Match textures (which channel fires with which) and ignore where they fired.",
        "Sinirsel stil aktarımı. Dokuları eşle (hangi kanal hangisiyle birlikte ateşliyor) ve nerede ateşlediklerini yok say.",
      ),
      logic: bi(
        "Gatys et al. build a Gram matrix, the uncentered covariance of channels across positions, and take MSE between the predicted and target Grams. You pass a flattened feature map plus its channel and position counts. The loss is invariant to permuting spatial locations, which is what 'style' means here.",
        "Gatys ve arkadaşları bir Gram matrisi kurar — kanalların konumlar boyunca merkezlenmemiş kovaryansı — ve tahmin ile hedef Gram'lar arasında MSE alır. Düzleştirilmiş bir özellik haritası ile kanal ve konum sayılarını verirsin. Kayıp uzamsal konumların permütasyonuna değişmezdir; buradaki 'stil' bu demektir.",
      ),
      gradient: bi(
        "Scalar. The Gram difference back-propagates to each spatial site of each channel, proportional to the other channel's activation.",
        "Skaler. Gram farkı her kanalın her uzamsal konumuna, diğer kanalın aktivasyonuyla orantılı olarak geri yayılır.",
      ),
      code: `double value = nexusloss::generative::style_loss<double>(
    predicted, target, /*channels*/ 64, /*positions*/ 32 * 32);`,
      params: [
        { name: "layout", detail: bi("Tensor length must be channels × positions.", "Tensör uzunluğu kanal × konum olmalıdır.") },
      ],
    },
    {
      id: "total-variation",
      name: "Total variation",
      api: "generative::total_variation_loss",
      formula: "mean |neighbor differences| over H and W",
      kind: bi("Free function · smoothness", "Serbest fonksiyon · pürüzsüzlük"),
      when: bi(
        "A cheap prior against speckle: style transfer, inpainting, decoding. Neighboring pixels should not flicker unless the content says so.",
        "Beneklenmeye karşı ucuz bir önsel: stil aktarımı, boyama, kod çözme. İçerik söylemedikçe komşu pikseller titreşmemelidir.",
      ),
      logic: bi(
        "Anisotropic TV sums absolute differences to the right and to the below. The last row and last column have no forward neighbor and are skipped. Channels are independent. It is not a comparison with a target; the image is its own regularizer.",
        "Yönlü TV, sağa ve aşağıya mutlak farkları toplar. Son satır ve son sütunun ileri komşusu yoktur, atlanır. Kanallar bağımsızdır. Bir hedefle karşılaştırma değildir; görüntü kendi düzenleyicisidir.",
      ),
      gradient: bi(
        "Each interior pixel receives sign contributions from the edges it touches. A flat region has gradient 0.",
        "Her iç piksel, değdiği kenarlardan işaret katkıları alır. Düz bir bölgenin gradyanı 0'dır.",
      ),
      code: `double value = nexusloss::generative::total_variation_loss<double>(
    image, /*height*/ 32, /*width*/ 32, /*channels*/ 3);`,
      params: [
        { name: "layout", detail: bi("Row-major, channel-last: index (y·W + x)·C + c. Length = H·W·C.", "Satır-major, kanal sonda: indeks (y·W + x)·C + c. Uzunluk = H·W·C.") },
      ],
    },
    {
      id: "diffusion",
      name: "Diffusion epsilon",
      api: "generative::diffusion_epsilon_loss",
      formula: "mean ‖ε_θ(x_t, t) − ε‖²",
      kind: bi("Free function · noise", "Serbest fonksiyon · gürültü"),
      when: bi(
        "DDPM training. The network predicts the noise that was added at this timestep, and the loss is MSE against that noise. The timestep embedding is the network's problem, not the loss's.",
        "DDPM eğitimi. Ağ, bu zaman adımında eklenen gürültüyü tahmin eder; kayıp o gürültüye karşı MSE'dir. Zaman gömüsü ağın sorunudur, kaybın değil.",
      ),
      logic: bi(
        "Ho et al. show that predicting ε is a reweighted variational bound on the reverse process. NexusLoss takes the two vectors and returns their mean squared error. Weighting by timestep, predicting x0 or v instead of ε, is a choice you make before calling this.",
        "Ho ve arkadaşları ε tahmin etmenin, ters süreç üzerinde yeniden ağırlıklandırılmış bir varyasyonel sınır olduğunu gösterir. NexusLoss iki vektörü alır ve ortalama karesel hatalarını döndürür. Zaman adımına göre ağırlıklandırmak, ε yerine x0 veya v tahmin etmek, bunu çağırmadan önce verdiğin bir karardır.",
      ),
      gradient: bi(
        "∂L/∂ε_θ = 2 (ε_θ − ε) / N. The true noise is constant.",
        "∂L/∂ε_θ = 2 (ε_θ − ε) / N. Gerçek gürültü sabittir.",
      ),
      code: `double value = nexusloss::generative::diffusion_epsilon_loss<double>(predicted_noise, noise);`,
      params: [
        { name: "noise", detail: bi("Same shape as the prediction, flattened.", "Tahminle aynı şekil, düzleştirilmiş.") },
      ],
    },
  ],
};

export const selfSupervised: CategoryDoc = {
  id: "self-supervised",
  index: "08",
  title: bi("Self-supervised", "Öz-denetimli"),
  lede: bi(
    "Two views of the same example, and sometimes a batch of other views playing the role of negatives. Similarities are numbers you computed. VICReg is the exception: it wants the raw projected embeddings and the batch layout.",
    "Aynı örneğin iki görünümü ve bazen negatif rolü oynayan başka görünümlerden bir batch. Benzerlikler senin hesapladığın sayılardır. İstisna VICReg'dir: ham izdüşüm gömülerini ve batch düzenini ister.",
  ),
  losses: [
    {
      id: "infonce",
      name: "InfoNCE",
      api: "self_supervised::InfoNCELoss<T>",
      formula: "−log exp(s⁺/τ) / (exp(s⁺/τ) + Σ exp(s⁻/τ))",
      kind: bi("Similarities · temperature", "Benzerlikler · sıcaklık"),
      when: bi(
        "Contrastive representation learning: one positive similarity and a list of negatives. SimCLR, MoCo and CLIP are this loss with different ways of building the list.",
        "Kontrastif temsil öğrenme: bir pozitif benzerlik ve bir negatif listesi. SimCLR, MoCo ve CLIP, listeyi kurma biçimleri farklı olan bu kayıptır.",
      ),
      logic: bi(
        "van den Oord et al. treat the positive as the correct class in a softmax over the batch. Temperature τ sharpens or softens that softmax. A small τ makes the loss focus on the hardest negative, and it also makes the gradient spiky, which is why 0.1 is a common and slightly dangerous default. The implementation is log-sum-exp stable.",
        "van den Oord ve arkadaşları pozitifi, batch üzerindeki bir softmax'ın doğru sınıfı sayar. Sıcaklık τ o softmax'ı keskinleştirir veya yumuşatır. Küçük τ kaybı en zor negatife odaklar ve gradyanı sivrileştirir; 0.1'in yaygın ve biraz tehlikeli bir varsayılan olmasının nedeni budur. Uygulama log-sum-exp kararlıdır.",
      ),
      gradient: bi(
        "backward() returns the positive slot first, then one entry per negative: softmax weight minus 1 on the positive, softmax weight on each negative, all divided by τ.",
        "backward() önce pozitif yuvayı, sonra negatif başına bir girdi döndürür: pozitifte softmax ağırlığı eksi 1, her negatifte softmax ağırlığı, hepsi τ'ya bölünür.",
      ),
      code: `nexusloss::self_supervised::InfoNCELoss<double> loss(0.1);
double value = loss.forward(/*positive similarity*/ 0.8, negatives);
auto gradient = loss.backward();`,
      params: [
        { name: "temperature", detail: bi("Default 0.1. Applied before the exp.", "Varsayılan 0.1. exp'ten önce uygulanır.") },
      ],
    },
    {
      id: "nt-xent",
      name: "NT-Xent",
      api: "self_supervised::NTXentLoss<T>",
      formula: "InfoNCE on normalized temperature-scaled dots",
      kind: bi("Subclass of InfoNCE", "InfoNCE alt sınıfı"),
      when: bi(
        "SimCLR's name for the same objective. Use it when the call site should say NT-Xent and the similarities are already cosine / τ in spirit.",
        "Aynı amacın SimCLR adıdır. Çağrı yeri NT-Xent demeliyse ve benzerlikler ruhen çoktan kosinüs / τ ise bunu kullan.",
      ),
      logic: bi(
        "NTXentLoss extends InfoNCELoss. The math does not change. The paper's 'normalized' part happened when you computed the cosine, and the 'temperature-scaled' part is the τ in the constructor. Passing raw dots and also dividing by τ double-counts the temperature.",
        "NTXentLoss, InfoNCELoss'u genişletir. Matematik değişmez. Makalenin 'normalize' kısmı kosinüsü hesapladığında oldu; 'sıcaklıkla ölçeklenmiş' kısmı kurucudaki τ'dur. Ham iç çarpım verip bir de τ'ya bölmek sıcaklığı iki kez sayar.",
      ),
      gradient: bi(
        "Identical to InfoNCE.",
        "InfoNCE ile aynı.",
      ),
      code: `nexusloss::self_supervised::NTXentLoss<double> loss(0.5);
double value = loss.forward(0.8, negatives);`,
      params: [
        { name: "temperature", detail: bi("Default 0.1, inherited.", "Varsayılan 0.1, miras alınır.") },
      ],
    },
    {
      id: "byol",
      name: "BYOL",
      api: "self_supervised::BYOLLoss<T>",
      formula: "2 − 2 · cos(q, z)",
      kind: bi("Two views · no negatives", "İki görünüm · negatif yok"),
      when: bi(
        "You do not want a negative bank. An online network predicts the target network's projection, and the target is a stop-gradient moving average you maintain elsewhere.",
        "Negatif bankası istemiyorsun. Çevrimiçi ağ, hedef ağın izdüşümünü tahmin eder; hedef, başka yerde tuttuğun dur-gradyanlı bir hareketli ortalamadır.",
      ),
      logic: bi(
        "Grill et al. regress one normalized view onto the other. The loss is a scaled cosine distance, zero when the directions match. Collapse is avoided by the predictor asymmetry and the slow target, not by this formula. Pass the prediction and the stopped target; do not differentiate the target side if you are following the paper.",
        "Grill ve arkadaşları bir normalize görünümü diğerine regresyon eder. Kayıp ölçeklenmiş bir kosinüs mesafesidir; yönler örtüşünce sıfırdır. Çökmeyi bu formül değil, tahminci asimetrisi ve yavaş hedef önler. Tahmini ve durdurulmuş hedefi ver; makaleyi izliyorsan hedef tarafını türevleme.",
      ),
      gradient: bi(
        "backward() differentiates the prediction side of the cosine distance. The target half of a symmetric gradient is your choice to drop.",
        "backward() kosinüs mesafesinin tahmin tarafını türevler. Simetrik bir gradyanın hedef yarısını düşürmek senin seçimindir.",
      ),
      code: `nexusloss::self_supervised::BYOLLoss<double> loss;
double value = loss.forward(prediction, target);
auto gradient = loss.backward();`,
      params: [
        { name: "vectors", detail: bi("Same length. Zero vectors make cosine undefined; normalize upstream.", "Aynı uzunluk. Sıfır vektörler kosinüsü tanımsız kılar; yukarıda normalize et.") },
      ],
    },
    {
      id: "barlow",
      name: "Barlow Twins",
      api: "self_supervised::BarlowTwinsLoss<T>",
      formula: "Σ_i (C_ii − 1)² + λ Σ_{i≠j} C_ij²",
      kind: bi("Cross-correlation matrix", "Çapraz korelasyon matrisi"),
      when: bi(
        "Two batches of projections, when you would rather penalize a redundancy matrix than sample negatives. You compute the cross-correlation; the loss reads that square matrix.",
        "Negatif örneklemek yerine bir artıklık matrisini cezalandırmayı tercih ettiğin iki izdüşüm batch'i. Çapraz korelasyonu sen hesaplarsın; kayıp o kare matrisi okur.",
      ),
      logic: bi(
        "Zbontar et al. want the cross-correlation between two views to look like the identity. Diagonal entries near 1 mean each feature still varies and agrees across views. Off-diagonal entries near 0 mean features are not copies of each other. λ, default 0.005, keeps the much larger number of off-diagonal terms from drowning the diagonal. The matrix is row-major, size dimensions × dimensions.",
        "Zbontar ve arkadaşları iki görünüm arasındaki çapraz korelasyonun birim matrise benzemesini ister. 1'e yakın köşegen, her özelliğin hâlâ değiştiği ve görünümler arasında uyuştuğu anlamına gelir. 0'a yakın köşegen dışı, özelliklerin birbirinin kopyası olmadığı anlamına gelir. Varsayılan λ = 0.005, çok daha fazla sayıdaki köşegen dışı terimin köşegeni boğmasını engeller. Matris satır-major, boyut × boyuttur.",
      ),
      gradient: bi(
        "∂L/∂C_ii = 2(C_ii − 1), ∂L/∂C_ij = 2λ C_ij. Scatter that onto the embeddings with your own correlation backward.",
        "∂L/∂C_ii = 2(C_ii − 1), ∂L/∂C_ij = 2λ C_ij. Bunu kendi korelasyon geri yayılımınla gömülere saç.",
      ),
      code: `nexusloss::self_supervised::BarlowTwinsLoss<double> loss(0.005);
double value = loss.forward(cross_correlation, /*dimensions*/ 128);
auto gradient = loss.backward();`,
      params: [
        { name: "lambda", detail: bi("Off-diagonal weight. Default 0.005.", "Köşegen dışı ağırlık. Varsayılan 0.005.") },
      ],
    },
    {
      id: "vicreg",
      name: "VICReg",
      api: "self_supervised::vicreg_loss",
      formula: "λ inv + μ var + ν cov",
      kind: bi("Free function · batch of views", "Serbest fonksiyon · görünüm batch'i"),
      when: bi(
        "Variance-invariance-covariance regularization when you do not want to form a cross-correlation matrix by hand. Two batches, same samples, same dimension.",
        "Çapraz korelasyon matrisini elle kurmak istemediğin varyans-değişmezlik-kovaryans düzenlemesi. İki batch, aynı örnekler, aynı boyut.",
      ),
      logic: bi(
        "Bardes et al. split the collapse problem into three terms. Invariance is MSE between the two views of each sample. Variance is a hinge that keeps each dimension's standard deviation above γ (default 1), so a dead unit is punished. Covariance penalizes off-diagonal correlations inside each view. Defaults λ = μ = 25, ν = 1 match the paper's emphasis on the first two. view_a is samples × dimensions, row-major.",
        "Bardes ve arkadaşları çökme sorununu üç terime böler. Değişmezlik, her örneğin iki görünümü arasındaki MSE'dir. Varyans, her boyutun standart sapmasını γ'nın (varsayılan 1) üstünde tutan bir hinge'dir; ölü bir birim cezalandırılır. Kovaryans her görünümün içindeki köşegen dışı korelasyonları cezalandırır. Varsayılan λ = μ = 25, ν = 1, makalenin ilk iki terime verdiği ağırlıkla uyumludur. view_a satır-major, örnek × boyuttur.",
      ),
      gradient: bi(
        "Scalar helper. All three terms are differentiable in the embeddings; the variance hinge is flat once a dimension is healthy.",
        "Skaler yardımcı. Üç terim de gömülerde türevlenebilir; bir boyut sağlıklıyken varyans hinge'i düzdür.",
      ),
      code: `double value = nexusloss::self_supervised::vicreg_loss<double>(
    view_a, view_b, /*samples*/ 64, /*dimensions*/ 128,
    /*invariance*/ 25, /*variance*/ 25, /*covariance*/ 1, /*gamma*/ 1);`,
      params: [
        { name: "gamma", detail: bi("Target standard deviation. Default 1.", "Hedef standart sapma. Varsayılan 1.") },
      ],
    },
    {
      id: "dino",
      name: "DINO",
      api: "self_supervised::dino_loss",
      formula: "− Σ p_teacher · log softmax(z_student / τ)",
      kind: bi("Free function · teacher probs", "Serbest fonksiyon · öğretmen olasılıkları"),
      when: bi(
        "Self-distillation with no labels. The teacher probabilities are a centered, sharpened softmax you computed with stop-gradient. The student is still in logits.",
        "Etiketsiz öz-damıtma. Öğretmen olasılıkları, dur-gradyanla hesapladığın merkezlenmiş, keskinleştirilmiş bir softmax'tır. Öğrenci hâlâ logitlerdedir.",
      ),
      logic: bi(
        "Caron et al. match a student distribution to a teacher distribution over prototype logits. The teacher is not differentiated. student_temperature, default 0.1, sharpens the student softmax. Centering the teacher to avoid collapse is outside this function: pass probabilities that already sum to one.",
        "Caron ve arkadaşları bir öğrenci dağılımını, prototip logitleri üzerinde bir öğretmen dağılımına eşler. Öğretmen türetilmez. student_temperature, varsayılan 0.1, öğrenci softmax'ını keskinleştirir. Çökmeyi önlemek için öğretmeni merkezlemek bu fonksiyonun dışındadır: toplamı bir olan olasılıklar ver.",
      ),
      gradient: bi(
        "Scalar. With respect to student logits the gradient is softmax(z/τ) − p_teacher, divided by τ.",
        "Skaler. Öğrenci logitlerine göre gradyan, τ'ya bölünmüş softmax(z/τ) − p_öğretmen'dir.",
      ),
      code: `double value = nexusloss::self_supervised::dino_loss<double>(
    student_logits, teacher_probabilities, /*student temperature*/ 0.1);`,
      params: [
        { name: "teacher_probabilities", detail: bi("Sum to 1. Stop-gradient before you store them.", "Toplamı 1. Saklamadan önce dur-gradyan uygula.") },
      ],
    },
  ],
};
