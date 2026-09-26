"use client";

import { useEffect, useState } from "react";
import { SiteNav } from "../components/SiteNav";
import type { Lang } from "../lib/docs/types";
import { useLocale } from "../lib/use-locale";

const copy = {
  loader: { en: "initializing loss engine", tr: "loss motoru hazırlanıyor" },
  eyebrow: { en: "C++20 · HEADER-ONLY · CPU FIRST", tr: "C++20 · SADECE BAŞLIK · ÖNCE CPU" },
  heroA: { en: "Loss functions", tr: "Loss fonksiyonları" },
  heroB: { en: "without the noise.", tr: "gürültüsüz." },
  heroLead: {
    en: "Train with intent. NexusLoss gives your models precise, transparent gradients in a single, dependency-free C++ header.",
    tr: "Niyetle eğit. NexusLoss, modellere tek ve bağımlılıksız bir C++ başlığında kesin, şeffaf gradyanlar verir.",
  },
  explore: { en: "Explore the docs", tr: "Dokümanı aç" },
  benchmarks: { en: "See benchmarks", tr: "Kıyasları gör" },
  proofLoss: { en: "loss functions", tr: "loss fonksiyonu" },
  proofDeps: { en: "runtime dependencies", tr: "çalışma zamanı bağımlılığı" },
  live: { en: "LIVE GRADIENT", tr: "CANLI GRADYAN" },
  stable: { en: "stable", tr: "kararlı" },
  ready: { en: "gradient ready", tr: "gradyan hazır" },
  tickA: { en: "BUILT FOR THE LAST MILE", tr: "SON ADIM İÇİN" },
  tickB: { en: "ANALYTICAL GRADIENTS", tr: "ANALİTİK GRADYANLAR" },
  tickC: { en: "ZERO MAGIC", tr: "SIFIR SİHİR" },
  quickLabel: { en: "01 / QUICK START", tr: "01 / HIZLI BAŞLANGIÇ" },
  quickA: { en: "Your first loss", tr: "İlk loss'un" },
  quickB: { en: "in under a minute.", tr: "bir dakikadan kısa." },
  quickLead: {
    en: "No framework lock-in. No build system ceremony. Include one header and start measuring what your model learns. Every loss exposes the same mental model: inputs in, value out, gradient back.",
    tr: "Çerçeveye kilit yok. Derleme töreni yok. Bir başlık ekle ve modelin ne öğrendiğini ölçmeye başla. Her loss aynı zihinsel modeli sunar: girdi girer, değer çıkar, gradyan geri döner.",
  },
  headerOnly: { en: "Header-only", tr: "Sadece başlık" },
  install: { en: "Install", tr: "Kur" },
  installCmake: { en: "Link the target with nexusloss::nexusloss.", tr: "Hedefi nexusloss::nexusloss ile bağla." },
  installManual: { en: "Copy the include directory into your project.", tr: "include klasörünü projenin içine kopyala." },
  include: { en: "Include", tr: "Ekle" },
  optimize: { en: "Optimize", tr: "Optimize et" },
  copied: { en: "Copied", tr: "Kopyalandı" },
  copyBtn: { en: "Copy", tr: "Kopyala" },
  captionA: { en: "✓ C++20 · float or double", tr: "✓ C++20 · float veya double" },
  captionB: { en: "gradient cached after forward()", tr: "gradyan forward() sonrası saklanır" },
  libLabel: { en: "02 / THE LIBRARY", tr: "02 / KÜTÜPHANE" },
  libA: { en: "One interface.", tr: "Tek arayüz." },
  libB: { en: "Every objective.", tr: "Her amaç." },
  libLead: {
    en: "From a two-line regression baseline to contrastive learning at scale. Every function is documented, tested and built to compose with your own training loop.",
    tr: "İki satırlık bir regresyon tabanından ölçekte kontrastif öğrenmeye. Her fonksiyon belgelenmiş, test edilmiş ve kendi eğitim döngünle birleşecek şekilde yazılmış.",
  },
  all: { en: "All", tr: "Tümü" },
  search: { en: "Search losses...", tr: "Loss ara..." },
  benchLabel: { en: "03 / BENCHMARKS", tr: "03 / KIYAS" },
  benchA: { en: "Small binary.", tr: "Küçük ikili." },
  benchB: { en: "Serious throughput.", tr: "Ciddi hız." },
  benchLead: {
    en: "Measured on an Apple M2 Pro · 1M elements · float64 · -O3. The gradient stays close to the data.",
    tr: "Apple M2 Pro · 1M eleman · float64 · -O3 üzerinde ölçüldü. Gradyan verinin yanında kalır.",
  },
  liveRun: { en: "LIVE RUN · 1,000,000 ELEMENTS", tr: "CANLI KOŞU · 1.000.000 ELEMAN" },
  lower: { en: "LOWER IS BETTER", tr: "DÜŞÜK OLAN İYİ" },
  faster: { en: "2.3× faster than the nearest baseline", tr: "en yakın taban çizgisinden 2.3× hızlı" },
  source: { en: "source:", tr: "kaynak:" },
  fitLabel: { en: "04 / HOW IT FITS", tr: "04 / NASIL OTURUR" },
  fitA: { en: "Predictable by", tr: "Tasarım gereği" },
  fitB: { en: "design.", tr: "öngörülebilir." },
  fitLead: {
    en: "NexusLoss stays deliberately small. It owns the objective and its analytical gradient; your model, optimizer and data pipeline remain yours.",
    tr: "NexusLoss bilinçli olarak küçük kalır. Amaca ve onun analitik gradyanına sahiptir; model, optimize edici ve veri hattı senin kalır.",
  },
  forward: { en: "Forward pass", tr: "İleri geçiş" },
  forwardBody: {
    en: "Compute a scalar or element-wise loss from spans. No hidden allocations.",
    tr: "Span'lerden skaler ya da eleman bazlı bir loss hesapla. Gizli ayırma yok.",
  },
  gradient: { en: "Gradient pass", tr: "Gradyan geçişi" },
  gradientBody: {
    en: "Read the cached derivative after forward(). Calling backward() too early fails loudly.",
    tr: "Türevi forward() sonrasında önbellekten oku. backward()'ı erken çağırmak yüksek sesle hata verir.",
  },
  reduction: { en: "Reduction", tr: "İndirgeme" },
  reductionBody: {
    en: "Choose Mean, Sum or None depending on how your batch is assembled.",
    tr: "Batch'in nasıl kurulduğuna göre Mean, Sum veya None seç.",
  },
  apiLabel: { en: "05 / API REFERENCE", tr: "05 / API REFERANSI" },
  apiA: { en: "Designed to be", tr: "Bir bakışta" },
  apiB: { en: "read at a glance.", tr: "okunacak şekilde." },
  apiLead: {
    en: "Consistent names. Predictable shapes. Gradients you can inspect. Start with the shared contract, then move into a focused namespace.",
    tr: "Tutarlı isimler. Öngörülebilir şekiller. İnceleyebileceğin gradyanlar. Ortak sözleşmeyle başla, sonra odaklı bir ad alanına geç.",
  },
  browse: { en: "Browse API reference", tr: "API referansını aç" },
  view: { en: "view signature", tr: "imzayı gör" },
  foot: { en: "Built for people who care about the derivative.", tr: "Türevi önemseyenler için." },
};

const filters = [
  { id: "all", en: "All", tr: "Tümü" },
  { id: "regression", en: "Regression", tr: "Regresyon" },
  { id: "classification", en: "Classification", tr: "Sınıflandırma" },
  { id: "segmentation", en: "Segmentation", tr: "Segmentasyon" },
  { id: "detection", en: "Detection", tr: "Tespit" },
];

const categories = [
  { id: "regression", n: "01", href: "/docs#regression", en: "Regression", tr: "Regresyon", countEn: "12 functions", countTr: "12 fonksiyon", blurb: "MSE, Huber, Quantile" },
  { id: "classification", n: "02", href: "/docs#classification", en: "Classification", tr: "Sınıflandırma", countEn: "18 functions", countTr: "18 fonksiyon", blurb: "BCE, CE, Focal" },
  { id: "segmentation", n: "03", href: "/docs#segmentation", en: "Segmentation", tr: "Segmentasyon", countEn: "8 functions", countTr: "8 fonksiyon", blurb: "Dice, IoU, Tversky" },
  { id: "detection", n: "04", href: "/docs#detection", en: "Detection", tr: "Tespit", countEn: "4 functions", countTr: "4 fonksiyon", blurb: "IoU, GIoU, CIoU" },
  { id: "generative", n: "05", href: "/docs#generative", en: "Generative", tr: "Üretken", countEn: "11 functions", countTr: "11 fonksiyon", blurb: "GAN, ELBO, diffusion" },
  { id: "self-supervised", n: "06", href: "/docs#self-supervised", en: "Self-supervised", tr: "Öz-denetimli", countEn: "5 functions", countTr: "5 fonksiyon", blurb: "InfoNCE, BYOL, VICReg" },
  { id: "ranking", n: "07", href: "/docs#ranking", en: "Ranking", tr: "Sıralama", countEn: "5 functions", countTr: "5 fonksiyon", blurb: "RankNet, ListNet, BPR" },
  { id: "sequence", n: "08", href: "/docs#sequence", en: "Sequences", tr: "Diziler", countEn: "3 functions", countTr: "3 fonksiyon", blurb: "CTC, MLM, NSP" },
  { id: "reinforcement", n: "09", href: "/docs#reinforcement", en: "Reinforcement", tr: "Pekiştirmeli", countEn: "4 functions", countTr: "4 fonksiyon", blurb: "PPO, policy, value" },
];

const apiItems = [
  { n: "01", name: "LossBase<T>", href: "/docs#contract", en: "Common forward / backward contract", tr: "Ortak forward / backward sözleşmesi" },
  { n: "02", name: "reduction::Mean", href: "/docs#reduction", en: "Normalize element-wise losses", tr: "Eleman bazlı loss'ları normalize eder" },
  { n: "03", name: "HuberLoss<T>", href: "/docs#huber", en: "Robust regression objective", tr: "Sağlam regresyon amacı" },
  { n: "04", name: "BCEWithLogitsLoss<T>", href: "/docs#bce-with-logits", en: "Numerically stable binary loss", tr: "Sayısal olarak kararlı ikili loss" },
  { n: "05", name: "classification::CrossEntropyLoss", href: "/docs#cross-entropy", en: "Single-sample class logits", tr: "Tek örnek sınıf logitleri" },
  { n: "06", name: "segmentation::DiceLoss", href: "/docs#dice", en: "Overlap-aware segmentation", tr: "Örtüşmeye duyarlı segmentasyon" },
];

const code = `#include <nexusloss/nexusloss.hpp>
#include <vector>

int main() {
  std::vector<double> prediction{2.0, 4.0, 3.0};
  std::vector<double> target{1.0, 5.0, 2.5};

  nexusloss::HuberLoss<double> loss(1.0);
  auto value = loss.forward(prediction, target).front();
  auto gradient = loss.backward();
}`;

function Icon({ name }: { name: "arrow" | "copy" }) {
  const paths = {
    arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
    copy: <><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></>,
  }[name];
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths}</svg>;
}

export default function Home() {
  const { lang, choose } = useLocale();
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const [query, setQuery] = useState("");
  const [dark, setDark] = useState(true);
  const [installTab, setInstallTab] = useState<"cmake" | "manual">("cmake");
  const [activeCategory, setActiveCategory] = useState("all");

  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 1550);
    return () => clearTimeout(timer);
  }, []);

  const copyCode = async () => {
    await navigator.clipboard?.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  if (loading || !lang) {
    return (
      <div className="loader">
        <div className="loader-mark"><span>N</span></div>
        <div className="loader-word">NEXUS<span>LOSS</span></div>
        <div className="loader-line"><i /></div>
        <small>{lang ? copy.loader[lang] : "NEXUSLOSS"}</small>
      </div>
    );
  }

  const visibleCategories = categories.filter((item) => {
    const matchesFilter = activeCategory === "all" || item.id === activeCategory;
    const blob = `${item.en} ${item.tr} ${item.blurb}`.toLowerCase();
    return matchesFilter && blob.includes(query.toLowerCase());
  });

  return (
    <main className={dark ? "site dark" : "site"}>
      <div className="noise" />
      <SiteNav lang={lang} onLang={choose} onTheme={() => setDark(!dark)} current="home" />

      <section className="hero shell">
        <div className="hero-copy">
          <div className="eyebrow"><span className="pulse" /> {copy.eyebrow[lang]}</div>
          <h1>{copy.heroA[lang]}<br /><i>{copy.heroB[lang]}</i></h1>
          <p>{copy.heroLead[lang]}</p>
          <div className="hero-buttons">
            <a className="primary-button" href="/docs">{copy.explore[lang]} <Icon name="arrow" /></a>
            <a className="text-button" href="#benchmarks">{copy.benchmarks[lang]} <span>↗</span></a>
          </div>
          <div className="hero-proof">
            <span>★ <strong>88</strong> {copy.proofLoss[lang]}</span>
            <span className="proof-line" />
            <span><strong>0</strong> {copy.proofDeps[lang]}</span>
          </div>
        </div>
        <div className="hero-visual">
          <div className="orb orb-one" /><div className="orb orb-two" /><div className="grid-plane" />
          <div className="signal-card">
            <div className="signal-top"><span>{copy.live[lang]}</span><span className="signal-dot">● {copy.stable[lang]}</span></div>
            <div className="signal-value">0.284<span>↓ 18.4%</span></div>
            <svg viewBox="0 0 440 110" preserveAspectRatio="none"><path className="chart-glow" d="M0 85 C35 78 48 34 80 51 S125 84 158 61 S194 5 229 30 S269 93 302 65 S347 44 374 58 S402 92 440 18" /><path className="chart-line" d="M0 85 C35 78 48 34 80 51 S125 84 158 61 S194 5 229 30 S269 93 302 65 S347 44 374 58 S402 92 440 18" /></svg>
            <div className="signal-foot"><span>epoch 042</span><span>hubER_loss&lt;double&gt;</span></div>
          </div>
          <div className="float-chip chip-a">∂L/∂x <strong>+0.42</strong></div>
          <div className="float-chip chip-b"><span>●</span> {copy.ready[lang]}</div>
        </div>
      </section>

      <section className="ticker"><div className="ticker-track"><span>{copy.tickA[lang]}</span><b>✦</b><span>{copy.tickB[lang]}</span><b>✦</b><span>{copy.tickC[lang]}</span><b>✦</b><span>{copy.tickA[lang]}</span><b>✦</b><span>{copy.tickB[lang]}</span></div></section>

      <section id="quickstart" className="quickstart shell">
        <div className="section-label">{copy.quickLabel[lang]}</div>
        <div className="quick-grid">
          <div>
            <h2>{copy.quickA[lang]}<br /><i>{copy.quickB[lang]}</i></h2>
            <p className="muted">{copy.quickLead[lang]}</p>
            <div className="install-tabs">
              <button className={installTab === "cmake" ? "active" : ""} onClick={() => setInstallTab("cmake")}>CMake</button>
              <button className={installTab === "manual" ? "active" : ""} onClick={() => setInstallTab("manual")}>{copy.headerOnly[lang]}</button>
            </div>
            <div className="install-command"><code>{installTab === "cmake" ? "add_subdirectory(NexusLoss)" : "cp -r include/nexusloss ./include"}</code><span>↗</span></div>
            <div className="steps">
              <div><b>01</b><span><strong>{copy.install[lang]}</strong><small>{installTab === "cmake" ? copy.installCmake[lang] : copy.installManual[lang]}</small></span></div>
              <div><b>02</b><span><strong>{copy.include[lang]}</strong><small><code>#include &lt;nexusloss/nexusloss.hpp&gt;</code></small></span></div>
              <div><b>03</b><span><strong>{copy.optimize[lang]}</strong><small>{lang === "tr" ? <>Önce <code>forward()</code>, sonra <code>backward()</code> çağır.</> : <>Call <code>forward()</code>, then <code>backward()</code>.</>}</small></span></div>
            </div>
          </div>
          <div className="code-window">
            <div className="window-bar"><span><i className="dot red" /><i className="dot yellow" /><i className="dot green" /></span><span>main.cpp</span><button onClick={copyCode}><Icon name="copy" /> {copied ? copy.copied[lang] : copy.copyBtn[lang]}</button></div>
            <pre><code>{code}</code></pre>
            <div className="code-caption"><span>{copy.captionA[lang]}</span><span>{copy.captionB[lang]}</span></div>
          </div>
        </div>
      </section>

      <section id="docs" className="docs-section shell">
        <div className="section-label">{copy.libLabel[lang]}</div>
        <div className="docs-heading">
          <div><h2>{copy.libA[lang]}<br /><i>{copy.libB[lang]}</i></h2></div>
          <p className="muted">{copy.libLead[lang]}</p>
        </div>
        <div className="docs-toolbar">
          <div className="category-filters">
            {filters.map((item) => <button key={item.id} className={activeCategory === item.id ? "active" : ""} onClick={() => setActiveCategory(item.id)}>{item[lang]}</button>)}
          </div>
          <label className="search-box"><span>⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={copy.search[lang]} /></label>
        </div>
        <div className="category-grid">
          {visibleCategories.map((item) => (
            <a href={item.href} className="category-card" key={item.id}>
              <span>{item.n}</span>
              <strong>{item[lang]}</strong>
              <small>{item.blurb}</small>
              <small>{lang === "tr" ? item.countTr : item.countEn} <Icon name="arrow" /></small>
            </a>
          ))}
        </div>
      </section>

      <section id="benchmarks" className="benchmark-section">
        <div className="shell">
          <div className="section-label">{copy.benchLabel[lang]}</div>
          <div className="bench-heading">
            <div><h2>{copy.benchA[lang]}<br /><i>{copy.benchB[lang]}</i></h2></div>
            <p className="muted">{copy.benchLead[lang]}</p>
          </div>
          <div className="bench-card">
            <div className="bench-meta"><span><i className="live-dot" /> {copy.liveRun[lang]}</span><span>{copy.lower[lang]}</span></div>
            <div className="bars">
              <div className="bar-row"><span>NexusLoss</span><div className="bar-track"><i className="bar-nexus" /></div><b>4.8 ms</b></div>
              <div className="bar-row"><span>PyTorch / CPU</span><div className="bar-track"><i className="bar-pytorch" /></div><b>11.2 ms</b></div>
              <div className="bar-row"><span>TensorFlow / CPU</span><div className="bar-track"><i className="bar-tf" /></div><b>14.7 ms</b></div>
            </div>
            <div className="bench-bottom"><span>{copy.faster[lang]}</span><span>{copy.source[lang]} <code>benchmarks/</code> ↗</span></div>
          </div>
        </div>
      </section>

      <section id="patterns" className="patterns-section shell">
        <div className="section-label">{copy.fitLabel[lang]}</div>
        <div className="pattern-grid">
          <div>
            <h2>{copy.fitA[lang]}<br /><i>{copy.fitB[lang]}</i></h2>
            <p className="muted">{copy.fitLead[lang]}</p>
          </div>
          <div className="pattern-cards">
            <div><span>01</span><strong>{copy.forward[lang]}</strong><small>{copy.forwardBody[lang]}</small></div>
            <div><span>02</span><strong>{copy.gradient[lang]}</strong><small>{copy.gradientBody[lang]}</small></div>
            <div><span>03</span><strong>{copy.reduction[lang]}</strong><small>{copy.reductionBody[lang]}</small></div>
          </div>
        </div>
      </section>

      <section id="api" className="api-section shell">
        <div className="api-copy">
          <div className="section-label">{copy.apiLabel[lang]}</div>
          <h2>{copy.apiA[lang]}<br /><i>{copy.apiB[lang]}</i></h2>
          <p className="muted">{copy.apiLead[lang]}</p>
          <a className="primary-button" href="/docs">{copy.browse[lang]} <Icon name="arrow" /></a>
        </div>
        <div className="api-list">
          {apiItems.map((item) => (
            <a href={item.href} key={item.name}>
              <span>{item.n}</span>
              <strong>{item.name}<small>{item[lang]}</small></strong>
              <small>{copy.view[lang]} <Icon name="arrow" /></small>
            </a>
          ))}
        </div>
      </section>
      <footer className="footer shell">
        <a className="brand" href="/"><span className="brand-mark">N</span><span>NEXUS<em>LOSS</em></span></a>
        <span>{copy.foot[lang]}</span>
        <span>MIT · C++20 · 2026</span>
      </footer>
    </main>
  );
}
