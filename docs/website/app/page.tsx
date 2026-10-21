"use client";

import { useEffect, useState } from "react";
import { SiteNav } from "../components/SiteNav";
import type { Lang } from "../lib/docs/types";
import { useLocale } from "../lib/use-locale";

const copy = {
  loader: { en: "initializing loss engine", tr: "loss motoru hazırlanıyor" },
  eyebrow: { en: "C++20 · ANALYTIC GRADIENTS · CPU", tr: "C++20 · ANALİTİK GRADYAN · CPU" },
  heroA: { en: "Loss functions", tr: "Loss fonksiyonları" },
  heroB: { en: "without the noise.", tr: "gürültüsüz." },
  heroLead: {
    en: "Nexus calls two types. CrossEntropy and MSE write dL into your buffer and return the mean. The family headers stay underneath.",
    tr: "Nexus iki tipi çağırır. CrossEntropy ve MSE, dL değerini senin tamponuna yazar ve ortalamayı döndürür. Aile başlıkları altta durur.",
  },
  explore: { en: "Explore the docs", tr: "Dokümanı aç" },
  benchmarks: { en: "See the facade", tr: "Cepheyi gör" },
  proofLoss: { en: "catalog entries", tr: "katalog kaydı" },
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
    en: "Link the static library, include the Nexus facade, and call forward_backward. The scalar is the mean. The gradient is already divided by the batch.",
    tr: "Statik kitaplığı bağla, Nexus cephesini ekle ve forward_backward çağır. Skaler ortalamadır. Gradyan batch'e bölünmüş halde tampona yazılır.",
  },
  headerOnly: { en: "Python", tr: "Python" },
  install: { en: "Install", tr: "Kur" },
  installCmake: { en: "Link the target with nexusloss::nexusloss. The facade header lives in the parent include/nexus/loss.", tr: "Hedefi nexusloss::nexusloss ile bağla. Cephe başlığı üst deponun include/nexus/loss yolundadır." },
  installManual: { en: "From the Nexus root, configure with -DNEXUS_BUILD_PYTHON=ON and build the _core target. Put python/src on PYTHONPATH.", tr: "Nexus kökünden -DNEXUS_BUILD_PYTHON=ON ile yapılandır ve _core hedefini derle. python/src yolunu PYTHONPATH'e koy." },
  include: { en: "Include", tr: "Ekle" },
  optimize: { en: "Call", tr: "Çağır" },
  copied: { en: "Copied", tr: "Kopyalandı" },
  copyBtn: { en: "Copy", tr: "Kopyala" },
  captionA: { en: "✓ C++20 · float logits and targets", tr: "✓ C++20 · float logit ve hedef" },
  captionB: { en: "grad is the caller buffer, scaled by 1/N", tr: "grad çağıranın tamponu, 1/N ile ölçekli" },
  libLabel: { en: "02 / THE LIBRARY", tr: "02 / KÜTÜPHANE" },
  libA: { en: "One interface.", tr: "Tek arayüz." },
  libB: { en: "Every objective.", tr: "Her amaç." },
  libLead: {
    en: "The training step uses CrossEntropy and MSE. The cards below are the family headers those calls and the rest of the catalog are built from.",
    tr: "Eğitim adımı CrossEntropy ve MSE kullanır. Alttaki kartlar, bu çağrıların ve kataloğun geri kalanının kurulduğu aile başlıklarıdır.",
  },
  all: { en: "All", tr: "Tümü" },
  search: { en: "Search losses...", tr: "Loss ara..." },
  benchLabel: { en: "03 / FACADE", tr: "03 / CEPHE" },
  benchA: { en: "Mean loss.", tr: "Ortalama kayıp." },
  benchB: { en: "Your buffer.", tr: "Senin tamponun." },
  benchLead: {
    en: "This repository does not publish timings. Both calls return the mean and write the gradient in place.",
    tr: "Bu depo süre yayımlamaz. İki çağrı da ortalamayı döndürür ve gradyanı yerinde yazar.",
  },
  liveRun: { en: "NEXUS CALLS · NO TIMINGS", tr: "NEXUS ÇAĞRILARI · SÜRE YOK" },
  lower: { en: "MEAN REDUCTION", tr: "ORTALAMA İNDİRGEME" },
  faster: { en: "CrossEntropy returns a Python dict. MSE returns a (loss, grad) tuple.", tr: "CrossEntropy Python'da sözlük döndürür. MSE (loss, grad) demeti döndürür." },
  source: { en: "header:", tr: "başlık:" },
  fitLabel: { en: "04 / HOW IT FITS", tr: "04 / NASIL OTURUR" },
  fitA: { en: "Predictable by", tr: "Tasarım gereği" },
  fitB: { en: "design.", tr: "öngörülebilir." },
  fitLead: {
    en: "The facade owns the objective and the analytical gradient. Your model, optimizer and data pipeline stay outside this call.",
    tr: "Cephe amaca ve analitik gradyana sahiptir. Model, eniyileyici ve veri hattı bu çağrının dışında kalır.",
  },
  forward: { en: "Caller buffer", tr: "Çağıran tamponu" },
  forwardBody: {
    en: "Pass grad yourself. forward_backward writes dL there and does not return a new vector.",
    tr: "grad tamponunu sen ver. forward_backward dL değerini oraya yazar, yeni vektör döndürmez.",
  },
  gradient: { en: "Mean", tr: "Ortalama" },
  gradientBody: {
    en: "The scalar is the mean. CrossEntropy divides the buffer by batch. MSE divides 2(x − y) by n.",
    tr: "Skaler ortalamadır. CrossEntropy tamponu batch'e böler. MSE, 2(x − y) değerini n'ye böler.",
  },
  reduction: { en: "Logits", tr: "Logitler" },
  reductionBody: {
    en: "CrossEntropy expects raw logits and a class index per row. Label smoothing must sit in [0, 1).",
    tr: "CrossEntropy ham logit ve satır başına sınıf indeksi bekler. Etiket yumuşatma [0, 1) içinde olmalıdır.",
  },
  apiLabel: { en: "05 / API REFERENCE", tr: "05 / API REFERANSI" },
  apiA: { en: "Designed to be", tr: "Bir bakışta" },
  apiB: { en: "read at a glance.", tr: "okunacak şekilde." },
  apiLead: {
    en: "Start with the two Nexus calls. The family classes underneath still use their own forward and backward.",
    tr: "İki Nexus çağrısıyla başla. Alttaki aile sınıfları kendi forward ve backward imzalarını kullanmaya devam eder.",
  },
  browse: { en: "Browse API reference", tr: "API referansını aç" },
  view: { en: "view signature", tr: "imzayı gör" },
  foot: { en: "Apache-2.0 · C++20", tr: "Apache-2.0 · C++20" },
  attrib: "Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.",
};

const filters = [
  { id: "all", en: "All", tr: "Tümü" },
  { id: "regression", en: "Regression", tr: "Regresyon" },
  { id: "classification", en: "Classification", tr: "Sınıflandırma" },
  { id: "segmentation", en: "Segmentation", tr: "Segmentasyon" },
  { id: "detection", en: "Detection", tr: "Tespit" },
];

const categories = [
  { id: "regression", n: "01", href: "/docs#regression", en: "Regression", tr: "Regresyon", countEn: "11 entries", countTr: "11 kayıt", blurb: "MSE, Huber, Quantile" },
  { id: "classification", n: "02", href: "/docs#classification", en: "Classification", tr: "Sınıflandırma", countEn: "15 entries", countTr: "15 kayıt", blurb: "BCE, CE, Focal" },
  { id: "segmentation", n: "03", href: "/docs#segmentation", en: "Segmentation", tr: "Segmentasyon", countEn: "9 entries", countTr: "9 kayıt", blurb: "Dice, IoU, Tversky" },
  { id: "detection", n: "04", href: "/docs#detection", en: "Detection", tr: "Tespit", countEn: "6 entries", countTr: "6 kayıt", blurb: "IoU, GIoU, CIoU" },
  { id: "metric", n: "05", href: "/docs#metric", en: "Metric learning", tr: "Metrik öğrenme", countEn: "8 entries", countTr: "8 kayıt", blurb: "Triplet, ArcFace" },
  { id: "ranking", n: "06", href: "/docs#ranking", en: "Ranking", tr: "Sıralama", countEn: "8 entries", countTr: "8 kayıt", blurb: "RankNet, ListNet, BPR" },
  { id: "generative", n: "07", href: "/docs#generative", en: "Generative", tr: "Üretken", countEn: "11 entries", countTr: "11 kayıt", blurb: "GAN, ELBO, diffusion" },
  { id: "self-supervised", n: "08", href: "/docs#self-supervised", en: "Self-supervised", tr: "Öz-denetimli", countEn: "6 entries", countTr: "6 kayıt", blurb: "InfoNCE, BYOL, VICReg" },
  { id: "distillation", n: "09", href: "/docs#distillation", en: "Distillation", tr: "Damıtım", countEn: "3 entries", countTr: "3 kayıt", blurb: "KL, feature, attention" },
  { id: "sequence", n: "10", href: "/docs#sequence", en: "Sequences", tr: "Diziler", countEn: "3 entries", countTr: "3 kayıt", blurb: "CTC, MLM, NSP" },
  { id: "point-cloud", n: "11", href: "/docs#point-cloud", en: "Point clouds", tr: "Nokta bulutu", countEn: "2 entries", countTr: "2 kayıt", blurb: "Chamfer, EMD" },
  { id: "reinforcement", n: "12", href: "/docs#reinforcement", en: "Reinforcement", tr: "Pekiştirmeli", countEn: "4 entries", countTr: "4 kayıt", blurb: "PPO, policy, value" },
  { id: "survival", n: "13", href: "/docs#survival", en: "Survival", tr: "Sağkalım", countEn: "2 entries", countTr: "2 kayıt", blurb: "Cox, Weibull" },
];

const apiItems = [
  { n: "01", name: "nexus::loss::CrossEntropy", href: "/docs#contract", en: "Batch logits, class indices, mean loss", tr: "Batch logit, sınıf indeksi, ortalama kayıp" },
  { n: "02", name: "nexus::loss::MSE", href: "/docs#mse", en: "mean((x − y)²), grad = 2(x − y) / n", tr: "mean((x − y)²), grad = 2(x − y) / n" },
  { n: "03", name: "classification::CrossEntropyLoss", href: "/docs#cross-entropy", en: "One row. The facade averages the batch.", tr: "Tek satır. Cephe batch'i ortalar." },
  { n: "04", name: "MSELoss<T>", href: "/docs#mse", en: "Family class with Mean, Sum or None", tr: "Mean, Sum veya None alan aile sınıfı" },
  { n: "05", name: "LossBase<T>", href: "/docs#reduction", en: "Cached forward / backward for element losses", tr: "Eleman kayıplarında önbellekli forward / backward" },
  { n: "06", name: "segmentation::DiceLoss", href: "/docs#dice", en: "Overlap-aware segmentation", tr: "Örtüşmeye duyarlı segmentasyon" },
];

const code = `#include "nexus/loss/losses.hpp"
#include <vector>

const std::vector<float> logits{2.0f, 0.0f, 0.0f, 2.0f};
const std::vector<int> targets{0, 1};
std::vector<float> ce_grad(logits.size());

const float ce = nexus::loss::CrossEntropy(0.0f).forward_backward(
    logits.data(), targets.data(), /*batch*/ 2, /*classes*/ 2, ce_grad.data());

const std::vector<float> prediction{1.0f, 2.0f};
const std::vector<float> target{1.0f, 0.0f};
std::vector<float> mse_grad(prediction.size());
const float mse = nexus::loss::MSE().forward_backward(
    prediction.data(), target.data(), static_cast<int>(prediction.size()), mse_grad.data());`;

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
  const [installTab, setInstallTab] = useState<"cmake" | "python">("cmake");
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
            <a className="text-button" href="#facade">{copy.benchmarks[lang]} <span>↗</span></a>
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
            <div className="signal-value">1/N<span>mean</span></div>
            <svg viewBox="0 0 440 110" preserveAspectRatio="none"><path className="chart-glow" d="M0 85 C35 78 48 34 80 51 S125 84 158 61 S194 5 229 30 S269 93 302 65 S347 44 374 58 S402 92 440 18" /><path className="chart-line" d="M0 85 C35 78 48 34 80 51 S125 84 158 61 S194 5 229 30 S269 93 302 65 S347 44 374 58 S402 92 440 18" /></svg>
            <div className="signal-foot"><span>forward_backward</span><span>CrossEntropy</span></div>
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
              <button className={installTab === "python" ? "active" : ""} onClick={() => setInstallTab("python")}>{copy.headerOnly[lang]}</button>
            </div>
            <div className="install-command"><code>{installTab === "cmake" ? "target_link_libraries(uygulama PRIVATE nexusloss::nexusloss)" : "cmake -S . -B build -DNEXUS_BUILD_PYTHON=ON"}</code><span>↗</span></div>
            <div className="steps">
              <div><b>01</b><span><strong>{copy.install[lang]}</strong><small>{installTab === "cmake" ? copy.installCmake[lang] : copy.installManual[lang]}</small></span></div>
              <div><b>02</b><span><strong>{copy.include[lang]}</strong><small>{installTab === "cmake" ? <code>#include &quot;nexus/loss/losses.hpp&quot;</code> : <code>import nexus</code>}</small></span></div>
              <div><b>03</b><span><strong>{copy.optimize[lang]}</strong><small>{lang === "tr" ? <><code>forward_backward</code> gradyanı tampona yazar ve ortalama kaybı döndürür.</> : <><code>forward_backward</code> writes the gradient and returns the mean loss.</>}</small></span></div>
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

      <section id="facade" className="benchmark-section">
        <div className="shell">
          <div className="section-label">{copy.benchLabel[lang]}</div>
          <div className="bench-heading">
            <div><h2>{copy.benchA[lang]}<br /><i>{copy.benchB[lang]}</i></h2></div>
            <p className="muted">{copy.benchLead[lang]}</p>
          </div>
          <div className="bench-card">
            <div className="bench-meta"><span><i className="live-dot" /> {copy.liveRun[lang]}</span><span>{copy.lower[lang]}</span></div>
            <div className="bars">
              <div className="bar-row"><span>CrossEntropy</span><span>softmax − target, then × 1/batch</span><b>mean</b></div>
              <div className="bar-row"><span>MSE</span><span>2 (x − y) / n</span><b>mean</b></div>
              <div className="bar-row"><span>Python</span><span>CE dict, MSE tuple</span><b>grad</b></div>
            </div>
            <div className="bench-bottom"><span>{copy.faster[lang]}</span><span>{copy.source[lang]} <code>include/nexus/loss/losses.hpp</code></span></div>
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
        <span>v0.1.0</span>
        <p className="footer-attrib">{copy.attrib}</p>
      </footer>
    </main>
  );
}
