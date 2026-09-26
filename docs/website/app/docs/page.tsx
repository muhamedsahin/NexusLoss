"use client";

import { useEffect, useMemo, useState } from "react";
import { SiteNav } from "../../components/SiteNav";
import { categories, lossCount } from "../../lib/docs/catalog";
import { tx, type Bi, type Lang } from "../../lib/docs/types";
import { useLocale } from "../../lib/use-locale";

const ui = {
  navDocs: { en: "Docs", tr: "Doküman" },
  navHome: { en: "Overview", tr: "Genel bakış" },
  navBench: { en: "Benchmarks", tr: "Kıyas" },
  navApi: { en: "Contract", tr: "Sözleşme" },
  start: { en: "Quick start", tr: "Hızlı başlangıç" },
  search: { en: "Search losses", tr: "Loss ara" },
  eyebrow: { en: "REFERENCE", tr: "REFERANS" },
  titleA: { en: "Every loss,", tr: "Her loss," },
  titleB: { en: "with the reason.", tr: "gerekçesiyle." },
  lede: {
    en: "Thirteen families, one header. Each entry says when to reach for the function, what the formula is actually doing, how the gradient behaves, and a snippet that matches the C++20 signatures.",
    tr: "On üç aile, tek başlık dosyası. Her madde fonksiyonu ne zaman seçeceğini, formülün gerçekte ne yaptığını, gradyanın nasıl davrandığını ve C++20 imzalarıyla örtüşen bir parçayı söyler.",
  },
  contractKicker: { en: "00  /  THE CONTRACT", tr: "00  /  SÖZLEŞME" },
  contractTitle: { en: "Forward, then backward.", tr: "Önce forward, sonra backward." },
  contractBody: {
    en: "Element-wise classes inherit LossBase. forward() checks shapes, reduces, and remembers the inputs. backward() reads that memory. Calling it first throws, on purpose. Structured losses — boxes, CTC, point clouds, lists — keep their own forward and backward, because a box is not a vector of independent residuals.",
    tr: "Eleman bazlı sınıflar LossBase'den türer. forward() şekilleri kontrol eder, indirger ve girdileri hatırlar. backward() o belleği okur. Onu önce çağırmak bilinçli olarak hata fırlatır. Yapısal kayıplar — kutular, CTC, nokta bulutları, listeler — kendi forward ve backward'larını tutar; çünkü bir kutu, bağımsız artıkların vektörü değildir.",
  },
  reductionTitle: { en: "Reduction", tr: "İndirgeme" },
  reductionBody: {
    en: "Mean divides the scalar and the gradient by N. Sum leaves the gradient unscaled, so it is N times larger. None returns every element and does not scale. BatchMean matches Mean on a flat vector; row-wise losses divide by the batch themselves. Weights, when you pass them, use the weight sum as the Mean denominator and multiply Sum or None element by element.",
    tr: "Mean, skaleri ve gradyanı N'ye böler. Sum gradyanı ölçeklemez, bu yüzden N kat büyüktür. None her elemanı döndürür ve ölçeklemez. BatchMean düz bir vektörde Mean ile aynıdır; satır bazlı kayıplar batch'e bölmeyi kendileri yapar. Ağırlık verirsen Mean paydası ağırlık toplamıdır; Sum ve None eleman eleman çarpılır.",
  },
  when: { en: "When to use it", tr: "Ne zaman kullanılır" },
  logic: { en: "What it is doing", tr: "Ne işe yarar" },
  gradient: { en: "The gradient", tr: "Gradyan" },
  params: { en: "Arguments", tr: "Argümanlar" },
  copy: { en: "Copy", tr: "Kopyala" },
  copied: { en: "Copied", tr: "Kopyalandı" },
  empty: { en: "No loss matches that search.", tr: "Bu aramaya uyan loss yok." },
  families: { en: "families", tr: "aile" },
  functions: { en: "functions", tr: "fonksiyon" },
  choice: { en: "your choice", tr: "senin seçimin" },
  autoTr: { en: "Turkey · automatic", tr: "Türkiye · otomatik" },
  autoEn: { en: "outside Turkey · English", tr: "Türkiye dışı · İngilizce" },
  unknown: { en: "location unknown · English", tr: "konum yok · İngilizce" },
  include: { en: "One include", tr: "Tek include" },
  foot: { en: "Built for people who care about the derivative.", tr: "Türevi önemseyenler için." },
};

function Icon({ name }: { name: "arrow" | "github" | "copy" | "sun" }) {
  const paths = {
    arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
    github: <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3.3-.4 6.7-1.6 6.7-7A5.4 5.4 0 0 0 19.3 4 5 5 0 0 0 19.2.5S18 0 15 2.2a13.7 13.7 0 0 0-6 0C6 0 4.8.5 4.8.5A5 5 0 0 0 4.7 4 5.4 5.4 0 0 0 3.3 7.5c0 5.4 3.4 6.6 6.7 7A4.8 4.8 0 0 0 9 18v4m-4 0c-3-.9-3-4-3-4m14 4v-4" />,
    copy: <><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></>,
    sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
  }[name];
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths}</svg>;
}

function say(text: Bi, lang: Lang) {
  return tx(text, lang);
}

export default function DocsPage() {
  const { lang, country, source, choose } = useLocale();
  const [dark, setDark] = useState(true);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState("contract");
  const [copied, setCopied] = useState("");

  useEffect(() => {
    if (!lang) return;
    const id = window.location.hash.replace("#", "");
    if (!id) return;
    document.getElementById(id)?.scrollIntoView();
  }, [lang]);

  useEffect(() => {
    if (!lang) return;
    const nodes = Array.from(document.querySelectorAll<HTMLElement>("[data-doc]"));
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible?.target.id) setActive(visible.target.id);
      },
      { rootMargin: "-15% 0px -55% 0px", threshold: [0.15, 0.4, 0.8] },
    );
    nodes.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, [lang, query]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!lang) return [];
    return categories
      .map((category) => ({
        ...category,
        losses: category.losses.filter((loss) => {
          if (!needle) return true;
          const blob = `${loss.name} ${loss.api} ${loss.formula} ${loss.when[lang]} ${loss.logic[lang]}`.toLowerCase();
          return blob.includes(needle);
        }),
      }))
      .filter((category) => category.losses.length > 0);
  }, [lang, query]);

  const copy = async (id: string, code: string) => {
    await navigator.clipboard?.writeText(code);
    setCopied(id);
    window.setTimeout(() => setCopied((current) => (current === id ? "" : current)), 1400);
  };

  if (!lang) {
    return (
      <div className="loader">
        <div className="loader-mark"><span>N</span></div>
        <div className="loader-word">NEXUS<span>LOSS</span></div>
        <div className="loader-line"><i /></div>
        <small>resolving locale</small>
      </div>
    );
  }

  const place =
    source === "choice"
      ? ui.choice[lang]
      : country === "TR"
        ? ui.autoTr[lang]
        : country
          ? ui.autoEn[lang]
          : ui.unknown[lang];

  return (
    <main className={dark ? "site dark docs-root" : "site docs-root"}>
      <div className="noise" />
      <SiteNav lang={lang} onLang={choose} onTheme={() => setDark(!dark)} current="docs" wide />

      <div className="docs-frame">
        <aside className="docs-side">
          <label className="docs-search"><span>⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={ui.search[lang]} /></label>
          <a className={active === "contract" ? "side-link current" : "side-link"} href="#contract"><b>00</b>{lang === "tr" ? "Sözleşme" : "Contract"}</a>
          <a className={active === "reduction" ? "side-link current" : "side-link"} href="#reduction"><b>00</b>{ui.reductionTitle[lang]}</a>
          {filtered.map((category) => (
            <div className="side-group" key={category.id}>
              <a className={active === category.id ? "side-label current" : "side-label"} href={`#${category.id}`}><b>{category.index}</b>{say(category.title, lang)}</a>
              {category.losses.map((loss) => (
                <a key={loss.id} className={active === loss.id ? "side-link current" : "side-link"} href={`#${loss.id}`}>{loss.name}</a>
              ))}
            </div>
          ))}
        </aside>

        <div className="docs-main">
          <header className="docs-hero">
            <div className="eyebrow"><span className="pulse" /> {ui.eyebrow[lang]} · {lossCount} {ui.functions[lang].toUpperCase()}</div>
            <h1>{ui.titleA[lang]}<br /><i>{ui.titleB[lang]}</i></h1>
            <p className="muted">{ui.lede[lang]}</p>
            <div className="docs-meta">
              <span>{lossCount} {ui.functions[lang]}</span>
              <span className="proof-line" />
              <span>{categories.length} {ui.families[lang]}</span>
              <span className="proof-line" />
              <span className="locale-pill">{place}</span>
            </div>
          </header>

          <section id="contract" data-doc className="doc-block">
            <div className="section-label">{ui.contractKicker[lang]}</div>
            <h2>{ui.contractTitle[lang]}</h2>
            <p className="muted">{ui.contractBody[lang]}</p>
            <div className="code-window docs-code">
              <div className="window-bar"><span><i className="dot red" /><i className="dot yellow" /><i className="dot green" /></span><span>train.cpp</span><button onClick={() => copy("contract", contractCode)}><Icon name="copy" /> {copied === "contract" ? ui.copied[lang] : ui.copy[lang]}</button></div>
              <pre><code>{contractCode}</code></pre>
            </div>
            <div className="contract-grid">
              <div><b>01</b><strong>LossBase</strong><small>{lang === "tr" ? "compute, forward, backward, gradient. Mean gradyanı 1/N ile ölçeklenir." : "compute, forward, backward, gradient. Mean scales the gradient by 1/N."}</small></div>
              <div><b>02</b><strong>float / double</strong><small>{lang === "tr" ? "Her sınıf bir şablondur. İki tipi karıştırma; span'ler aynı T olmalıdır." : "Every class is a template. Do not mix the two; spans must share T."}</small></div>
              <div><b>03</b><strong>{ui.include[lang]}</strong><small><code>#include &lt;nexusloss/nexusloss.hpp&gt;</code></small></div>
            </div>
          </section>

          <section id="reduction" data-doc className="doc-block">
            <div className="section-label">00  /  REDUCTION</div>
            <h2>{ui.reductionTitle[lang]}</h2>
            <p className="muted">{ui.reductionBody[lang]}</p>
            <div className="reduction-table">
              {[
                ["Mean", lang === "tr" ? "Ortalama. Varsayılan. Gradyan 1/N." : "Average. Default. Gradient is 1/N."],
                ["Sum", lang === "tr" ? "Toplam. Gradyan ölçeklenmez." : "Total. Gradient is not scaled."],
                ["None", lang === "tr" ? "Eleman bazlı tensör. Özet yok." : "Element-wise tensor. No summary."],
                ["BatchMean", lang === "tr" ? "Düz vektörde Mean ile aynı." : "Same as Mean on a flat vector."],
              ].map(([name, detail]) => (
                <div key={name}><code>{name}</code><span>{detail}</span></div>
              ))}
            </div>
          </section>

          {filtered.length === 0 && <p className="docs-empty">{ui.empty[lang]}</p>}

          {filtered.map((category) => (
            <section key={category.id} id={category.id} data-doc className="doc-chapter">
              <div className="chapter-index">{category.index}</div>
              <h2>{say(category.title, lang)}</h2>
              <p className="muted">{say(category.lede, lang)}</p>
              {category.losses.map((loss) => (
                <article key={loss.id} id={loss.id} data-doc className="loss-card">
                  <div className="loss-kicker"><span>{say(category.title, lang)}</span><span>{say(loss.kind, lang)}</span></div>
                  <h3>{loss.name}</h3>
                  <code className="loss-api">{loss.api}</code>
                  <pre className="loss-formula">{loss.formula}</pre>
                  <div className="loss-split">
                    <div><b>01</b><strong>{ui.when[lang]}</strong><p>{say(loss.when, lang)}</p></div>
                    <div><b>02</b><strong>{ui.logic[lang]}</strong><p>{say(loss.logic, lang)}</p></div>
                  </div>
                  <div className="grad-note"><b>{ui.gradient[lang]}</b><p>{say(loss.gradient, lang)}</p></div>
                  {loss.params.length > 0 && (
                    <div className="param-list">
                      <span className="param-label">{ui.params[lang]}</span>
                      {loss.params.map((param) => (
                        <div key={param.name}><code>{param.name}</code><span>{say(param.detail, lang)}</span></div>
                      ))}
                    </div>
                  )}
                  <div className="code-window docs-code">
                    <div className="window-bar"><span><i className="dot red" /><i className="dot yellow" /><i className="dot green" /></span><span>{loss.id}.cpp</span><button onClick={() => copy(loss.id, loss.code)}><Icon name="copy" /> {copied === loss.id ? ui.copied[lang] : ui.copy[lang]}</button></div>
                    <pre><code>{loss.code}</code></pre>
                  </div>
                </article>
              ))}
            </section>
          ))}

          <footer className="footer docs-foot">
            <a className="brand" href="/"><span className="brand-mark">N</span><span>NEXUS<em>LOSS</em></span></a>
            <span>{ui.foot[lang]}</span>
            <span>MIT · C++20 · 2026</span>
          </footer>
        </div>
      </div>
    </main>
  );
}

const contractCode = `#include <nexusloss/nexusloss.hpp>
#include <vector>

std::vector<double> prediction{2.0, 4.0, 3.0};
std::vector<double> target{1.0, 5.0, 2.5};

nexusloss::HuberLoss<double> loss(1.0);
double value = loss.forward(prediction, target).front();
std::vector<double> gradient = loss.backward();

loss.set_reduction(nexusloss::core::ReductionType::None);
auto each = loss.forward(prediction, target);`;
