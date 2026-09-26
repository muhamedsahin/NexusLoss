"use client";

import type { Lang } from "../lib/docs/types";

const links = [
  { id: "home", href: "/", en: "Overview", tr: "Genel bakış" },
  { id: "docs", href: "/docs", en: "Docs", tr: "Doküman" },
  { id: "benchmarks", href: "/#benchmarks", en: "Benchmarks", tr: "Kıyas" },
  { id: "contract", href: "/docs#contract", en: "Contract", tr: "Sözleşme" },
] as const;

function Icon({ name }: { name: "arrow" | "github" | "sun" }) {
  const paths = {
    arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
    github: <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3.3-.4 6.7-1.6 6.7-7A5.4 5.4 0 0 0 19.3 4 5 5 0 0 0 19.2.5S18 0 15 2.2a13.7 13.7 0 0 0-6 0C6 0 4.8.5 4.8.5A5 5 0 0 0 4.7 4 5.4 5.4 0 0 0 3.3 7.5c0 5.4 3.4 6.6 6.7 7A4.8 4.8 0 0 0 9 18v4m-4 0c-3-.9-3-4-3-4m14 4v-4" />,
    sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
  }[name];
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths}</svg>;
}

export function SiteNav({
  lang,
  onLang,
  onTheme,
  current,
  wide,
}: {
  lang: Lang;
  onLang: (lang: Lang) => void;
  onTheme: () => void;
  current: "home" | "docs";
  wide?: boolean;
}) {
  return (
    <nav className={wide ? "nav shell docs-nav" : "nav shell"}>
      <a className="brand" href="/"><span className="brand-mark">N</span><span>NEXUS<em>LOSS</em></span><b>v0.1</b></a>
      <div className="nav-links">
        {links.map((link) => (
          <a key={link.id} className={link.id === current ? "is-current" : undefined} href={link.href}>
            {link[lang]}
          </a>
        ))}
      </div>
      <div className="nav-actions">
        <div className="lang-switch" role="group" aria-label="Language">
          <button className={lang === "tr" ? "active" : ""} onClick={() => onLang("tr")}>TR</button>
          <button className={lang === "en" ? "active" : ""} onClick={() => onLang("en")}>EN</button>
        </div>
        <button className="icon-button" onClick={onTheme} aria-label="Toggle theme"><Icon name="sun" /></button>
        <a className="github" href="https://github.com" aria-label="GitHub"><Icon name="github" /></a>
        <a className="nav-cta" href="/#quickstart">{lang === "tr" ? "Hızlı başlangıç" : "Quick start"} <Icon name="arrow" /></a>
      </div>
    </nav>
  );
}
