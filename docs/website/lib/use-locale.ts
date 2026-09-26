"use client";

import { useEffect, useState } from "react";
import type { Lang } from "./docs/types";

export function useLocale() {
  const [lang, setLang] = useState<Lang | null>(null);
  const [country, setCountry] = useState("");
  const [source, setSource] = useState("lookup");

  useEffect(() => {
    const stored = window.localStorage.getItem("nexusloss-lang");
    if (stored === "en" || stored === "tr") {
      setLang(stored);
      setSource("choice");
      document.documentElement.lang = stored;
      return;
    }
    let cancelled = false;
    fetch("/api/locale")
      .then((response) => response.json())
      .then((data: { lang?: string; country?: string; source?: string }) => {
        if (cancelled) return;
        const next: Lang = data.lang === "tr" ? "tr" : "en";
        setLang(next);
        setCountry(data.country ?? "");
        setSource(data.source ?? "fallback");
        document.documentElement.lang = next;
      })
      .catch(() => {
        if (!cancelled) {
          setLang("en");
          document.documentElement.lang = "en";
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const choose = (next: Lang) => {
    setLang(next);
    setSource("choice");
    window.localStorage.setItem("nexusloss-lang", next);
    document.documentElement.lang = next;
  };

  return { lang, country, source, choose };
}
