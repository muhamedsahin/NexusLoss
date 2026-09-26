export type Lang = "en" | "tr";

export type Bi = { en: string; tr: string };

export const bi = (en: string, tr: string): Bi => ({ en, tr });

export const tx = (value: Bi, lang: Lang) => value[lang];

export type LossDoc = {
  id: string;
  name: string;
  api: string;
  formula: string;
  kind: Bi;
  when: Bi;
  logic: Bi;
  gradient: Bi;
  code: string;
  params: { name: string; detail: Bi }[];
};

export type CategoryDoc = {
  id: string;
  index: string;
  title: Bi;
  lede: Bi;
  losses: LossDoc[];
};
