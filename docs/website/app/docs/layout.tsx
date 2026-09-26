import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "NexusLoss docs — every loss, explained",
  description:
    "Classified reference for every NexusLoss objective: what it measures, why the gradient looks that way, and a copy-ready C++20 example.",
};

export default function DocsLayout({ children }: { children: React.ReactNode }) {
  return children;
}
