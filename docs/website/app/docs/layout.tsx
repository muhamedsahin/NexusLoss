import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "NexusLoss docs — facade and families",
  description:
    "CrossEntropy and MSE write the gradient into the caller buffer and return the mean. The family catalog documents the headers behind that facade.",
};

export default function DocsLayout({ children }: { children: React.ReactNode }) {
  return children;
}
