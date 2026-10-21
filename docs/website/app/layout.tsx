import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NexusLoss — Analytic loss and gradients",
  description: "C++20 loss library. Nexus calls CrossEntropy and MSE; both write the gradient into the caller buffer and return the mean.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="tr">
      <body>{children}</body>
    </html>
  );
}
