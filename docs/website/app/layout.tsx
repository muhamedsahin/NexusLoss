import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NexusLoss — Loss functions, engineered",
  description: "C++20 header-only loss library documentation and benchmarks.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="tr">
      <body>{children}</body>
    </html>
  );
}
