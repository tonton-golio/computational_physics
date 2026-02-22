import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "katex/dist/katex.min.css";
import "./globals.css";
import { Header } from "@/components/layout/Header";

import { WebVitalsReporter } from "@/components/performance/WebVitalsReporter";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { SuggestionBox } from "@/components/layout/SuggestionBox";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Koala Brain",
  description: "Interactive computational physics education — where precision meets accessibility. Masters-level content on quantum optics, continuum mechanics, and more.",
  keywords: ["computational physics", "quantum optics", "continuum mechanics", "interactive visualizations", "physics education"],
  authors: [{ name: "Anton" }],
  openGraph: {
    title: "Koala Brain",
    description: "Interactive computational physics education — where precision meets accessibility.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <WebVitalsReporter />
        <div className="flex min-h-screen flex-col">
          <Header />
          <main className="flex-1">
            {children}
          </main>

        </div>
        <SuggestionBox />
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
