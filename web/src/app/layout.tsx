import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { SmoothScrollProvider } from "@/providers/smooth-scroll";
import { AppShell } from "@/components/layout/app-shell";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "FAST-TRACK AI | Security Operations Command Center",
  description:
    "Next-generation real-time weapon, crime and violence detection platform with edge AI vision.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable} dark`}>
      <body className="bg-navy-900 text-slate-100 antialiased selection:bg-neon-cyan/20 selection:text-neon-cyan">
        {/* Animated Background Mesh & Faint Grid */}
        <div className="bg-mesh-container">
          <div className="mesh-blob-1" />
          <div className="mesh-blob-2" />
          <div className="mesh-blob-3" />
        </div>
        <div className="fixed inset-0 bg-grid-pattern pointer-events-none z-0" />

        <SmoothScrollProvider>
          <AppShell>{children}</AppShell>
        </SmoothScrollProvider>
      </body>
    </html>
  );
}
