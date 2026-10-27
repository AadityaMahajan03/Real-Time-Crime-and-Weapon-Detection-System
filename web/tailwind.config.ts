import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        background: "#070B14",
        foreground: "#F8FAFC",
        navy: {
          950: "#04070D",
          900: "#070B14",
          850: "#0A101D",
          800: "#0E1726",
          700: "#152238",
          600: "#1E304F",
        },
        surface: {
          DEFAULT: "rgba(255, 255, 255, 0.03)",
          hover: "rgba(255, 255, 255, 0.06)",
          active: "rgba(255, 255, 255, 0.09)",
        },
        neon: {
          cyan: "#22D3EE",
          green: "#4ADE80",
          amber: "#FBBF24",
          red: "#F43F5E",
          violet: "#A78BFA",
        },
        border: "rgba(255, 255, 255, 0.10)",
        ring: "#22D3EE",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "Inter", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "JetBrains Mono", "monospace"],
      },
      borderRadius: {
        "2xl": "1rem",
        "3xl": "1.5rem",
      },
      boxShadow: {
        glass: "0 8px 32px 0 rgba(0, 0, 0, 0.37)",
        "glow-cyan": "0 0 25px -4px rgba(34, 211, 238, 0.35)",
        "glow-green": "0 0 25px -4px rgba(74, 222, 128, 0.35)",
        "glow-amber": "0 0 25px -4px rgba(251, 191, 36, 0.35)",
        "glow-red": "0 0 25px -4px rgba(244, 63, 94, 0.40)",
        "glow-violet": "0 0 25px -4px rgba(167, 139, 250, 0.35)",
      },
      animation: {
        "pulse-subtle": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "mesh-shift": "meshShift 20s ease-in-out infinite alternate",
        "radar-sweep": "radarSweep 4s linear infinite",
      },
      keyframes: {
        meshShift: {
          "0%": { transform: "translate(0%, 0%) scale(1)" },
          "50%": { transform: "translate(4%, 3%) scale(1.05)" },
          "100%": { transform: "translate(-3%, -2%) scale(1)" },
        },
        radarSweep: {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
