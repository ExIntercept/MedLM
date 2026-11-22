/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Theme colours are CSS variables (src/index.css), switched by the
        // "dark" class on <html>.
        surface: "rgb(var(--c-surface) / <alpha-value>)",
        slate: {
          bg: "rgb(var(--c-bg) / <alpha-value>)",
          border: "rgb(var(--c-border) / <alpha-value>)",
        },
        clinical: {
          teal: "rgb(var(--c-teal) / <alpha-value>)",
          "teal-dark": "rgb(var(--c-teal-text) / <alpha-value>)",
          "teal-hover": "rgb(var(--c-teal-hover) / <alpha-value>)",
          "teal-tint": "rgb(var(--c-teal-tint) / <alpha-value>)",
          ink: "rgb(var(--c-ink) / <alpha-value>)",
          muted: "rgb(var(--c-muted) / <alpha-value>)",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(20, 32, 30, 0.04), 0 1px 3px rgba(20, 32, 30, 0.04)",
        elevated: "0 12px 32px rgba(20, 32, 30, 0.10)",
      },
    },
  },
  plugins: [],
}
