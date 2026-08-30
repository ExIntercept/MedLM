/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        slate: {
          bg: "#f8fafc",
          border: "#e2e8f0",
        },
        clinical: {
          teal: "#0d9488",
          "teal-dark": "#0f766e",
          "teal-tint": "#f0fdfa",
          ink: "#0f172a",
          muted: "#64748b",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(15, 23, 42, 0.04)",
        elevated: "0 12px 32px rgba(15, 23, 42, 0.08)",
      },
    },
  },
  plugins: [],
}
