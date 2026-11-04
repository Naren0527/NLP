/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg: {
          main: '#050505',
          panel: '#0A0A0A',
        },
        surface: {
          DEFAULT: '#111111',
          hover: '#1A1A1A',
          selected: '#202020',
          panel: '#0A0A0A',
        },
        text: {
          primary: '#F5F5F5',
          secondary: '#A1A1A1',
          muted: '#666666',
        },
        border: {
          DEFAULT: '#242424',
          hover: '#333333',
        },
        btn: {
          primaryBg: '#F5F5F5',
          primaryText: '#080808',
        },
        accent: {
          primary: '#8B8CF7',
          secondary: '#A78BFA',
        },
        status: {
          success: '#4ADE80',
          warning: '#FBBF24',
          error: '#F87171',
        },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      borderRadius: {
        'panel': '24px',
        'card': '20px',
      },
    },
  },
  plugins: [],
}
