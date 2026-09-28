/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:        '#050608',
        surface:   '#0A0A0C',
        raised:    '#0F0F12',
        border:    'rgba(255, 255, 255, 0.06)',
        primary:   '#DC2626',
        navy:      '#1E3A8A',
        text:      '#EDEDEF',
        'text-2':  '#B4B4BD',
        muted:     '#8C8C96',
        copper:    '#B08D57',
        reference: '#6E8CA8',
        positive:  '#5E8C6A',
        'heat-1':  '#2F6F8F',
        'heat-2':  '#B79237',
        'heat-3':  '#BE6A2E',
        'heat-4':  '#A23A30',
        'heat-5':  '#6E2020',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
};
