/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Mirror the CSS custom properties in index.css, not literal hex
        // values, so bg-*/text-* utility classes respond to data-theme too.
        bg:        'var(--bg)',
        surface:   'var(--surface)',
        raised:    'var(--raised)',
        border:    'var(--border)',
        primary:   'var(--primary)',
        navy:      'var(--navy)',
        text:      'var(--text)',
        'text-2':  'var(--text-2)',
        muted:     'var(--muted)',
        copper:    'var(--copper)',
        reference: 'var(--reference)',
        positive:  'var(--positive)',
        'heat-1':  'var(--heat-1)',
        'heat-2':  'var(--heat-2)',
        'heat-3':  'var(--heat-3)',
        'heat-4':  'var(--heat-4)',
        'heat-5':  'var(--heat-5)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
};
