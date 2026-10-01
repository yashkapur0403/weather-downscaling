import type { Metadata, Viewport } from 'next';
import '../index.css';
import { GithubFab } from '../components/nav/GithubFab';

export const metadata: Metadata = {
  title: 'Obsidian — Panchayat-level Crop Intelligence',
  description:
    'IMD rainfall downscaled from 28 km to 5 km using terrain and atmospheric context. Search 87,735 Gram Panchayats across 15 Indian states.',
  openGraph: {
    title: 'Obsidian — Panchayat-level Crop Intelligence',
    description: 'Downscaled rainfall for every Gram Panchayat.',
  },
};

export const viewport: Viewport = {
  // Mobile browser chrome color; follows system preference since this is
  // static SSR metadata and can't read the user's manual toggle choice.
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#FAF9F6' },
    { media: '(prefers-color-scheme: dark)', color: '#050608' },
  ],
};

// Runs before hydration so the correct theme is on <html> for the very first
// paint — without this, the page would flash dark (the CSS default) before
// a saved "light" preference could apply. suppressHydrationWarning on <html>
// is needed because this script sets the data-theme attribute that the
// server-rendered markup below doesn't have.
const THEME_INIT_SCRIPT = `(function(){try{var t=localStorage.getItem('theme');if(t!=='light'&&t!=='dark'){t=window.matchMedia('(prefers-color-scheme: light)').matches?'light':'dark';}document.documentElement.setAttribute('data-theme',t);}catch(e){}})();`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body>
        {children}
        <GithubFab />
      </body>
    </html>
  );
}
