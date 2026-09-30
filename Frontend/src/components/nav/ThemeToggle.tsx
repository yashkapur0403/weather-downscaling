'use client';

import { useEffect, useState } from 'react';
import { Sun, Moon } from 'lucide-react';

type Theme = 'light' | 'dark';

/**
 * Mirrors whatever the blocking theme-init script (see layout.tsx head) put
 * on <html data-theme>, so there is no flash on mount — this just reads what
 * is already there rather than deciding a theme itself.
 */
function currentTheme(): Theme {
  if (typeof document === 'undefined') return 'dark';
  return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

export function ThemeToggle() {
  // Start undefined so the server-rendered markup and the first client
  // render match; the real theme is applied in an effect once mounted.
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    setTheme(currentTheme());
  }, []);

  const toggle = () => {
    const next: Theme = currentTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try {
      localStorage.setItem('theme', next);
    } catch {
      // private browsing / storage disabled — theme just won't persist
    }
    setTheme(next);
  };

  // Render the dark-mode icon as the placeholder before mount (matches the
  // site's default theme), so there's no layout shift once hydrated.
  const active = theme ?? 'dark';

  return (
    <button
      type="button"
      onClick={toggle}
      className="theme-toggle-btn"
      aria-label={`Switch to ${active === 'dark' ? 'light' : 'dark'} mode`}
      title={`Switch to ${active === 'dark' ? 'light' : 'dark'} mode`}
    >
      {active === 'dark' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
    </button>
  );
}
