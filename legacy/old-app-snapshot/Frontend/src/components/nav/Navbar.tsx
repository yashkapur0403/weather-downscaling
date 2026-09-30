'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

const sections = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'how-it-works', label: 'How It Works' },
  { id: 'model-info', label: 'Model' },
];

export function Navbar() {
  const [activeSection, setActiveSection] = useState('dashboard');

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActiveSection(visible.target.id);
      },
      { rootMargin: '-20% 0px -55% 0px', threshold: [0.1, 0.35, 0.6] },
    );

    sections.forEach(({ id }) => {
      const section = document.getElementById(id);
      if (section) observer.observe(section);
    });
    return () => observer.disconnect();
  }, []);

  const scrollTo = (id: string) => {
    setActiveSection(id);
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <nav className="dashboard-navbar">
      <div className="relative flex h-12 items-center px-4">
        <Link href="/" className="flex items-center gap-2.5 no-underline">
          <div style={{ background: '#1E3A8A', width: '1.75rem', height: '1.75rem', borderRadius: '0.375rem', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
            <svg viewBox="0 0 20 20" fill="none" style={{ width: '0.875rem', height: '0.875rem' }}>
              <circle cx="10" cy="10" r="7" stroke="white" strokeWidth="1.5" />
              <path d="M10 6v4l2.5 2.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </div>
          <div>
            <p className="font-bold text-xs leading-none" style={{ color: 'var(--text)' }}>Obsidian</p>
            <p className="text-[0.5rem] tracking-widest uppercase" style={{ color: 'var(--muted)' }}>Weather Intelligence</p>
          </div>
        </Link>

        <div className="hidden md:flex absolute left-1/2 -translate-x-1/2 items-center gap-8">
          {sections.map(({ id, label }) => (
            <button
              key={id}
              onClick={() => scrollTo(id)}
              className="nav-section-link text-xs font-semibold transition-colors uppercase cursor-pointer bg-transparent border-none"
              aria-current={activeSection === id ? 'page' : undefined}
              data-active={activeSection === id}
            >
              {label}
            </button>
          ))}
        </div>

        <a
          href="https://github.com/yashkapur0403/weather-downscaling"
          target="_blank"
          rel="noopener noreferrer"
          className="hidden md:block ml-auto text-xs font-semibold no-underline uppercase tracking-wide"
          style={{ color: 'var(--muted)' }}
        >
          GitHub ↗
        </a>
      </div>
    </nav>
  );
}
