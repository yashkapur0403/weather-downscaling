'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Brand } from './Brand';
import { LayoutDashboard } from 'lucide-react';

const links = [
  { id: 'overview',     label: 'Overview'     },
  { id: 'how-it-works', label: 'How It Works' },
  { id: 'explainable-ai', label: 'Explainable AI' },
  { id: 'results',      label: 'Results'      },
  { id: 'faq',          label: 'FAQ'          },
];

const GITHUB_URL = 'https://github.com/yashkapur0403/weather-downscaling';

export function LandingNav() {
  const [activeSection, setActiveSection] = useState('');
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActiveSection(visible.target.id);
      },
      { rootMargin: '-20% 0px -55% 0px', threshold: [0.1, 0.35, 0.6] },
    );
    links.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const scrollTo = (id: string) => {
    setActiveSection(id);
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <nav
      className="dashboard-navbar"
      style={{
        background: 'var(--canvas)',
        transition: 'background 0.3s ease',
      }}
    >

      <div className="relative flex h-[3.25rem] items-center px-5 md:px-8">
        {/* Brand */}
        <Brand href="/" />

        {/* Centered nav links */}
        <div className="hidden md:flex absolute left-1/2 -translate-x-1/2 items-center gap-6">
          {links.map(({ id, label }) => (
            <button
              key={id}
              onClick={() => scrollTo(id)}
              className="nav-section-link text-[0.7rem] font-semibold uppercase cursor-pointer bg-transparent border-none transition-colors"
              data-active={activeSection === id}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Right actions */}
        <div className="ml-auto flex items-center gap-3">
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="hidden md:flex items-center gap-1.5 text-[0.68rem] font-medium no-underline uppercase tracking-wide transition-colors"
            style={{ color: 'var(--muted)' }}
            onMouseEnter={e => (e.currentTarget.style.color = 'var(--text)')}
            onMouseLeave={e => (e.currentTarget.style.color = 'var(--muted)')}
          >
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
            </svg>
            GitHub
          </a>

          <Link
            href="/dashboard"
            className="landing-cta landing-cta-primary flex items-center gap-1.5"
            style={{ padding: '0.4rem 0.85rem', fontSize: '0.65rem' }}
          >
            <LayoutDashboard className="w-3 h-3" />
            Dashboard
          </Link>
        </div>
      </div>
    </nav>
  );
}
