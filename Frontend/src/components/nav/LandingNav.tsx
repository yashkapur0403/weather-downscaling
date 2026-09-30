'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Brand } from './Brand';
import { ThemeToggle } from './ThemeToggle';
import { LayoutDashboard } from 'lucide-react';

const links = [
  { id: 'overview',     label: 'Overview'     },
  { id: 'how-it-works', label: 'How It Works' },
  { id: 'explainable-ai', label: 'Explainable AI' },
  { id: 'results',      label: 'Results'      },
  { id: 'faq',          label: 'FAQ'          },
];

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
    <nav className={`dashboard-navbar${scrolled ? ' is-scrolled' : ''}`}>
      <div className={`navbar-row relative flex items-center px-5 md:px-6 ${scrolled ? 'h-10' : 'h-12'}`}>
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
          <ThemeToggle />
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
