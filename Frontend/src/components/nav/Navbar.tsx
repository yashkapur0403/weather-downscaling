'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Brand } from './Brand';
import { ThemeToggle } from './ThemeToggle';

const sections = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'how-it-works', label: 'How It Works' },
  { id: 'model-info', label: 'Model' },
];

export function Navbar() {
  const [activeSection, setActiveSection] = useState('dashboard');
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

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
    <nav className={`dashboard-navbar${scrolled ? ' is-scrolled' : ''}`}>
      <div className={`navbar-row relative flex items-center px-4 ${scrolled ? 'h-10' : 'h-12'}`}>
        <Brand href="/" />

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

        <div className="ml-auto flex items-center gap-4">
          <ThemeToggle />
          <Link
            href="/"
            className="landing-cta landing-cta-ghost"
            style={{ padding: '0.4rem 0.8rem', fontSize: '0.6rem' }}
          >
            <ArrowLeft className="w-3 h-3" />
            Home
          </Link>
        </div>
      </div>
    </nav>
  );
}
