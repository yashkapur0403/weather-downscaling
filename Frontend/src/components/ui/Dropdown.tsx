'use client';

import { useEffect, useRef, useState } from 'react';
import { ChevronDown } from 'lucide-react';

interface DropdownOption {
  value: string;
  label: string;
}

interface DropdownProps {
  value: string;
  options: readonly DropdownOption[];
  onChange: (value: string) => void;
  triggerClassName?: string;
  triggerStyle?: React.CSSProperties;
  icon?: React.ReactNode;
  title?: string;
  align?: 'left' | 'right';
}

/**
 * A fully custom, React-positioned dropdown — not a native <select>.
 *
 * Native <select> popups are rendered by the OS/browser outside the normal
 * layout, and once an ancestor uses backdrop-filter or certain transforms
 * (as several panels in this app now do for the glass look), Chromium can
 * mis-anchor that popup far from its trigger with no theme styling applied.
 * This component renders its own list in React, so position and styling are
 * always correct and consistent everywhere it's used.
 */
export function Dropdown({
  value,
  options,
  onChange,
  triggerClassName = 'advisory-select',
  triggerStyle,
  icon,
  title,
  align = 'left',
}: DropdownProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const selected = options.find(o => o.value === value);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <div className="relative" ref={rootRef}>
      <button
        type="button"
        className={triggerClassName}
        style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', ...triggerStyle }}
        onClick={() => setOpen(o => !o)}
        title={title}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        {icon}
        <span>{selected?.label ?? value}</span>
        <ChevronDown className="w-3 h-3 flex-shrink-0" style={{ color: 'var(--muted)' }} />
      </button>

      {open && (
        <div
          className="sidebar-dropdown"
          role="listbox"
          style={{
            top: 'calc(100% + 4px)',
            left: align === 'left' ? 0 : 'auto',
            right: align === 'right' ? 0 : 'auto',
            minWidth: '100%',
            width: 'max-content',
            maxWidth: '16rem',
          }}
        >
          {options.map(o => (
            <button
              key={o.value}
              type="button"
              role="option"
              aria-selected={o.value === value}
              className="sidebar-dropdown-item"
              style={{
                fontSize: '0.7rem',
                color: o.value === value ? 'var(--text)' : 'var(--text-2)',
                background: o.value === value ? 'rgba(176, 141, 87, 0.08)' : undefined,
              }}
              onClick={() => {
                onChange(o.value);
                setOpen(false);
              }}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
