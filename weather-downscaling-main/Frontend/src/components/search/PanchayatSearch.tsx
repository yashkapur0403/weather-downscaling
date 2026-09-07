'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { searchPanchayats } from '../../api/backend';
import type { Panchayat } from '../../types';
import { Search, MapPin, Loader2 } from 'lucide-react';

interface PanchayatSearchProps {
  onSelect: (p: Panchayat) => void;
  placeholder?: string;
}

export function PanchayatSearch({ onSelect, placeholder = 'Search panchayat, block or district…' }: PanchayatSearchProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Panchayat[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    function handle(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handle);
    return () => document.removeEventListener('mousedown', handle);
  }, []);

  const handleChange = useCallback((val: string) => {
    setQuery(val);
    setError(null);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!val.trim()) {
      setResults([]);
      setOpen(false);
      return;
    }
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const data = await searchPanchayats(val, 15);
        setResults(data);
        setOpen(true);
        if (data.length === 0) setError('No panchayats found. Try a district or block name.');
      } catch (err) {
        setError('Backend not connected — start the FastAPI server on port 8001.');
        setResults([]);
        setOpen(true);
      } finally {
        setLoading(false);
      }
    }, 300);
  }, []);

  const handleSelect = useCallback((p: Panchayat) => {
    setQuery(`${p.panchayat_name}, ${p.block_name}, ${p.district}`);
    setOpen(false);
    onSelect(p);
  }, [onSelect]);

  return (
    <div ref={wrapperRef} className="relative w-full">
      {/* Input */}
      <div className="relative">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-muted w-5 h-5 pointer-events-none" />
        <input
          type="text"
          className="search-input"
          placeholder={placeholder}
          value={query}
          onChange={e => handleChange(e.target.value)}
          onFocus={() => results.length > 0 && setOpen(true)}
          aria-label="Search panchayat"
          autoComplete="off"
        />
        {loading && (
          <Loader2 className="absolute right-4 top-1/2 -translate-y-1/2 text-muted w-4 h-4 animate-spin" />
        )}
      </div>

      {/* Dropdown */}
      {open && (
        <div className="dropdown fade-in">
          {error ? (
            <div className="p-4 text-sm text-text-2">
              <span className="badge badge-red mr-2">Error</span>
              {error}
            </div>
          ) : results.length === 0 ? (
            <div className="p-4 text-sm text-muted text-center">No results</div>
          ) : (
            results.map(p => (
              <button
                key={p.panchayat_id}
                className="dropdown-item"
                onClick={() => handleSelect(p)}
              >
                <div className="flex items-start gap-2.5">
                  <MapPin className="w-4 h-4 text-muted mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-semibold text-text">
                      {p.panchayat_name}
                    </p>
                    <p className="text-xs text-text-2">
                      {p.block_name} · {p.district} · {p.state}
                    </p>
                    <p className="text-xs font-mono text-primary mt-0.5">
                      {p.rainfall_mm.toFixed(2)} mm — {p.date}
                    </p>
                  </div>
                </div>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
