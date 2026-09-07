'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { searchPanchayats } from '../../api/backend';
import type { Panchayat } from '../../types';
import { classifyRisk, RISK_LABEL } from '../../types';
import { Search, MapPin, Loader2, Layers, Clock, X, BarChart3, Activity } from 'lucide-react';

// ── Recent searches helpers ───────────────────────────────────────────────────
const RECENT_KEY = 'freebuff_recent_searches';
const MAX_RECENT = 6;

function loadRecent(): Panchayat[] {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveRecent(p: Panchayat) {
  const list = loadRecent().filter(r => r.panchayat_id !== p.panchayat_id);
  list.unshift(p);
  localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, MAX_RECENT)));
}

function clearRecent() {
  localStorage.removeItem(RECENT_KEY);
}

// ── Component ─────────────────────────────────────────────────────────────────
interface LeftSidebarProps {
  onSelect: (p: Panchayat) => void;
  onGenerate: () => void;
  isLoading: boolean;
  hasSelected: boolean;
  selected: Panchayat | null;
  modelStatus?: 'loading' | 'ready' | 'unavailable';
}

export function LeftSidebar({
  onSelect,
  onGenerate,
  isLoading,
  hasSelected,
  selected,
  modelStatus = 'loading',
}: LeftSidebarProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Panchayat[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Recent searches — empty on server, populated after hydration
  const [recent, setRecent] = useState<Panchayat[]>([]);
  useEffect(() => {
    setRecent(loadRecent());
  }, []);

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

  // Search handler
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
        const data = await searchPanchayats(val.trim(), 15);
        setResults(data);
        setOpen(true);
        if (data.length === 0) setError('No panchayats found.');
      } catch {
        setError('Backend not connected.');
        setResults([]);
        setOpen(true);
      } finally {
        setLoading(false);
      }
    }, 300);
  }, []);

  const handleSelect = useCallback((p: Panchayat) => {
    setQuery(`${p.panchayat_name}`);
    setOpen(false);
    onSelect(p);
    saveRecent(p);
    setRecent(loadRecent());
  }, [onSelect]);

  const handleClearRecent = useCallback(() => {
    clearRecent();
    setRecent([]);
  }, []);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-4 py-3" style={{ borderBottom: '1px solid var(--hairline)' }}>
        <div className="flex items-center gap-2">
          <div className="w-1 h-1 rounded-full" style={{ background: 'var(--muted)' }} />
          <span className="label-sm" style={{ color: 'var(--muted)' }}>Projection Controls</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-5">
        {/* Location Search */}
        <div className="space-y-2">
          <label className="label-sm block" style={{ color: 'var(--muted)' }}>Location</label>
          <div ref={wrapperRef} className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none" style={{ color: 'var(--muted)' }} />
            <input
              type="text"
              className="sidebar-input pl-10 pr-10"
              placeholder="Search panchayat..."
              value={query}
              onChange={e => handleChange(e.target.value)}
              onFocus={() => results.length > 0 && setOpen(true)}
              autoComplete="off"
            />
            {loading && (
              <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 animate-spin" style={{ color: 'var(--muted)' }} />
            )}
            {open && (
              <div className="sidebar-dropdown">
                {error ? (
                  <div className="p-3 text-xs" style={{ color: 'var(--muted)' }}>{error}</div>
                ) : results.length === 0 ? (
                  <div className="p-3 text-xs text-center" style={{ color: 'var(--muted)' }}>No results</div>
                ) : (
                  results.map(p => (
                    <button
                      key={p.panchayat_id}
                      className="sidebar-dropdown-item"
                      onClick={() => handleSelect(p)}
                    >
                      <MapPin className="w-3 h-3 mt-0.5 flex-shrink-0" style={{ color: 'var(--muted)' }} />
                      <div>
                        <p className="text-xs font-semibold" style={{ color: 'var(--text)' }}>{p.panchayat_name}</p>
                        <p className="text-[0.6rem]" style={{ color: 'var(--muted)' }}>{p.block_name} · {p.district}</p>
                        <p className="text-[0.6rem] font-mono" style={{ color: 'var(--heat-3)' }}>{p.rainfall_mm.toFixed(2)} mm</p>
                      </div>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>
          {hasSelected && (
            <div className="flex items-center gap-1.5 pl-1">
              <div className="w-1 h-1 rounded-full" style={{ background: 'var(--positive)' }} />
              <p className="text-[0.6rem] font-mono" style={{ color: 'var(--muted)' }}>Location locked</p>
            </div>
          )}
        </div>

        {/* Data Layer */}
        <div className="space-y-2">
          <label className="label-sm block" style={{ color: 'var(--muted)' }}>Data Layer</label>
          <div className="sidebar-select">
            <Layers className="w-4 h-4" style={{ color: 'var(--muted)' }} />
            <span className="text-xs" style={{ color: 'var(--text)' }}>U-Net Model D (DEM + ERA5)</span>
          </div>
          {/* Model Status Indicator */}
          <div className="flex items-center gap-2 mt-2">
            <div className={`w-2 h-2 rounded-full ${modelStatus === 'ready' ? 'bg-green-500' : modelStatus === 'loading' ? 'bg-yellow-500 animate-pulse' : 'bg-red-500'}`} />
            <span className="text-[0.6rem]" style={{ color: 'var(--muted)' }}>
              Model: {modelStatus === 'ready' ? 'Ready' : modelStatus === 'loading' ? 'Loading...' : 'Unavailable'}
            </span>
          </div>
        </div>

        {/* Generate Button */}
        <button
          onClick={onGenerate}
          disabled={!hasSelected || isLoading}
          className="sidebar-button"
        >
          {isLoading ? (
            <span className="flex items-center justify-center gap-2">
              <div className="w-3 h-3 border border-current/30 border-t-current rounded-full animate-spin" />
              Analyzing...
            </span>
          ) : (
            <span>Run Projection →</span>
          )}
        </button>

        {/* Quick Stats */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-3.5 h-3.5" style={{ color: 'var(--muted)' }} />
            <label className="label-sm">Dataset Overview</label>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {[
              { label: 'Panchayats', value: '8,236' },
              { label: 'States', value: '6' },
              { label: 'Reference', value: 'CHIRPS' },
            ].map(s => (
              <div key={s.label} className="sidebar-stat-card">
                <p className="text-[0.55rem] uppercase tracking-wider" style={{ color: 'var(--muted)' }}>{s.label}</p>
                <p className="font-mono font-semibold text-xs" style={{ color: 'var(--text)' }}>{s.value}</p>
              </div>
            ))}
          </div>

          {selected && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Activity className="w-3.5 h-3.5" style={{ color: 'var(--muted)' }} />
                <label className="label-sm">Selection Summary</label>
              </div>
              <div className="p-3 space-y-2" style={{ background: 'var(--raised)', border: '1px solid var(--hairline)', borderRadius: '4px' }}>
                <div className="flex items-center justify-between">
                  <span className="text-[0.6rem]" style={{ color: 'var(--muted)' }}>Rainfall</span>
                  <span className="font-mono text-xs font-semibold" style={{ color: 'var(--heat-3)' }}>{selected.rainfall_mm.toFixed(2)} mm</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[0.6rem]" style={{ color: 'var(--muted)' }}>Risk Level</span>
                  <span className="text-[0.6rem] font-mono font-bold uppercase tracking-wider px-1.5 py-0.5 rounded" style={{
                    background: classifyRisk(selected.rainfall_mm) === 'very_heavy' ? 'rgba(162,58,48,0.15)' : classifyRisk(selected.rainfall_mm) === 'heavy' ? 'rgba(190,106,46,0.15)' : classifyRisk(selected.rainfall_mm) === 'moderate' ? 'rgba(183,146,55,0.15)' : 'rgba(47,111,143,0.15)',
                    color: classifyRisk(selected.rainfall_mm) === 'very_heavy' ? 'var(--heat-4)' : classifyRisk(selected.rainfall_mm) === 'heavy' ? 'var(--heat-3)' : classifyRisk(selected.rainfall_mm) === 'moderate' ? 'var(--heat-2)' : 'var(--heat-1)',
                  }}>
                    {RISK_LABEL[classifyRisk(selected.rainfall_mm)]}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[0.6rem]" style={{ color: 'var(--muted)' }}>Grid Cells</span>
                  <span className="font-mono text-xs" style={{ color: 'var(--text)' }}>{selected.n_cells}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[0.6rem]" style={{ color: 'var(--muted)' }}>Mapping</span>
                  <span className="font-mono text-[0.6rem]" style={{ color: 'var(--text-2)' }}>{selected.mapping_method.replace('_', ' ')}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Recent Searches */}
        {recent.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="label-sm block" style={{ color: 'var(--muted)' }}>Recent</label>
              <button
                onClick={handleClearRecent}
                className="flex items-center gap-1 text-[0.55rem] uppercase tracking-wider transition-opacity hover:opacity-100 opacity-40"
                style={{ color: 'var(--muted)' }}
              >
                <X className="w-2.5 h-2.5" />
                Clear
              </button>
            </div>
            <div className="space-y-1">
              {recent.map(p => (
                <button
                  key={p.panchayat_id}
                  className="sidebar-recent-item"
                  onClick={() => handleSelect(p)}
                >
                  <Clock className="w-3 h-3 flex-shrink-0" style={{ color: 'var(--muted)', opacity: 0.5 }} />
                  <div className="min-w-0">
                    <p className="text-[0.65rem] font-medium truncate" style={{ color: 'var(--text)' }}>{p.panchayat_name}</p>
                    <p className="text-[0.55rem] truncate" style={{ color: 'var(--muted)' }}>{p.district} · {p.state}</p>
                  </div>
                  <span className="text-[0.55rem] font-mono flex-shrink-0 ml-auto" style={{ color: 'var(--heat-3)' }}>
                    {p.rainfall_mm.toFixed(1)}mm
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-3" style={{ borderTop: '1px solid var(--hairline)' }}>
        <p className="text-[0.6rem] leading-relaxed" style={{ color: 'rgba(140,140,150,0.5)' }}>
          Obsidian uses local state only — no third-party cookies.
        </p>
      </div>
    </div>
  );
}
