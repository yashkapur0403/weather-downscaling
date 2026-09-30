'use client';

import type { Panchayat, ModelMetrics } from '../../types';
import { classifyRisk, RISK_LABEL } from '../../types';
import { CloudRain, Brain, MapPin, Info, Thermometer, Droplets, Mountain } from 'lucide-react';

interface RightSidebarProps {
  selected: Panchayat | null;
  metrics: ModelMetrics | null;
  loadingMetrics: boolean;
}

export function RightSidebar({
  selected,
  metrics,
  loadingMetrics,
}: RightSidebarProps) {
  const risk = selected ? classifyRisk(selected.rainfall_mm) : null;

  if (!selected) {
    return (
      <div className="h-full flex flex-col items-center justify-center px-6">
        <div className="w-8 h-8 rounded-full flex items-center justify-center mb-3" style={{ border: '1px solid var(--hairline)' }}>
          <div className="w-2 h-2 rounded-full" style={{ background: 'var(--muted)', opacity: 0.4 }} />
        </div>
        <p className="text-[0.6rem] font-mono uppercase tracking-widest text-center" style={{ color: 'var(--muted)' }}>Select a location to begin</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4" style={{ borderBottom: '1px solid var(--hairline)' }}>
        <div className="flex items-center gap-2">
          <div className="w-1 h-1 rounded-full" style={{ background: 'var(--muted)' }} />
          <span className="text-[0.6rem] font-mono uppercase tracking-[0.25em] font-semibold" style={{ color: 'var(--muted)' }}>Projected Indicators</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
        {/* Location */}
        <div className="flex items-center gap-2 pb-3" style={{ borderBottom: '1px solid var(--hairline)' }}>
          <MapPin className="w-4 h-4 flex-shrink-0" style={{ color: 'var(--heat-3)' }} />
          <div>
            <p className="text-xs font-semibold" style={{ color: 'var(--text)' }}>{selected.panchayat_name}</p>
            <p className="text-[0.6rem]" style={{ color: 'var(--muted)' }}>{selected.block_name} · {selected.district} · {selected.state}</p>
          </div>
        </div>

        {/* Rainfall */}
        <div className="metric-panel">
          <div className="flex items-center gap-2 mb-2">
            <CloudRain className="w-4 h-4" style={{ color: 'var(--heat-3)' }} />
            <span className="label-sm">Rainfall</span>
            {risk && (
              <span className="text-[0.6rem] font-mono font-bold uppercase tracking-wider px-1.5 py-0.5 rounded ml-auto" style={{
                background: risk === 'very_heavy' ? 'rgba(162,58,48,0.15)' : risk === 'heavy' ? 'rgba(190,106,46,0.15)' : risk === 'moderate' ? 'rgba(183,146,55,0.15)' : 'rgba(47,111,143,0.15)',
                color: risk === 'very_heavy' ? 'var(--heat-4)' : risk === 'heavy' ? 'var(--heat-3)' : risk === 'moderate' ? 'var(--heat-2)' : 'var(--heat-1)',
              }}>
                {RISK_LABEL[risk]}
              </span>
            )}
          </div>
          <p className="metric-number text-2xl">{selected.rainfall_mm.toFixed(2)}</p>
          <p className="text-[0.6rem] mt-0.5" style={{ color: 'var(--muted)' }}>mm/day · {selected.n_cells} cell{selected.n_cells !== 1 ? 's' : ''} · {selected.mapping_method}</p>
        </div>

        {/* Context fields */}
        <div className="grid grid-cols-3 gap-2">
          <div className="metric-panel">
            <Thermometer className="w-3.5 h-3.5 mb-2" style={{ color: 'var(--heat-3)' }} />
            <p className="metric-number text-lg">{selected.temperature_c.toFixed(1)}°</p>
            <p className="text-[0.55rem] mt-0.5" style={{ color: 'var(--muted)' }}>Temperature</p>
          </div>
          <div className="metric-panel">
            <Droplets className="w-3.5 h-3.5 mb-2" style={{ color: 'var(--heat-1)' }} />
            <p className="metric-number text-lg">{selected.humidity_pct}%</p>
            <p className="text-[0.55rem] mt-0.5" style={{ color: 'var(--muted)' }}>Humidity</p>
          </div>
          <div className="metric-panel">
            <Mountain className="w-3.5 h-3.5 mb-2" style={{ color: 'var(--reference)' }} />
            <p className="metric-number text-lg">{selected.elevation_m}</p>
            <p className="text-[0.55rem] mt-0.5" style={{ color: 'var(--muted)' }}>Elevation m</p>
          </div>
        </div>

        {/* Model MAE */}
        <div className="metric-panel">
          <div className="flex items-center gap-2 mb-2">
            <Brain className="w-4 h-4" style={{ color: 'var(--reference)' }} />
            <span className="label-sm">Model MAE</span>
            {metrics?.mae_improvement_pct && (
              <span className="text-[0.6rem] font-mono font-bold px-1.5 py-0.5 rounded ml-auto" style={{ background: 'rgba(94,140,106,0.15)', color: 'var(--positive)' }}>
                {metrics.mae_improvement_pct}% better
              </span>
            )}
          </div>
          {loadingMetrics && !metrics ? (
            <div className="skeleton h-6 w-20 rounded" />
          ) : (
            <p className="metric-number text-2xl">{metrics?.selected_model.test_mae_mm?.toFixed(2) ?? '—'}</p>
          )}
          <p className="text-[0.6rem] mt-0.5" style={{ color: 'var(--muted)' }}>vs IMD bilinear baseline (7.67 mm)</p>
        </div>

        {/* XAI */}
        <div className="pt-4" style={{ borderTop: '1px solid var(--hairline)' }}>
          <p className="label-sm mb-2">Explanation</p>
          <p className="text-[0.7rem] leading-relaxed" style={{ color: 'var(--text-2)' }}>
            {selected.panchayat_name} received{' '}
            <span style={{ color: 'var(--text)' }}>{selected.rainfall_mm.toFixed(1)} mm/day</span>{' '}
            on {selected.date} —{' '}
            {risk === 'very_heavy' ? 'classified as very heavy rainfall, likely causing flooding concerns.' :
             risk === 'heavy' ? 'classified as heavy rainfall, significant for agriculture.' :
             risk === 'moderate' ? 'classified as moderate rainfall, beneficial for crops.' :
             risk === 'light' ? 'classified as light rainfall, minimal impact.' :
             'no significant rainfall recorded.'}
            {' '}The value comes from {selected.n_cells} grid cell{selected.n_cells !== 1 ? 's' : ''} via {selected.mapping_method.replace('_', ' ')} mapping.
          </p>
          <p className="text-[0.7rem] leading-relaxed mt-2" style={{ color: 'var(--text-2)' }}>
            {selected.mapping_method === 'direct_grid' ? 'Grid cell centres fell directly inside the panchayat polygon — most accurate mapping.' :
             selected.mapping_method === 'area_weighted' ? 'Area-weighted average of overlapping 0.05° cells used for this small polygon.' :
             'Nearest-cell fallback used (< 15 km) — no cell centre inside the polygon.'}
          </p>
        </div>

        {/* Reference */}
        <div className="flex items-start gap-2 p-3" style={{ background: 'var(--raised)', border: '1px solid var(--hairline)', borderRadius: '4px' }}>
          <Info className="w-3 h-3 mt-0.5 flex-shrink-0" style={{ color: 'var(--muted)' }} />
          <p className="text-[0.6rem] leading-relaxed" style={{ color: 'var(--muted)' }}>
            Scored against CHIRPS v2.0 — a reference product, not absolute ground truth.
          </p>
        </div>
      </div>
    </div>
  );
}
