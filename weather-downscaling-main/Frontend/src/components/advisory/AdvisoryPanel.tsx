'use client';

import { useState, useEffect, useCallback } from 'react';
import { fetchAdvisory } from '../../api/backend';
import type { Panchayat, AdvisoryResponse, CropType, CropStage } from '../../types';
import { SEVERITY_STYLE, CROP_OPTIONS, STAGE_OPTIONS } from '../../types';
import { Sprout, AlertTriangle, CheckCircle, Info, Loader2, ChevronDown } from 'lucide-react';

interface AdvisoryPanelProps {
  selected: Panchayat | null;
  isProjectionRun: boolean;
}

// ── Client-side fallback advisory (used when backend /api/advisory is offline) ─
function generateLocalAdvisory(p: Panchayat, crop: CropType, stage: CropStage): AdvisoryResponse {
  const rainfall = p.rainfall_mm;
  let severity: AdvisoryResponse['severity'] = 'info';
  let advisoryText = '';
  const actions: string[] = [];

  if (rainfall >= 64.5) {
    severity = 'alert';
    advisoryText = `Extremely heavy rainfall (${rainfall.toFixed(1)} mm) predicted. Potential for severe flooding and crop submersion.`;
    actions.push('Delay all field operations', 'Clear drainage channels immediately', 'Prepare for potential crop loss assessment');
  } else if (rainfall >= 24.5) {
    severity = 'warning';
    advisoryText = `Heavy rainfall (${rainfall.toFixed(1)} mm) expected. Risk of waterlogging in low-lying fields.`;
    actions.push('Ensure field drainage is functional', 'Avoid pesticide/fertilizer application for 48h', 'Monitor crop for lodging');
  } else if (rainfall >= 6.5) {
    severity = 'watch';
    advisoryText = `Moderate rainfall (${rainfall.toFixed(1)} mm). Beneficial for soil moisture but monitor for excess.`;
    actions.push('Review irrigation schedule — rain may reduce need', 'Inspect field margins for early drainage issues');
  } else {
    severity = 'info';
    advisoryText = `Light or no rainfall (${rainfall.toFixed(1)} mm). Conditions are stable for field operations.`;
    actions.push('Continue scheduled operations', 'Maintain regular irrigation if dry conditions persist');
  }

  if (weather?.temperature_c != null && weather.temperature_c > 38) {
    actions.push('High temperature stress — ensure adequate irrigation');
  }

  return {
    advisory_text: advisoryText,
    severity,
    actions,
    evidence: {
      rainfall_mm: rainfall,
      risk_level: (() => {
        if (rainfall < 2.5) return 'no_rain';
        if (rainfall < 10) return 'light';
        if (rainfall < 25) return 'moderate';
        if (rainfall < 50) return 'heavy';
        return 'very_heavy';
      })(),
      temperature_c: null,
      humidity_pct: null,
    },
    data_date: p.date,
    disclaimer: 'Client-side advisory based on rainfall thresholds only. Not a substitute for official meteorological guidance.',
    crop,
    stage,
  };
}

export function AdvisoryPanel({ selected, isProjectionRun }: AdvisoryPanelProps) {
  const [crop, setCrop] = useState<CropType>('general');
  const [stage, setStage] = useState<CropStage>('general');
  const [advisory, setAdvisory] = useState<AdvisoryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadAdvisory = useCallback(async () => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetchAdvisory(selected.panchayat_id, crop, stage);
      setAdvisory(res);
    } catch {
      // Backend /api/advisory not available — fall back to local rules
      setAdvisory(generateLocalAdvisory(selected, crop, stage));
    } finally {
      setLoading(false);
    }
  }, [selected, crop, stage]);

  // Auto-fetch when selection changes or crop/stage changes
  useEffect(() => {
    if (isProjectionRun && selected) {
      loadAdvisory();
    } else {
      setAdvisory(null);
    }
  }, [isProjectionRun, selected, crop, stage, loadAdvisory]);

  if (!isProjectionRun || !selected) return null;

  const sev = advisory ? SEVERITY_STYLE[advisory.severity] : null;

  return (
    <div className="advisory-panel">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3" style={{ borderBottom: '1px solid var(--hairline)' }}>
        <div className="flex items-center gap-2">
          <Sprout className="w-4 h-4" style={{ color: 'var(--positive)' }} />
          <span className="label-sm">Crop Advisory</span>
        </div>
        <div className="flex items-center gap-2">
          {/* Crop selector */}
          <div className="relative">
            <select
              className="advisory-select"
              value={crop}
              onChange={e => setCrop(e.target.value as CropType)}
            >
              {CROP_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
            <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 pointer-events-none" style={{ color: 'var(--muted)' }} />
          </div>
          {/* Stage selector */}
          <div className="relative">
            <select
              className="advisory-select"
              value={stage}
              onChange={e => setStage(e.target.value as CropStage)}
            >
              {STAGE_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
            <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 pointer-events-none" style={{ color: 'var(--muted)' }} />
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="px-5 py-4 space-y-3">
        {loading ? (
          <div className="flex items-center gap-2 py-4 justify-center">
            <Loader2 className="w-4 h-4 animate-spin" style={{ color: 'var(--muted)' }} />
            <span className="text-xs" style={{ color: 'var(--muted)' }}>Generating advisory...</span>
          </div>
        ) : advisory ? (
          <>
            {/* Severity badge */}
            <div
              className="flex items-center gap-2 px-3 py-2 rounded"
              style={{ background: sev!.bg, border: `1px solid ${sev!.border}` }}
            >
              {advisory.severity === 'alert' || advisory.severity === 'warning' ? (
                <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: sev!.color }} />
              ) : advisory.severity === 'watch' ? (
                <Info className="w-4 h-4 flex-shrink-0" style={{ color: sev!.color }} />
              ) : (
                <CheckCircle className="w-4 h-4 flex-shrink-0" style={{ color: sev!.color }} />
              )}
              <span className="text-[0.65rem] font-mono font-bold uppercase tracking-wider" style={{ color: sev!.color }}>
                {advisory.severity}
              </span>
            </div>

            {/* Advisory text */}
            <p className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>
              {advisory.advisory_text}
            </p>

            {/* Actions */}
            {advisory.actions.length > 0 && (
              <div className="space-y-1.5">
                <p className="label-sm">Recommended Actions</p>
                <ul className="space-y-1">
                  {advisory.actions.map((action, i) => (
                    <li key={i} className="flex items-start gap-2 text-[0.65rem]" style={{ color: 'var(--text-2)' }}>
                      <span className="font-mono mt-px flex-shrink-0" style={{ color: sev!.color }}>→</span>
                      {action}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Evidence row */}
            <div className="flex items-center gap-3 pt-2" style={{ borderTop: '1px solid var(--hairline)' }}>
              <span className="text-[0.55rem] font-mono" style={{ color: 'var(--muted)' }}>
                {advisory.evidence.rainfall_mm.toFixed(1)} mm
              </span>
              {advisory.evidence.temperature_c != null && (
                <span className="text-[0.55rem] font-mono" style={{ color: 'var(--muted)' }}>
                  {advisory.evidence.temperature_c}°C
                </span>
              )}
              {advisory.evidence.humidity_pct != null && (
                <span className="text-[0.55rem] font-mono" style={{ color: 'var(--muted)' }}>
                  {advisory.evidence.humidity_pct}% RH
                </span>
              )}
              <span className="text-[0.55rem] font-mono ml-auto" style={{ color: 'var(--muted)', opacity: 0.5 }}>
                {advisory.data_date}
              </span>
            </div>

            {/* Disclaimer */}
            <p className="text-[0.55rem] leading-relaxed pt-1" style={{ color: 'rgba(140,140,150,0.4)' }}>
              {advisory.disclaimer}
            </p>
          </>
        ) : null}
      </div>
    </div>
  );
}
