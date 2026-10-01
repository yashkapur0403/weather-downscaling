'use client';

import { useState, useEffect, useCallback } from 'react';
import { fetchAdvisory, ApiError } from '../../api/backend';
import type { Panchayat, AdvisoryResponse, CropType, CropStage, ActionItem } from '../../types';
import { SEVERITY_STYLE, CROP_OPTIONS, STAGE_OPTIONS } from '../../types';
import { Sprout, AlertTriangle, CheckCircle, Info, Loader2, ChevronDown } from 'lucide-react';

interface AdvisoryPanelProps {
  selected: Panchayat | null;
  isProjectionRun: boolean;
}

/** Why the panel has no advisory to show.
 *  - `withheld`    — the server refused: the rainfall we hold is not the value
 *                    it has for this Panchayat and date (409/422).
 *  - `unavailable` — the advisory service could not be reached at all. */
interface AdvisoryError {
  kind: 'withheld' | 'unavailable';
  message: string;
}

/** Compact rendering of a value for the evidence line. */
function fmtValue(v: unknown): string {
  if (v == null) return '';
  if (typeof v === 'number') return String(Math.round(v * 100) / 100);
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (typeof v === 'string') return v;
  if (typeof v === 'object') {
    return Object.entries(v as Record<string, unknown>)
      .filter(([, x]) => x != null && x !== '')
      .map(([k, x]) => `${k.replace(/_/g, ' ')} ${fmtValue(x)}`)
      .join(', ');
  }
  return '';
}

/** "clay 389, rainfall 26.2" from a record of inputs/evidence. */
function evidenceList(obj: Record<string, unknown> | undefined): string {
  if (!obj) return '';
  return Object.entries(obj)
    .filter(([, v]) => v != null && v !== '')
    .map(([k, v]) => `${k.replace(/_/g, ' ')} ${fmtValue(v)}`.trim())
    .filter(s => s && !s.endsWith(' '))
    .join(', ');
}

export function AdvisoryPanel({ selected, isProjectionRun }: AdvisoryPanelProps) {
  const [crop, setCrop] = useState<CropType>('general');
  const [stage, setStage] = useState<CropStage>('general');
  const [advisory, setAdvisory] = useState<AdvisoryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<AdvisoryError | null>(null);

  const loadAdvisory = useCallback(async () => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetchAdvisory(
        selected.panchayat_id,
        crop,
        stage,
        selected.rainfall_mm,
        selected.temperature_c ?? null,
        selected.humidity_pct ?? null,
        selected.date || '2022-07-10',
        selected.panchayat_name,
      );
      setAdvisory(res);
    } catch (e) {
      // A rainfall mismatch (409) or an unverifiable Panchayat/date (422) means
      // the inputs we hold are not the inputs the server can vouch for. Do NOT
      // paper over that with a client-side rainfall-only guess: that would swap a
      // loud, actionable error for confident advice about the wrong numbers.
      if (e instanceof ApiError && (e.status === 409 || e.status === 422)) {
        const detail = (e.body ?? {}) as { error?: string; supplied_mm?: number;
                                           expected_mm?: number; reason?: string };
        setError({
          kind: 'withheld',
          message: detail.error
            ? `${detail.error}${detail.expected_mm != null ? ` (expected ${detail.expected_mm} mm, got ${detail.supplied_mm} mm)` : ''}`
            : `Advisory refused for this Panchayat and date (${detail.reason ?? e.status}).`,
        });
        setAdvisory(null);
        return;
      }
      // The advisory SERVICE is unreachable. The rules live on the server and read
      // the rainfall together with temperature, soil, vegetation and land cover;
      // there is no client-side copy that is guaranteed to agree with it, so we
      // report the failure rather than show an invented second opinion.
      setError({
        kind: 'unavailable',
        message: e instanceof Error ? e.message : 'The advisory service could not be reached.',
      });
      setAdvisory(null);
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
  // Prefer the evidence-bearing action items; fall back to the plain list.
  const actionsWithEvidence: ActionItem[] = advisory
    ? advisory.action_items ?? advisory.actions.map(a => ({
        action: a, risk: '', rule_id: '', severity: 'none' as const,
        evidence: {}, crop_relevant: false, stage_relevant: false,
      }))
    : [];

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
        ) : error ? (
          <div
            className="flex items-start gap-2 px-3 py-2 rounded"
            style={{ background: 'rgba(162,58,48,0.10)', border: '1px solid rgba(162,58,48,0.35)' }}
          >
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-px" style={{ color: 'var(--heat-4)' }} />
            <div className="space-y-1">
              <p className="text-[0.65rem] font-mono font-bold uppercase tracking-wider" style={{ color: 'var(--heat-4)' }}>
                {error.kind === 'withheld' ? 'Advisory withheld' : 'Advisory unavailable'}
              </p>
              <p className="text-[0.65rem] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                {error.message}
              </p>
              <p className="text-[0.6rem] leading-relaxed" style={{ color: 'var(--muted)' }}>
                {error.kind === 'withheld'
                  ? 'The advisory is rule-based on the Layer-1 rainfall for a specific Panchayat and date, so it refuses to run unless that value can be verified. Re-run the projection for this Panchayat.'
                  : 'The advisory rules run on the server, against rainfall, temperature, soil, vegetation and land cover together. They are not reproduced in the browser, so no advice is shown rather than advice that could disagree with the server. The rainfall shown elsewhere is unaffected.'}
              </p>
              {error.kind === 'unavailable' && (
                <button
                  type="button"
                  onClick={loadAdvisory}
                  className="text-[0.6rem] font-mono underline underline-offset-2 pt-0.5"
                  style={{ color: 'var(--heat-4)' }}
                >
                  Retry
                </button>
              )}
            </div>
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

            {/* Recommended actions - each carries the rule + evidence behind it */}
            {actionsWithEvidence.length > 0 && (
              <div className="space-y-1.5">
                <p className="label-sm">Recommended Actions</p>
                <ul className="space-y-2">
                  {actionsWithEvidence.map((item, i) => {
                    const ev = evidenceList(item.evidence);
                    return (
                      <li key={i} className="text-[0.65rem]" style={{ color: 'var(--text-2)' }}>
                        <div className="flex items-start gap-2">
                          <span className="font-mono mt-px flex-shrink-0" style={{ color: sev!.color }}>→</span>
                          <div>
                            <span>{item.action}</span>
                            {(item.risk || ev) && (
                              <p className="text-[0.55rem] leading-relaxed mt-0.5" style={{ color: 'var(--muted)' }}>
                                {item.risk}{ev ? ` — evidence: ${ev}` : ''}
                                {item.crop_relevant ? ' · crop-dependent' : ''}
                                {item.stage_relevant ? ' · stage-dependent' : ''}
                              </p>
                            )}
                          </div>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            {/* Why this advisory? — agricultural evidence, SEPARATE from the model XAI */}
            {(advisory.advisory_context || advisory.soil_context || advisory.crop_suitability ||
              (advisory.fired_rules && advisory.fired_rules.length > 0) || advisory.evidence_groups) && (
              <details className="pt-1" style={{ borderTop: '1px solid var(--hairline)' }}>
                <summary className="label-sm cursor-pointer select-none" style={{ color: 'var(--text-2)' }}>
                  Why this advisory?
                </summary>
                <div className="space-y-2 pt-2">
                  {advisory.advisory_context?.explanation && (
                    <p className="text-[0.6rem] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                      {advisory.advisory_context.explanation}
                    </p>
                  )}

                  {advisory.soil_context && (
                    <div>
                      <p className="text-[0.55rem] font-mono uppercase tracking-wider" style={{ color: 'var(--muted)' }}>
                        Soil characteristics{advisory.soil_context.depth ? ` · ${advisory.soil_context.depth}` : ''}
                      </p>
                      {advisory.soil_context.available ? (
                        <ul className="mt-1 space-y-0.5">
                          {advisory.soil_context.properties.map(p => (
                            <li key={p.key} className="flex justify-between text-[0.6rem]" style={{ color: 'var(--text-2)' }}>
                              <span>{p.label}</span>
                              <span className="font-mono">{p.value}{p.unit ? ` ${p.unit}` : ''}</span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-[0.55rem] mt-0.5 leading-relaxed" style={{ color: 'var(--muted)' }}>
                          {advisory.soil_context.note}
                        </p>
                      )}
                    </div>
                  )}

                  {advisory.crop_suitability && !advisory.crop_suitability.available && (
                    <p className="text-[0.55rem] leading-relaxed" style={{ color: 'var(--muted)' }}>
                      {advisory.crop_suitability.reason}
                    </p>
                  )}

                  {advisory.fired_rules && advisory.fired_rules.length > 0 && (
                    <div>
                      <p className="text-[0.55rem] font-mono uppercase tracking-wider" style={{ color: 'var(--muted)' }}>Triggered rules</p>
                      <ul className="mt-1 space-y-1">
                        {advisory.fired_rules.map(r => {
                          const ev = evidenceList(r.inputs);
                          return (
                            <li key={r.rule_id} className="text-[0.55rem] leading-relaxed" style={{ color: 'var(--muted)' }}>
                              <span style={{ color: 'var(--text-2)' }}>{r.name}</span> — {r.condition}
                              {ev ? ` (${ev})` : ''}
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}

                  {advisory.evidence_groups && (
                    <div className="space-y-0.5">
                      <p className="text-[0.55rem] font-mono uppercase tracking-wider" style={{ color: 'var(--muted)' }}>Evidence</p>
                      {(['observed', 'user_provided', 'derived'] as const).map(g => (
                        <p key={g} className="text-[0.55rem] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                          <span style={{ color: 'var(--muted)' }}>{g.replace(/_/g, ' ')}: </span>
                          {evidenceList(advisory.evidence_groups![g] as Record<string, unknown>) || '—'}
                        </p>
                      ))}
                      {advisory.evidence_groups.not_available.length > 0 && (
                        <p className="text-[0.55rem] leading-relaxed" style={{ color: 'var(--muted)' }}>
                          not available: {advisory.evidence_groups.not_available.join('; ')}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </details>
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
