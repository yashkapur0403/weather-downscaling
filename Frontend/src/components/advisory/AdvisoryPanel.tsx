'use client';

import { useState, useEffect, useCallback } from 'react';
import { fetchAdvisory, ApiError } from '../../api/backend';
import type { Panchayat, AdvisoryResponse, CropType, CropStage, LangCode } from '../../types';
import { SEVERITY_STYLE, CROP_OPTIONS, STAGE_OPTIONS, LANGUAGE_OPTIONS } from '../../types';
import { Sprout, AlertTriangle, CheckCircle, Info, Loader2, Languages } from 'lucide-react';
import { Dropdown } from '../ui/Dropdown';

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

export function AdvisoryPanel({ selected, isProjectionRun }: AdvisoryPanelProps) {
  const [crop, setCrop] = useState<CropType>('general');
  const [stage, setStage] = useState<CropStage>('general');
  // Regional language for the advisory text. The rule engine and Groq always
  // run in English; when this is not 'en-IN' the backend additionally routes
  // the result through Sarvam AI to translate it before it reaches the UI.
  const [lang, setLang] = useState<LangCode>('en-IN');
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
        selected.date || '2022-07-10',
        selected.panchayat_name,
        undefined,
        lang,
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
  }, [selected, crop, stage, lang]);

  // Auto-fetch when selection changes or crop/stage/language changes
  useEffect(() => {
    if (isProjectionRun && selected) {
      loadAdvisory();
    } else {
      setAdvisory(null);
    }
  }, [isProjectionRun, selected, crop, stage, lang, loadAdvisory]);

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
          <Dropdown value={crop} options={CROP_OPTIONS} onChange={v => setCrop(v as CropType)} />
          <Dropdown value={stage} options={STAGE_OPTIONS} onChange={v => setStage(v as CropStage)} />
          {/* Regional language selector, translated server-side via Sarvam AI */}
          <Dropdown
            value={lang}
            options={LANGUAGE_OPTIONS}
            onChange={v => setLang(v as LangCode)}
            icon={<Languages className="w-3 h-3 flex-shrink-0" style={{ color: 'var(--muted)' }} />}
            title="Translate this advisory into a regional language (Sarvam AI)"
            align="right"
          />
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
