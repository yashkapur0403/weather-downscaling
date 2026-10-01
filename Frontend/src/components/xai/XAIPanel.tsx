'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchExplanation } from '../../api/backend';
import type { Panchayat, ModelMetrics, ExplainRequest, ExplainResponse, ExplainFactor } from '../../types';
import { classifyRisk } from '../../types';
import { Brain, Loader2, RefreshCw, ArrowUp, ArrowDown, Minus, Info } from 'lucide-react';

interface XAIPanelProps {
  selected: Panchayat | null;
  metrics: ModelMetrics | null;
  isProjectionRun: boolean;
}

const DEFAULT_CHANNELS = ['imd_rain', 'dem', 'era5_t2m', 'era5_t2m_max', 'era5_dewp'];

const METHOD_LABELS: Record<string, string> = {
  direct_grid:      'Grid cell centres fall directly inside the panchayat polygon.',
  area_weighted:    'Weighted average of overlapping 0.05° cells (small polygon).',
  nearest_fallback: 'Nearest grid cell centroid used (< 15 km fallback).',
};

const EFFECT_STYLE: Record<ExplainFactor['effect'], { color: string; Icon: typeof ArrowUp; label: string }> = {
  increases: { color: 'var(--heat-3)', Icon: ArrowUp,   label: 'raises estimate' },
  decreases: { color: 'var(--heat-1)', Icon: ArrowDown, label: 'lowers estimate' },
  neutral:   { color: 'var(--muted)',  Icon: Minus,     label: 'little effect' },
};

// ── Build the grounded context sent to POST /api/explain ─────────────────────
function buildRequest(p: Panchayat, metrics: ModelMetrics | null, question?: string): ExplainRequest {
  return {
    panchayat_name: p.panchayat_name,
    block_name: p.block_name,
    district: p.district,
    state: p.state,
    date: p.date || '2022-07-10',
    lat: p.lat,
    lon: p.lon,
    prediction: {
      rainfall_mm: p.rainfall_mm,
      risk_level: classifyRisk(p.rainfall_mm),
      temperature_c: p.temperature_c ?? null,
      humidity_pct: p.humidity_pct ?? null,
      elevation_m: p.elevation_m ?? null,
    },
    mapping: {
      method: p.mapping_method,
      n_cells: p.n_cells,
      fallback_distance_m: p.fallback_distance_m,
    },
    model: {
      name: metrics?.selected_model.name ?? 'U-Net Model D (DEM + ERA5)',
      channels: metrics?.channels ?? DEFAULT_CHANNELS,
      test_mae_mm: metrics?.selected_model.test_mae_mm ?? 6.45,
      baseline_mae_mm: metrics?.baseline.test_mae_mm ?? 7.67,
      mae_improvement_pct: metrics?.mae_improvement_pct ?? 12.2,
      reference_product: metrics?.reference_product ?? 'CHIRPS v2.0',
    },
    question: question ?? null,
    language: 'en',
  };
}

// ── Client-side fallback (used when backend /api/explain or Groq is offline) ─
function generateLocalExplanation(req: ExplainRequest, reason?: string): ExplainResponse {
  const { prediction: pr, mapping, model } = req;
  const factors: ExplainFactor[] = [
    {
      factor: 'IMD coarse rainfall (0.25°)',
      effect: pr.rainfall_mm >= 2.5 ? 'increases' : 'neutral',
      weight: 0.9,
      detail: 'The bilinear-upsampled IMD value is the starting point; the U-Net only learns a correction on top of it.',
    },
  ];
  if (pr.elevation_m != null) {
    factors.push({
      factor: 'Elevation (SRTM DEM)',
      effect: pr.elevation_m > 500 ? 'increases' : 'neutral',
      weight: pr.elevation_m > 500 ? 0.6 : 0.2,
      detail: pr.elevation_m > 500
        ? `At ${pr.elevation_m} m, orographic lift on windward slopes tends to add rainfall.`
        : `Terrain is relatively flat (${pr.elevation_m} m), so elevation adds little.`,
    });
  }
  if (pr.humidity_pct != null) {
    factors.push({
      factor: 'Moisture (ERA5 dewpoint)',
      effect: pr.humidity_pct > 70 ? 'increases' : pr.humidity_pct < 50 ? 'decreases' : 'neutral',
      weight: pr.humidity_pct > 70 ? 0.5 : 0.3,
      detail: `Relative humidity of ${pr.humidity_pct}% is used as a proxy for available moisture.`,
    });
  }
  if (pr.temperature_c != null) {
    factors.push({
      factor: 'Temperature (ERA5 T₂ₘ)',
      effect: 'neutral',
      weight: 0.2,
      detail: `Mean temperature of ${pr.temperature_c}°C gives the model thermal context for convection.`,
    });
  }

  const confidence = mapping.method === 'nearest_fallback' ? 'low' : mapping.n_cells > 1 ? 'high' : 'medium';

  return {
    summary: `${pr.rainfall_mm.toFixed(1)} mm/day estimated for ${req.panchayat_name} on ${req.date}.`,
    explanation:
      `The model starts from IMD's 28 km rainfall, upsamples it to a 5 km grid, and then a U-Net adds a local ` +
      `correction using terrain and ERA5 weather. The panchayat value is then taken from ` +
      `${mapping.n_cells} grid cell${mapping.n_cells !== 1 ? 's' : ''}. ` +
      (model.mae_improvement_pct != null
        ? `On the 2022 test season this model's error was ${model.mae_improvement_pct}% lower than the IMD baseline (scored against ${model.reference_product}).`
        : ''),
    factors,
    confidence,
    confidence_note: METHOD_LABELS[mapping.method] ?? mapping.method,
    answer: req.question
      ? 'The language model is offline, so follow-up questions cannot be answered right now. The factors above are rule-based.'
      : null,
    provider: 'rules',
    model: 'client-side fallback',
    fallback_reason: reason ?? null,
    generated_at: new Date().toISOString(),
  };
}

export function XAIPanel({ selected, metrics, isProjectionRun }: XAIPanelProps) {
  const [result, setResult] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const explain = useCallback(async () => {
    if (!selected) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const req = buildRequest(selected, metrics);
    setLoading(true);
    try {
      const res = await fetchExplanation(req, ctrl.signal);
      if (!ctrl.signal.aborted) setResult(res);
    } catch (err) {
      if (ctrl.signal.aborted) return;
      // Backend /api/explain not available, fall back to local rules
      const local = generateLocalExplanation(req, `Backend request failed: ${err instanceof Error ? err.message : String(err)}`.slice(0, 300));
      setResult(local);
    } finally {
      if (!ctrl.signal.aborted) setLoading(false);
    }
  }, [selected, metrics]);

  useEffect(() => {
    if (isProjectionRun && selected) {
      explain();
    } else {
      abortRef.current?.abort();
      setResult(null);
    }
    return () => abortRef.current?.abort();
  }, [isProjectionRun, selected, explain]);

  // Reserved space: always visible, explains what will appear once a panchayat is downscaled
  if (!isProjectionRun || !selected) {
    return (
      <div className="advisory-panel">
        <div className="flex items-center justify-between px-5 py-3" style={{ borderBottom: '1px solid var(--hairline)' }}>
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4" style={{ color: 'var(--copper)' }} />
            <span className="label-sm">Explainable AI · why this number</span>
          </div>
          <span
            className="text-[0.55rem] font-mono uppercase tracking-wider px-2 py-0.5"
            style={{ borderRadius: '999px', color: 'var(--muted)', border: '1px solid var(--hairline-strong)' }}
          >
            Groq LLM
          </span>
        </div>
        <div className="px-5 py-5 flex items-start gap-3">
          <Info className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: 'var(--muted)' }} />
          <div>
            <p className="text-xs font-semibold mb-1" style={{ color: 'var(--text)' }}>
              {selected ? 'Downscale this panchayat to get an explanation' : 'No prediction to explain yet'}
            </p>
            <p className="text-[0.65rem] leading-relaxed" style={{ color: 'var(--muted)' }}>
              After you get downscaled rainfall, a language model explains what drove the estimate
              (IMD rainfall, terrain, moisture, temperature), how confident it is, and answers follow-up questions.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const isLLM = result?.provider && result.provider !== 'rules';

  return (
    <div className="advisory-panel">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3" style={{ borderBottom: '1px solid var(--hairline)' }}>
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4" style={{ color: 'var(--copper)' }} />
          <span className="label-sm">Explainable AI · why this number</span>
        </div>
        <div className="flex items-center gap-2">
          {result && (
            <span
              className="text-[0.55rem] font-mono uppercase tracking-wider px-2 py-0.5"
              style={{
                borderRadius: '999px',
                color: isLLM ? 'var(--positive)' : 'var(--muted)',
                border: `1px solid ${isLLM ? 'rgba(94,140,106,0.35)' : 'var(--hairline-strong)'}`,
              }}
              title={isLLM ? `Generated by ${result.provider} · ${result.model}` : 'Groq LLM unreachable — showing rule-based explanation'}
            >
              {isLLM ? `${result.provider} · ${result.model}` : 'LLM offline · rules'}
            </span>
          )}
          <button
            type="button"
            onClick={() => explain()}
            disabled={loading}
            className="p-1 rounded-md transition-opacity hover:opacity-100 opacity-60 disabled:opacity-30"
            title="Regenerate explanation"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} style={{ color: 'var(--muted)' }} />
          </button>
        </div>
      </div>

      {result && !isLLM && result.fallback_reason && !result.fallback_reason.includes("GROQ_API_KEY is not set") && (
        <p className="px-5 pt-3 text-[0.6rem] font-mono" style={{ color: 'var(--muted)' }}>
          Why rules: {result.fallback_reason}
        </p>
      )}

      {/* Content */}
      <div className="px-5 py-4">
        {loading || !result ? (
          <div className="flex items-center gap-2 py-4 justify-center">
            <Loader2 className="w-4 h-4 animate-spin" style={{ color: 'var(--muted)' }} />
            <span className="text-xs" style={{ color: 'var(--muted)' }}>Asking the model to explain its estimate...</span>
          </div>
        ) : (
          <div className="grid md:grid-cols-2 gap-5">
            {/* Left: narrative + follow-up */}
            <div className="space-y-3">
              <p className="text-xs font-semibold" style={{ color: 'var(--text)' }}>{result.summary}</p>
              <p className="text-xs leading-relaxed whitespace-pre-line" style={{ color: 'var(--text-2)' }}>
                {result.explanation}
              </p>

              <div className="flex items-start gap-2 pl-3" style={{ borderLeft: '2px solid var(--copper)' }}>
                <div>
                  <p className="label-sm mb-0.5">Confidence · {result.confidence}</p>
                  <p className="text-[0.65rem] leading-relaxed" style={{ color: 'var(--muted)' }}>{result.confidence_note}</p>
                </div>
              </div>
            </div>

            {/* Right: factor attribution */}
            <div>
              <p className="label-sm mb-3">What drove the estimate</p>
              <ul className="space-y-3">
                {result.factors.map((f, i) => {
                  const s = EFFECT_STYLE[f.effect] ?? EFFECT_STYLE.neutral;
                  const w = Math.max(0, Math.min(1, f.weight));
                  return (
                    <li key={i}>
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <span className="text-xs" style={{ color: 'var(--text)' }}>{f.factor}</span>
                        <span className="flex items-center gap-1 text-[0.55rem] font-mono uppercase" style={{ color: s.color }}>
                          <s.Icon className="w-3 h-3" />
                          {s.label}
                        </span>
                      </div>
                      <div className="h-1.5 w-full rounded-full" style={{ background: 'var(--raised)' }}>
                        <div className="h-1.5 rounded-full" style={{ width: `${w * 100}%`, background: s.color }} />
                      </div>
                      <p className="text-[0.6rem] leading-relaxed mt-1" style={{ color: 'var(--muted)' }}>{f.detail}</p>
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <div className="flex items-start gap-2 pt-3 mt-4" style={{ borderTop: '1px solid var(--hairline)' }}>
          <Info className="w-3 h-3 mt-0.5 flex-shrink-0" style={{ color: 'var(--muted)' }} />
          <p className="text-[0.55rem] leading-relaxed" style={{ color: 'rgba(140,140,150,0.6)' }}>
            Explanations are written by a language model from the prediction inputs shown here. Factor weights are
            qualitative, not SHAP values. Accuracy is scored against CHIRPS v2.0 — a reference product, not ground truth.
          </p>
        </div>
      </div>
    </div>
  );
}