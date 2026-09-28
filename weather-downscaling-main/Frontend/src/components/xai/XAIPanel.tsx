'use client';

import type { Panchayat, Weather, ModelMetrics } from '../../types';
import { classifyRisk, RISK_LABEL, RISK_BG } from '../../types';
import { Brain, ChevronRight, Info } from 'lucide-react';

interface XAIPanelProps {
  panchayat: Panchayat;
  weather: Weather | null;
  metrics: ModelMetrics | null;
  loading?: boolean;
}

const CHANNEL_LABELS: Record<string, string> = {
  imd_rain:    'IMD Rainfall (coarse 0.25°)',
  dem:         'SRTM Elevation (DEM)',
  era5_t2m:    'ERA5-Land Temperature (T₂ₘ)',
  era5_t2m_max:'ERA5-Land Max Temperature',
  era5_dewp:   'ERA5-Land Dewpoint (Td)',
};

const METHOD_LABELS: Record<string, string> = {
  direct_grid:      'Grid cell centres fall directly inside the panchayat polygon.',
  area_weighted:    'Weighted average of overlapping 0.05° cells (small polygon).',
  nearest_fallback: 'Nearest grid cell centroid used (< 15 km fallback).',
};

function PlainXAI({ panchayat, weather, metrics }: Omit<XAIPanelProps, 'loading'>) {
  const risk = classifyRisk(panchayat.rainfall_mm);
  const improvement = metrics?.mae_improvement_pct;
  const elevation = weather?.elevation_m;
  const humidity = weather?.humidity_pct;

  // Rule-based plain-English explanation
  const explanationParts: string[] = [];
  explanationParts.push(
    `The U-Net model learned a spatial correction to the IMD bilinear baseline at 0.25° (~28 km) resolution, producing a 0.05° (~5 km) estimate for ${panchayat.panchayat_name}.`
  );
  if (elevation !== null && elevation !== undefined) {
    explanationParts.push(
      elevation > 500
        ? `Terrain elevation (${elevation} m) is significant — the model's DEM channel helps capture orographic enhancement, where rising air produces extra rainfall on windward slopes.`
        : `Terrain is relatively flat (${elevation} m) — elevation effects are minor here.`
    );
  }
  if (humidity !== null && humidity !== undefined) {
    explanationParts.push(
      humidity > 70
        ? `Atmospheric moisture is high (${humidity}% RH), which the model's ERA5-Land dewpoint channel uses as a proxy for available moisture for precipitation.`
        : `Atmospheric moisture is moderate (${humidity}% RH).`
    );
  }
  if (improvement) {
    explanationParts.push(
      `On the unseen 2022 monsoon test set, this model achieved ${improvement}% lower MAE than the IMD bilinear baseline (6.45 mm vs 7.67 mm), scored against CHIRPS v2.0 as the reference.`
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {explanationParts.map((p, i) => (
        <p key={i} className="text-sm text-text-2 leading-relaxed">{p}</p>
      ))}
    </div>
  );
}

export function XAIPanel({ panchayat, weather, metrics, loading }: XAIPanelProps) {
  const risk = classifyRisk(panchayat.rainfall_mm);

  return (
    <div className="card">
      {/* Header */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-10 h-10 rounded-lg bg-navy/10 flex items-center justify-center">
          <Brain className="w-5 h-5 text-navy" />
        </div>
        <div>
          <h3 className="heading-md">Explainable AI</h3>
          <p className="text-xs text-text-2">How was this prediction computed?</p>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Left: computation chain */}
        <div>
          <p className="label-sm mb-3">Computation Chain</p>
          <div className="flex flex-col gap-2">
            {/* Step 1 */}
            <div className="flex items-start gap-3 p-3 bg-bg rounded-lg">
              <span className="step-badge text-xs" style={{ width: '1.6rem', height: '1.6rem', fontSize: '0.6rem' }}>01</span>
              <div>
                <p className="text-sm font-semibold text-text">IMD Bilinear Baseline</p>
                <p className="text-xs text-text-2">0.25° raw rainfall bilinearly upsampled to 0.05° grid — the starting point.</p>
              </div>
            </div>
            <ChevronRight className="w-4 h-4 text-muted self-center" />
            {/* Step 2 */}
            <div className="flex items-start gap-3 p-3 bg-bg rounded-lg">
              <span className="step-badge text-xs" style={{ width: '1.6rem', height: '1.6rem', fontSize: '0.6rem' }}>02</span>
              <div>
                <p className="text-sm font-semibold text-text">U-Net Residual Correction</p>
                <p className="text-xs text-text-2">~150k-param model predicts spatial correction using DEM + ERA5-Land context. Output = baseline + residual.</p>
              </div>
            </div>
            <ChevronRight className="w-4 h-4 text-muted self-center" />
            {/* Step 3 */}
            <div className="flex items-start gap-3 p-3 bg-bg rounded-lg">
              <span className="step-badge text-xs" style={{ width: '1.6rem', height: '1.6rem', fontSize: '0.6rem' }}>03</span>
              <div>
                <p className="text-sm font-semibold text-text">Panchayat Aggregation</p>
                <p className="text-xs text-text-2">
                  {METHOD_LABELS[panchayat.mapping_method] ?? panchayat.mapping_method}
                  {' '}({panchayat.n_cells} cell{panchayat.n_cells !== 1 ? 's' : ''})
                </p>
              </div>
            </div>
          </div>

          {/* Model inputs */}
          <div className="mt-4">
            <p className="label-sm mb-2">Model Inputs (Channels)</p>
            <div className="flex flex-col gap-1">
              {(metrics?.channels ?? ['imd_rain', 'dem', 'era5_t2m', 'era5_t2m_max', 'era5_dewp']).map(ch => (
                <div key={ch} className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-navy flex-shrink-0" />
                  <p className="text-xs text-text-2">{CHANNEL_LABELS[ch] ?? ch}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: plain English + metrics */}
        <div>
          <p className="label-sm mb-3">Plain English Explanation</p>
          {loading ? (
            <div className="flex flex-col gap-2">
              {[...Array(3)].map((_, i) => <div key={i} className="skeleton h-4 w-full" />)}
            </div>
          ) : (
            <PlainXAI panchayat={panchayat} weather={weather} metrics={metrics} />
          )}

          {/* Model performance */}
          {metrics && (
            <div className="mt-5 p-3 bg-bg rounded-lg">
              <p className="label-sm mb-2">Model Performance (2022 Test Set)</p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <p className="text-muted">MAE</p>
                  <p className="font-mono font-semibold text-text">{metrics.selected_model.test_mae_mm} mm</p>
                </div>
                <div>
                  <p className="text-muted">vs Baseline</p>
                  <p className="font-mono font-semibold text-primary">
                    {metrics.mae_improvement_pct}% lower ↓
                  </p>
                </div>
                <div>
                  <p className="text-muted">Correlation</p>
                  <p className="font-mono font-semibold text-text">{metrics.selected_model.test_correlation}</p>
                </div>
                <div>
                  <p className="text-muted">Params</p>
                  <p className="font-mono font-semibold text-text">{metrics.n_parameters}</p>
                </div>
              </div>
            </div>
          )}

          {/* Reference note */}
          <div className="mt-3 flex items-start gap-2 p-2.5 bg-amber-50 rounded-lg border border-amber-100">
            <Info className="w-3.5 h-3.5 text-amber-600 mt-0.5 flex-shrink-0" />
            <p className="text-xs text-amber-700">
              Scored against CHIRPS v2.0 — a reference product, not absolute ground truth.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
