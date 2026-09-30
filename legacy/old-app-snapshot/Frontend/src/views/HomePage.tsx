'use client';

import { useState, useEffect, useCallback } from 'react';
import { LeftSidebar } from '../components/sidebar/LeftSidebar';
import { RightSidebar } from '../components/sidebar/RightSidebar';
import { MapView } from '../components/map/MapView';
import { queryWeather, fetchMetrics, geocodePanchayat } from '../api/backend';
import type { Panchayat, ModelMetrics, QueryRequest, QueryResponse } from '../types';
import { AdvisoryPanel } from '../components/advisory/AdvisoryPanel';
import { Database, Cpu, Map, Brain, Info } from 'lucide-react';

const HOW_IT_WORKS_STEPS = [
  { n: '01', icon: <Database className="w-5 h-5" />, title: 'IMD Coarse Input', desc: 'Daily rainfall from IMD at 0.25° (~28 km) is bilinearly upsampled to 0.05° as the starting baseline.' },
  { n: '02', icon: <Cpu className="w-5 h-5" />, title: 'U-Net Residual Correction', desc: 'A ~150k-param residual U-Net learns the spatial correction using SRTM terrain elevation and ERA5-Land atmospheric context.' },
  { n: '03', icon: <Map className="w-5 h-5" />, title: 'Panchayat Aggregation', desc: 'The 0.05° field is spatially joined to 8,236 LGD Gram Panchayat polygons using area-weighted averaging.' },
  { n: '04', icon: <Brain className="w-5 h-5" />, title: 'Explainable AI Output', desc: 'Each prediction is fully traceable — model inputs, mapping method, and performance metrics are shown alongside the value.' },
];

export function HomePage() {
  const [selected, setSelected] = useState<Panchayat | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const [isProjectionRun, setIsProjectionRun] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'unavailable'>('loading');

  useEffect(() => {
    setLoadingMetrics(true);
    fetchMetrics()
      .then(setMetrics)
      .catch(() => {})
      .finally(() => setLoadingMetrics(false));
  }, []);



  const handleSelect = useCallback(async (p: Panchayat) => {
    try {
      const coordinates = await geocodePanchayat(p);
      p = { ...p, ...coordinates };
    } catch (error) {
      console.warn('Location geocoding failed; keeping the search result.', error);
    }
    setSelected(p);
    setIsProjectionRun(false);
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!selected) return;
    setIsAnalyzing(true);
    
    try {
      // Use the new queryWeather API to get predictions
      const request: QueryRequest = {
        panchayat_name: selected.panchayat_name,
        block_name: selected.block_name,
        district: selected.district,
        state: selected.state,
        lat: selected.lat || undefined,
        lon: selected.lon || undefined,
        date: selected.date || '2022-07-10',
        requested_metrics: ['rainfall', 'temperature', 'humidity', 'elevation'],
        raw_location_text: `${selected.panchayat_name}, ${selected.district}`,
      };
      
      const response: QueryResponse = await queryWeather(request);
      setModelStatus(response.model_status);
      
      // Update the selected panchayat with the prediction data
      setSelected({
        ...selected,
        rainfall_mm: response.prediction.rainfall_mm || selected.rainfall_mm,
        temperature_c: response.prediction.temperature_c ?? selected.temperature_c,
        humidity_pct: response.prediction.humidity_pct ?? selected.humidity_pct,
        elevation_m: response.prediction.elevation_m ?? selected.elevation_m,
      });
      
      setIsProjectionRun(true);
    } catch (error) {
      console.error('Projection failed:', error);
      setModelStatus('unavailable');
    } finally {
      setIsAnalyzing(false);
    }
  }, [selected]);



  return (
    <div className="flex flex-col min-h-screen" style={{ background: 'var(--canvas)' }}>
      {/* Dashboard area */}
      <div id="dashboard" className="flex flex-1" style={{ height: 'calc(100vh - 3rem)' }}>
        {/* Left Sidebar */}
        <aside className="dashboard-left-panel">
          <LeftSidebar
            onSelect={handleSelect}
            onGenerate={handleGenerate}
            isLoading={isAnalyzing}
            hasSelected={selected !== null}
            selected={selected}
            modelStatus={modelStatus}
          />
        </aside>

        {/* Center Map + Advisory */}
        <main className="dashboard-map-area">
          <div className="dashboard-map-container">
            <MapView selected={isProjectionRun ? selected : null} />
          </div>
          <AdvisoryPanel
            selected={selected}
            isProjectionRun={isProjectionRun}
          />
        </main>

        {/* Right Sidebar — only after projection */}
        {isProjectionRun && (
          <aside className="dashboard-right-panel animate-[fadeIn_0.3s_ease_forwards]">
            <RightSidebar
              selected={selected}
              metrics={metrics}
              loadingMetrics={loadingMetrics}
            />
          </aside>
        )}
      </div>

      {/* ── HOW IT WORKS ─────────────────────────────────────────────────── */}
      <section id="how-it-works" className="bottom-sections py-16">
        <div className="max-w-3xl mx-auto px-6">
          <div className="text-center mb-10">
            <h2 className="font-bold text-xl mb-2" style={{ color: 'var(--text)' }}>How It Works</h2>
            <p className="text-sm" style={{ color: 'var(--muted)' }}>
              A transparent, reproducible pipeline from IMD coarse data to panchayat-level downscaled estimates.
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {HOW_IT_WORKS_STEPS.map(step => (
              <div key={step.n} className="p-5" style={{ background: 'var(--raised)', border: '1px solid var(--hairline)', borderRadius: '4px' }}>
                <div className="flex items-center gap-3 mb-3">
                  <span className="font-mono text-[0.65rem] font-bold" style={{ color: 'var(--muted)' }}>{step.n}</span>
                  <div className="w-8 h-8 rounded flex items-center justify-center" style={{ background: 'var(--panel)', color: 'var(--reference)' }}>
                    {step.icon}
                  </div>
                </div>
                <h3 className="font-bold text-sm mb-1" style={{ color: 'var(--text)' }}>{step.title}</h3>
                <p className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── MODEL ────────────────────────────────────────────────────────── */}
      <section id="model-info" className="bottom-sections py-16">
        <div className="max-w-3xl mx-auto px-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded flex items-center justify-center" style={{ background: 'var(--raised)', color: 'var(--copper)' }}>
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h2 className="font-bold text-lg" style={{ color: 'var(--text)' }}>Selected Model — U-Net + DEM + ERA5-Land</h2>
              <p className="text-xs" style={{ color: 'var(--muted)' }}>Validated on unseen 2022 monsoon data (Jun–Sep, 122 days)</p>
            </div>
          </div>

          {loadingMetrics ? (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[...Array(4)].map((_, i) => <div key={i} className="skeleton h-16 rounded" />)}
            </div>
          ) : metrics ? (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
                {[
                  { label: 'Test MAE', value: `${metrics.selected_model.test_mae_mm} mm` },
                  { label: 'Test RMSE', value: `${metrics.selected_model.test_rmse_mm} mm` },
                  { label: 'Correlation', value: `${metrics.selected_model.test_correlation}` },
                  { label: 'MAE Improvement', value: `${metrics.mae_improvement_pct}%`, accent: true },
                ].map(m => (
                  <div key={m.label} className="p-3 text-center" style={{ background: 'var(--raised)', border: '1px solid var(--hairline)', borderRadius: '4px' }}>
                    <p className="label-sm mb-1">{m.label}</p>
                    <p className="font-mono font-bold text-lg" style={{ color: m.accent ? 'var(--heat-3)' : 'var(--text)' }}>{m.value}</p>
                  </div>
                ))}
              </div>

              {/* Ablation table */}
              <div className="overflow-x-auto" style={{ border: '1px solid var(--hairline)', borderRadius: '4px' }}>
                <table className="w-full text-xs">
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--hairline)' }}>
                      <th className="text-left py-2 px-3 label-sm">Model</th>
                      <th className="text-right py-2 px-3 label-sm">MAE (mm)</th>
                      <th className="text-right py-2 px-3 label-sm">RMSE (mm)</th>
                      <th className="text-right py-2 px-3 label-sm">Corr</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.test_variants.map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid var(--hairline)' }}>
                        <td className="py-2 px-3 font-medium" style={{ color: 'var(--text)' }}>{row.model}</td>
                        <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--text-2)' }}>{row.MAE_mm}</td>
                        <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--text-2)' }}>{row.RMSE_mm}</td>
                        <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--text-2)' }}>{row.corr}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 p-4" style={{ background: 'var(--raised)', border: '1px solid var(--hairline)', borderRadius: '4px' }}>
                <p className="label-sm mb-2">Model Details</p>
                <div className="grid grid-cols-2 gap-2 text-xs" style={{ color: 'var(--text-2)' }}>
                  <p>Parameters: ~150k</p>
                  <p>Architecture: Residual U-Net</p>
                  <p>Temporal split: 2019–20 train</p>
                  <p>Val: 2021 · Test: 2022</p>
                  <p>Channels: IMD rain + DEM + ERA5</p>
                  <p>Reference: CHIRPS v2.0</p>
                </div>
              </div>
            </>
          ) : (
            <div className="p-4" style={{ background: 'rgba(176,141,87,0.05)', border: '1px solid rgba(176,141,87,0.2)', borderRadius: '4px' }}>
              <p className="text-sm" style={{ color: 'var(--copper)' }}>Backend not connected — start FastAPI server to see live metrics.</p>
            </div>
          )}

          <div className="mt-6 flex items-start gap-2 p-3" style={{ background: 'var(--raised)', border: '1px solid var(--hairline)', borderRadius: '4px' }}>
            <Info className="w-3 h-3 mt-0.5 flex-shrink-0" style={{ color: 'var(--muted)' }} />
            <p className="text-[0.65rem] leading-relaxed" style={{ color: 'var(--muted)' }}>
              Scored against CHIRPS v2.0 — a reference product, not absolute ground truth. No accuracy claims made.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 text-center" style={{ borderTop: '1px solid var(--hairline)' }}>
        <p className="text-[0.6rem] font-mono uppercase tracking-widest" style={{ color: 'var(--muted)', opacity: 0.5 }}>
          Obsidian · 6 states · 8,236 panchayats · 2022-07-10 dataset
        </p>
      </footer>
    </div>
  );
}
