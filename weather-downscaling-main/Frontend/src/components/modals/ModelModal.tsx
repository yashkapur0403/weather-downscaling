'use client';

import { useEffect, useState } from 'react';
import { Cpu, X } from 'lucide-react';
import { fetchMetrics } from '../../api/backend';
import type { ModelMetrics } from '../../types';

interface Props {
  onClose: () => void;
}

export function ModelModal({ onClose }: Props) {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMetrics()
      .then(setMetrics)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.4)', backdropFilter: 'blur(8px)' }} onClick={onClose}>
      <div className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto rounded-xl" style={{ background: 'var(--surface)', border: '1px solid var(--border)', boxShadow: '0 25px 60px rgba(0,0,0,0.15)' }} onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b" style={{ borderColor: 'var(--border)' }}>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0" style={{ background: 'rgba(30,58,138,0.1)' }}>
              <Cpu className="w-5 h-5" style={{ color: 'var(--navy)' }} />
            </div>
            <div>
              <h2 className="font-bold text-lg" style={{ color: 'var(--text)' }}>Selected Model — U-Net + DEM + ERA5-Land</h2>
              <p className="text-xs mt-0.5" style={{ color: 'var(--muted)' }}>Validated on unseen 2022 monsoon data (Jun–Sep, 122 days)</p>
            </div>
          </div>
          <button onClick={onClose} className="w-8 h-8 rounded-lg flex items-center justify-center transition-colors cursor-pointer border-none bg-transparent" style={{ color: 'var(--muted)' }} onMouseEnter={e => e.currentTarget.style.color = 'var(--text)' as any} onMouseLeave={e => e.currentTarget.style.color = 'var(--muted)' as any}>
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-5">
          {loading ? (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="skeleton h-16 rounded-lg" />
              ))}
            </div>
          ) : metrics ? (
            <>
              {/* Key metrics */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
                <div className="p-3 rounded-lg text-center" style={{ background: 'var(--bg)' }}>
                  <p className="label-sm mb-1">Test MAE</p>
                  <p className="font-mono font-bold text-lg" style={{ color: 'var(--text)' }}>{metrics.selected_model.test_mae_mm} mm</p>
                </div>
                <div className="p-3 rounded-lg text-center" style={{ background: 'var(--bg)' }}>
                  <p className="label-sm mb-1">Test RMSE</p>
                  <p className="font-mono font-bold text-lg" style={{ color: 'var(--text)' }}>{metrics.selected_model.test_rmse_mm} mm</p>
                </div>
                <div className="p-3 rounded-lg text-center" style={{ background: 'var(--bg)' }}>
                  <p className="label-sm mb-1">Correlation</p>
                  <p className="font-mono font-bold text-lg" style={{ color: 'var(--text)' }}>{metrics.selected_model.test_correlation}</p>
                </div>
                <div className="p-3 rounded-lg text-center" style={{ background: 'var(--bg)' }}>
                  <p className="label-sm mb-1">MAE Improvement</p>
                  <p className="font-mono font-bold text-lg" style={{ color: 'var(--primary)' }}>{metrics.mae_improvement_pct}%</p>
                </div>
              </div>

              {/* Ablation table */}
              <div className="mb-4">
                <p className="label-sm mb-3">Ablation Results (Test Set)</p>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border)' }}>
                        <th className="text-left py-2 px-3 label-sm">Model</th>
                        <th className="text-right py-2 px-3 label-sm">MAE (mm)</th>
                        <th className="text-right py-2 px-3 label-sm">RMSE (mm)</th>
                        <th className="text-right py-2 px-3 label-sm">Corr</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.test_variants.map((row, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                          <td className="py-2 px-3 font-medium" style={{ color: 'var(--text)' }}>{row.model}</td>
                          <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--text-2)' }}>{row.MAE_mm}</td>
                          <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--text-2)' }}>{row.RMSE_mm}</td>
                          <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--text-2)' }}>{row.corr}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Model details */}
              <div className="p-4 rounded-lg" style={{ background: 'var(--bg)' }}>
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
            <div className="p-4 rounded-lg" style={{ background: '#FEF3C7', border: '1px solid #FDE68A' }}>
              <p className="text-sm" style={{ color: '#92400E' }}>Backend not connected — start FastAPI server to see live metrics.</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-[0.65rem] text-center" style={{ color: 'var(--muted)' }}>
            Scored against CHIRPS v2.0 — a reference product, not absolute ground truth · No accuracy claims made
          </p>
        </div>
      </div>
    </div>
  );
}
