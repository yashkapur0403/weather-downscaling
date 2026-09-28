'use client';

import { Database, Cpu, Map, Brain, X } from 'lucide-react';

const STEPS = [
  {
    n: '01',
    icon: <Database className="w-5 h-5" />,
    title: 'IMD Coarse Input',
    desc: 'Daily rainfall from IMD at 0.25° (~28 km) is bilinearly upsampled to 0.05° as the starting baseline.',
  },
  {
    n: '02',
    icon: <Cpu className="w-5 h-5" />,
    title: 'U-Net Residual Correction',
    desc: 'A ~150k-param residual U-Net learns the spatial correction using SRTM terrain elevation and ERA5-Land atmospheric context.',
  },
  {
    n: '03',
    icon: <Map className="w-5 h-5" />,
    title: 'Panchayat Aggregation',
    desc: 'The 0.05° field is spatially joined to 8,236 LGD Gram Panchayat polygons using area-weighted averaging.',
  },
  {
    n: '04',
    icon: <Brain className="w-5 h-5" />,
    title: 'Explainable AI Output',
    desc: 'Each prediction is fully traceable — model inputs, mapping method, and performance metrics are shown alongside the value.',
  },
];

interface Props {
  onClose: () => void;
}

export function HowItWorksModal({ onClose }: Props) {
  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.4)', backdropFilter: 'blur(8px)' }} onClick={onClose}>
      <div className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto rounded-xl" style={{ background: 'var(--surface)', border: '1px solid var(--border)', boxShadow: '0 25px 60px rgba(0,0,0,0.15)' }} onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b" style={{ borderColor: 'var(--border)' }}>
          <div>
            <h2 className="font-bold text-lg" style={{ color: 'var(--text)' }}>How It Works</h2>
            <p className="text-xs mt-1" style={{ color: 'var(--muted)' }}>A transparent, reproducible pipeline from IMD coarse data to panchayat-level downscaled estimates.</p>
          </div>
          <button onClick={onClose} className="w-8 h-8 rounded-lg flex items-center justify-center transition-colors cursor-pointer border-none bg-transparent" style={{ color: 'var(--muted)' }} onMouseEnter={e => e.currentTarget.style.color = 'var(--text)' as any} onMouseLeave={e => e.currentTarget.style.color = 'var(--muted)' as any}>
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Steps */}
        <div className="px-6 py-5 space-y-4">
          {STEPS.map(step => (
            <div key={step.n} className="flex items-start gap-4 p-4 rounded-lg" style={{ background: 'var(--bg)' }}>
              <div className="flex items-center gap-3 flex-shrink-0">
                <span className="step-badge" style={{ width: '2rem', height: '2rem', fontSize: '0.65rem' }}>{step.n}</span>
                <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'var(--surface)', color: 'var(--navy)' }}>
                  {step.icon}
                </div>
              </div>
              <div>
                <h3 className="font-bold text-sm mb-1" style={{ color: 'var(--text)' }}>{step.title}</h3>
                <p className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>{step.desc}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-[0.65rem] text-center" style={{ color: 'var(--muted)' }}>
            IMD rainfall downscaled from 28 km to 5 km using terrain and atmospheric context. Search 8,236 Gram Panchayats across 6 Indian states.
          </p>
        </div>
      </div>
    </div>
  );
}
