'use client';

import { useState } from 'react';
import Link from 'next/link';
import {
  ArrowRight,
  Brain,
  Cpu,
  Database,
  Layers,
  Map,
  Minus,
  Mountain,
  Plus,
  Thermometer,
  TriangleAlert,
  CloudRain,
  Activity,
  GitBranch,
  Droplets,
  Info,
  Search,
  Sprout,
} from 'lucide-react';
import { LandingNav } from '../components/nav/LandingNav';

const GITHUB_URL = 'https://github.com/yashkapur0403/weather-downscaling';

/* ── Hero stats ──────────────────────────────────────────────────────────── */
const HERO_STATS = [
  { label: 'Output grid', value: '0.05°', note: '~5 km', icon: <Map className="w-4 h-4" /> },
  { label: 'Monsoon days', value: '610', note: 'Jun–Sep 2018–22', icon: <CloudRain className="w-4 h-4" /> },
  { label: 'Land cells', value: '47,250', note: 'fine grid', icon: <Layers className="w-4 h-4" /> },
  { label: 'Parameters', value: '~150k', note: 'residual U-Net', icon: <Cpu className="w-4 h-4" /> },
];

/* ── The resolution ladder ───────────────────────────────────────────────── */
const LADDER = [
  { tag: 'INPUT',  title: 'IMD 0.25°',     value: '28 km',        note: 'Gridded daily rainfall — the coarse product that actually exists operationally.' },
  { tag: 'REFINE', title: 'Residual U-Net', value: '+ DEM + ERA5', note: 'Learns the terrain- and atmosphere-driven correction to the bilinear baseline.' },
  { tag: 'OUTPUT', title: '0.05°',          value: '5 km',         note: 'Scored against CHIRPS v2.0 — a reference product, not ground truth.' },
];

/* ── Pipeline steps ──────────────────────────────────────────────────────── */
const PIPELINE = [
  { n: '01', icon: <Database className="w-5 h-5" />, title: 'IMD Coarse Input',        desc: 'Daily rainfall from IMD at 0.25° (~28 km) is bilinearly upsampled to 0.05° as the starting baseline.' },
  { n: '02', icon: <Cpu       className="w-5 h-5" />, title: 'U-Net Residual Correction', desc: 'A ~150k-param residual U-Net learns the spatial correction using SRTM terrain elevation and ERA5-Land atmospheric context.' },
  { n: '03', icon: <Map       className="w-5 h-5" />, title: 'Panchayat Aggregation',   desc: 'The 0.05° field is spatially joined to 86,103 LGD Gram Panchayat polygons using area-weighted averaging.' },
  { n: '04', icon: <Brain     className="w-5 h-5" />, title: 'Explainable AI Output',   desc: 'Each prediction is fully traceable — model inputs, mapping method, and performance metrics are shown alongside the value.' },
];

/* ── Data sources ────────────────────────────────────────────────────────── */
const SOURCES = [
  'IMD 0.25° rainfall',
  'CHIRPS v2.0 (0.05°)',
  'SRTM 30 m (Terrarium)',
  'ERA5-Land (Open-Meteo)',
  'GADM 4.1 admin',
  'SoilGrids v2.0',
  'NOAA CDR VIIRS NDVI',
  'ESA WorldCover 2021',
  'LGD Gram Panchayats (CC0)',
];

/* ── Ablation results ────────────────────────────────────────────────────── */
const RESULTS = [
  { model: 'A — Bilinear IMD baseline',       valMae: '6.72', valCorr: '0.257', testMae: '7.67',  testCorr: '0.304', f1: '0.217' },
  { model: 'B2 — Bias-corrected bilinear',    valMae: '9.69', valCorr: '0.400', testMae: '10.15', testCorr: '0.436', f1: '0.365', flagged: true },
  { model: 'B — U-Net, rainfall only',        valMae: '5.35', valCorr: '0.283', testMae: '6.63',  testCorr: '0.304', f1: '0.055' },
  { model: 'C — U-Net + DEM',                 valMae: '5.39', valCorr: '0.289', testMae: '6.58',  testCorr: '0.314', f1: '0.061' },
  { model: 'Cw — U-Net + DEM, weighted loss', valMae: '6.04', valCorr: '0.372', testMae: '6.91',  testCorr: '0.419', f1: '0.312', flagged: true },
  { model: 'D — U-Net + DEM + ERA5-Land',     valMae: '5.41', valCorr: '0.331', testMae: '6.45',  testCorr: '0.376', f1: '0.146', selected: true },
];

/* ── Key numbers strip ───────────────────────────────────────────────────── */
const KEY_NUMBERS = [
  { value: '28 → 5 km', label: 'Resolution gain',   note: '0.25° → 0.05°' },
  { value: '86,103',    label: 'Panchayats mapped', note: '98.1% of 87,735' },
  { value: '15',        label: 'States covered',    note: 'Deccan region' },
  { value: '610',       label: 'Monsoon days',      note: 'Jun–Sep, 2018–22' },
  { value: '~150k',     label: 'Parameters',        note: 'residual U-Net' },
  { value: '−15.9%',    label: 'MAE vs baseline',   note: '2022 test season' },
];

/* ── Why 28 km is not enough ─────────────────────────────────────────────── */
const PROBLEM = [
  { big: '1 value',  title: 'One number per ~28 km cell', body: "IMD's operational grid gives a single rainfall value for an area far larger than most villages. Hills, valleys and rain shadows are averaged away." },
  { big: '25 values', title: 'Inside the same area',       body: 'At 0.05° that one cell splits into about 25 fine cells, so a windward slope and a leeward valley can finally get different estimates.' },
  { big: '86,103',   title: 'Panchayat-level answers',     body: 'Fine cells are joined to LGD boundaries, so a farmer or officer searches by Gram Panchayat instead of by grid coordinates.' },
];

/* ── Panchayat mapping methods (Layer 2) ─────────────────────────────────── */
const MAPPING_METHODS = [
  { title: 'Direct grid',      tag: 'Highest confidence', body: 'Fine-cell centres fall inside the panchayat polygon, so the value is read straight from those cells.' },
  { title: 'Area-weighted',    tag: 'Small polygons',     body: 'For small panchayats, a weighted average of every overlapping 0.05° cell, weighted by overlap area.' },
  { title: 'Nearest fallback', tag: 'Lower confidence',   body: 'Where no cell overlaps, the nearest cell centroid (within 15 km) is used and the value is flagged.' },
];

const COVERAGE = [
  { value: '87,735', label: 'Panchayats in region' },
  { value: '86,103', label: 'Mapped' },
  { value: '98.1%',  label: 'Coverage' },
  { value: '122',    label: 'Days, 2022 monsoon' },
];

/* ── What the dashboard does ─────────────────────────────────────────────── */
const DASH_FEATURES = [
  { icon: <Search    className="w-5 h-5" />, title: 'Find a panchayat', body: 'Search by name, block, district or state across 86,103 Gram Panchayats.' },
  { icon: <CloudRain className="w-5 h-5" />, title: 'Downscale',        body: 'Get the 5 km rainfall estimate with temperature, humidity and elevation for that place.' },
  { icon: <Sprout    className="w-5 h-5" />, title: 'Field advisory',   body: 'Crop- and growth-stage-specific guidance drawn from the rainfall estimate.' },
  { icon: <Brain     className="w-5 h-5" />, title: 'Explain the number', body: 'A language model explains what drove the estimate, how confident it is, and answers follow-up questions.' },
];

/* ── Explainable AI panel contents ───────────────────────────────────────── */
const XAI_PARTS = [
  { title: 'Summary',         body: 'One sentence stating the estimate for the place and date.' },
  { title: 'What drove it',   body: 'IMD rainfall, terrain, moisture and temperature, each marked as raising, lowering or barely changing the value.' },
  { title: 'Confidence',      body: 'Low, medium or high, tied to how the panchayat was mapped (direct, area-weighted or nearest fallback).' },
  { title: 'Ask a follow-up', body: 'Type a question such as "why is it higher than nearby areas?" and get an answer grounded in the same inputs.' },
];

/* ── Honest limitations ──────────────────────────────────────────────────── */
const LIMITATIONS = [
  { title: 'Not a forecast',        body: 'It refines existing coarse rainfall. It does not predict future weather.' },
  { title: 'Reference, not truth',  body: 'Scored against CHIRPS v2.0. IMD and CHIRPS disagree at daily scale, so part of every error is their disagreement.' },
  { title: 'Pilot-scale evidence',  body: 'Reported metrics come from the Western Ghats pilot; Deccan-wide training is still pending.' },
  { title: 'Monsoon only',          body: 'June–September data only. There is nothing for the winter season.' },
  { title: 'Heavy rain is hard',    body: 'MAE-trained models under-detect ≥25 mm days. The weighted variant trades about 0.7 mm of MAE for better detection.' },
  { title: 'No uncertainty band',   body: 'Every value is a single deterministic estimate, not a probability range.' },
];

/* ── FAQ ─────────────────────────────────────────────────────────────────── */
const FAQ: { q: string; a: string[] }[] = [
  {
    q: 'Is this a weather forecast?',
    a: [
      'No. Obsidian is not a forecasting system — it is a downscaling engine. It takes a coarse daily rainfall field that already exists (IMD at 0.25°) and refines it to 0.05° using terrain and atmospheric context. Everything served here is historical refinement, not a prediction of future weather.',
      'For operational use, the same Layer-1 model would consume IMD Block forecasts as its coarse input, and their forecast error would propagate through the pipeline.',
    ],
  },
  {
    q: 'What does "downscaling" actually mean here?',
    a: [
      'Agricultural advisories need rainfall at roughly 5 km, but the operational gridded product is 0.25° (~28 km). Layer 1 learns the mapping: coarse IMD rainfall + SRTM elevation + ERA5-Land daily mean/max temperature and dewpoint go into a small residual U-Net, which emits a 0.05° rainfall field.',
      'The network learns the residual on top of a bilinearly-upsampled baseline rather than the rainfall itself. By construction of that parameterization it can never do worse than the baseline.',
    ],
  },
  {
    q: 'Is CHIRPS the ground truth?',
    a: [
      'No, and this distinction matters. CHIRPS v2.0 at 0.05° is used as the fine-resolution reference product, not as absolute truth. IMD and CHIRPS disagree substantially at daily scale — the domain-mean daily coarse correlation on the Deccan build is about 0.356 — so part of every error term is their disagreement rather than model error.',
      'They agree far better at weekly and monthly aggregates, which is where the value of downscaling is clearest. No accuracy claims are made against real-world rainfall.',
    ],
  },
  {
    q: 'Which region and period does the dataset cover?',
    a: [
      'The full Layer-1 dataset is built for the Deccan region — 11.5–25.5°N, 71.5–81.25°E — covering five monsoon seasons (2018–2022): 610 days and 47,250 land cells on the fine grid. It is monsoon-only (June–September); there is no winter data.',
    ],
  },
  {
    q: 'How is rainfall mapped from grid cells to a panchayat?',
    a: [
      'The 0.05° field is intersected with LGD Gram Panchayat polygons and aggregated by area-weighted averaging. The admin auxiliary layer already assigns every land cell a state, district and block, so block-level aggregation needs no additional GIS work; panchayat-level mapping adds the LGD boundary layer on top of that.',
      'Each output carries its cell count and mapping method (direct grid, area-weighted, or nearest fallback) so the aggregation behind any value is auditable.',
    ],
  },
  {
    q: 'Where do the panchayat boundaries come from?',
    a: [
      'From the Local Government Directory (LGD), Ministry of Panchayati Raj, distributed in a CC0 public-domain bundle. Because it is LGD-derived it carries real LGD codes, which makes it the authoritative tier.',
      'Non-LGD boundary sets from data.gov.in, Datameet or state GIS portals remain geometry-only substitutes and are not equivalent — they lack the LGD codes the mapping depends on.',
    ],
  },
  {
    q: 'How is the data split between training and testing?',
    a: [
      'Purely by time, with no leakage: training uses 2018 + 2019 + 2020 (366 days), validation uses 2021 (122 days) and testing uses 2022 (122 days). Normalization statistics are computed from the training years only.',
      'Model selection uses the lowest validation MAE, where rows within 0.1 mm are treated as tied and the tie is broken by validation correlation. The test set is never used for tuning or model selection.',
    ],
  },
  {
    q: 'Does anything get imputed or filled in?',
    a: [
      'No. An IMD coarse cell counts as land only if at least 50% of aligned days have valid rainfall. Fine pixels enter the target and mask only if the parent coarse cell is land and CHIRPS is valid on every aligned day, so the coastal strip within one CHIRPS cell of ocean is structurally missing and is excluded rather than filled — 44,243 of 47,250 target-valid fine cells qualify, and the remaining ~3,007 are that strip.',
      'Inputs keep filled values over sea so convolutions stay finite, but no target, sample day or metric ever uses sea. Elsewhere, soil layers retain 8.8–9.4% genuine SoilGrids nulls and NDVI has ~8.0% monsoon cloud gaps — neither is imputed, so downstream statistics must be NaN-aware.',
    ],
  },
  {
    q: 'Why is the model so small, and why is there no uncertainty estimate?',
    a: [
      'The residual U-Net is deliberately small at ~150,000 parameters so it stays reproducible and runnable on modest hardware. Two honest consequences follow.',
      'Heavy rainfall remains hard for MAE-trained models; the heavy-rain-weighted loss variant trades roughly 0.7 mm of MAE for far better detection of ≥25 mm events, which is the more relevant trade for agriculture. And there is currently no probabilistic or uncertainty output — every value is a single deterministic estimate.',
    ],
  },
  {
    q: 'Can I reproduce the dataset and results myself?',
    a: [
      'Yes. Create a virtual environment, install requirements.txt, then run the region-aware pipeline in order: download_or_export.py (IMD, CHIRPS, DEM, ERA5), build_aux.py (admin, soil, NDVI, land cover), fetch_lgd_panchayats.py, preprocess.py, and finally verify_dataset.py — an end-to-end QA pass that exits non-zero on any problem.',
      'Every stage is cached and resumable, so re-runs are cheap. The exact training commands and the order to run them against the Deccan configuration are documented in HANDOVER.md.',
    ],
  },
  {
    q: 'What is deliberately not included?',
    a: [
      'Wind is not a model channel. The raw ERA5 cache now holds clean 10 m wind for all 610 days, but the frozen dataset was preprocessed with the five-channel baseline (imd_rain, dem, era5_t2m, era5_t2m_max, era5_dewp) and wind was intentionally kept out; adding it requires a documented re-preprocess.',
      'ERA5-Land soil moisture (0–7 cm) is built but not yet materialized — the free daily API quota ran out mid-build. It resumes additively on a later day and is intended as a Layer-3 advisory input, not a U-Net channel.',
      'ESA WorldCereal crop type was evaluated and excluded: the only no-authentication distribution is global, agro-ecological-zone-specific GeoTIFF volumes, and aligning it to our lattice is a standalone task outside the data freeze.',
    ],
  },
];

/* ── FAQ accordion component ─────────────────────────────────────────────── */
function FaqSection() {
  const [open, setOpen] = useState<number | null>(0);

  return (
    <div className="space-y-2">
      {FAQ.map((item, i) => {
        const isOpen = open === i;
        return (
          <div key={item.q} className="faq-item">
            <button
              className="faq-trigger"
              onClick={() => setOpen(isOpen ? null : i)}
              aria-expanded={isOpen}
            >
              <span
                className="step-badge flex-shrink-0"
                style={{
                  background: isOpen ? 'rgba(176, 141, 87, 0.2)' : undefined,
                  borderColor: isOpen ? 'var(--accent)' : undefined,
                }}
              >
                {isOpen ? <Minus className="w-3 h-3" /> : <Plus className="w-3 h-3" />}
              </span>
              <span
                className="text-sm font-semibold"
                style={{ color: isOpen ? 'var(--text)' : 'var(--text-2)' }}
              >
                {item.q}
              </span>
            </button>
            {isOpen && (
              <div className="faq-panel space-y-3">
                {item.a.map((para, j) => (
                  <p key={j}>{para}</p>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ── Section header ──────────────────────────────────────────────────────── */
function SectionHeader({ eyebrow, title, sub }: { eyebrow: string; title: string; sub: string }) {
  return (
    <div className="flex items-start gap-3 mb-8">
      <div className="w-0.5 h-10 rounded-full mt-1 flex-shrink-0" style={{ background: 'var(--copper)' }} />
      <div>
        <p className="label-sm mb-1" style={{ color: 'var(--accent)' }}>{eyebrow}</p>
        <h2 className="font-bold text-2xl" style={{ color: 'var(--text)', letterSpacing: '-0.02em' }}>{title}</h2>
        <p className="text-sm mt-1 max-w-2xl" style={{ color: 'var(--muted)' }}>{sub}</p>
      </div>
    </div>
  );
}

/* ── Coarse → fine illustration (deterministic, not real model output) ───── */
const LEVEL_COLORS = ['var(--raised-hover)', 'var(--heat-1)', 'var(--heat-2)', 'var(--heat-3)', 'var(--heat-4)'];
const LEVEL_LABELS = ['None', 'Light', 'Moderate', 'Heavy', 'Very heavy'];
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const toLevel = (v: number) => Math.min(4, Math.floor(clamp01(v) * 5));

const COARSE_N = 4;
const FINE_N = 20;
const coarseAt = (r: number, c: number) =>
  0.5 + 0.45 * Math.sin(r * 1.7 + c * 0.9 + 0.6) * Math.cos(c * 1.3 - r * 0.4);

function fineAt(i: number, j: number) {
  // bilinear sample of the coarse field, then add local detail the coarse grid cannot hold
  const y = ((i + 0.5) / FINE_N) * COARSE_N - 0.5;
  const x = ((j + 0.5) / FINE_N) * COARSE_N - 0.5;
  const y0 = Math.max(0, Math.min(COARSE_N - 1, Math.floor(y)));
  const x0 = Math.max(0, Math.min(COARSE_N - 1, Math.floor(x)));
  const y1 = Math.min(COARSE_N - 1, y0 + 1);
  const x1 = Math.min(COARSE_N - 1, x0 + 1);
  const ty = clamp01(y - y0);
  const tx = clamp01(x - x0);
  const base =
    coarseAt(y0, x0) * (1 - ty) * (1 - tx) + coarseAt(y0, x1) * (1 - ty) * tx +
    coarseAt(y1, x0) * ty * (1 - tx) + coarseAt(y1, x1) * ty * tx;
  return base + 0.16 * Math.sin(i * 0.9) * Math.cos(j * 1.1 + i * 0.3);
}

function GridIllustration() {
  const coarse = Array.from({ length: COARSE_N * COARSE_N }, (_, k) =>
    toLevel(coarseAt(Math.floor(k / COARSE_N), k % COARSE_N)));
  const fine = Array.from({ length: FINE_N * FINE_N }, (_, k) =>
    toLevel(fineAt(Math.floor(k / FINE_N), k % FINE_N)));

  return (
    <div className="rounded-2xl p-5" style={{ background: 'var(--panel)', border: '1px solid var(--hairline)' }}>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="label-sm mb-2">IMD · 0.25° · ~28 km</p>
          <div className="grid gap-px aspect-square" style={{ gridTemplateColumns: `repeat(${COARSE_N}, 1fr)`, background: 'var(--hairline)', border: '1px solid var(--hairline)' }}>
            {coarse.map((l, k) => <div key={k} style={{ background: LEVEL_COLORS[l] }} />)}
          </div>
        </div>
        <div>
          <p className="label-sm mb-2" style={{ color: 'var(--accent)' }}>Downscaled · 0.05° · ~5 km</p>
          <div className="grid aspect-square" style={{ gridTemplateColumns: `repeat(${FINE_N}, 1fr)`, border: '1px solid var(--hairline-strong)' }}>
            {fine.map((l, k) => <div key={k} style={{ background: LEVEL_COLORS[l] }} />)}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 mt-4 pt-4" style={{ borderTop: '1px solid var(--hairline)' }}>
        {LEVEL_LABELS.map((label, l) => (
          <span key={label} className="flex items-center gap-1.5 text-[0.55rem] font-mono uppercase tracking-wider" style={{ color: 'var(--muted)' }}>
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: LEVEL_COLORS[l], border: '1px solid var(--hairline-strong)' }} />
            {label}
          </span>
        ))}
      </div>
      <p className="text-[0.6rem] mt-3" style={{ color: 'var(--muted)' }}>
        Illustration only — a synthetic field showing the idea, not real model output.
      </p>
    </div>
  );
}

/* ── Main page ───────────────────────────────────────────────────────────── */
export function LandingPage() {
  return (
    <div className="navbar-offset min-h-screen" style={{ background: 'var(--canvas)' }}>
      <LandingNav />

      <main>

        {/* ═══ HERO ═══════════════════════════════════════════════════════ */}
        <section id="overview" className="max-w-6xl mx-auto px-6 pt-16 pb-16">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">

            <div>
              <div className="flex flex-wrap items-center gap-2 mb-5">
                <span className="stat-pill">
                  <Activity className="w-3 h-3" />
                  SIH 2026 · Layer 1
                </span>
                <span className="stat-pill">
                  <GitBranch className="w-3 h-3" />
                  Coarse-to-fine
                </span>
              </div>

              <h1
                className="font-bold leading-[1.08] mb-5"
                style={{ color: 'var(--text)', fontSize: 'clamp(2rem, 5vw, 3.2rem)', letterSpacing: '-0.03em' }}
              >
                Rainfall at <span style={{ color: 'var(--accent)' }}>5 km</span>,
                <br />
                for every <span style={{ color: 'var(--accent)' }}>Gram Panchayat</span>.
              </h1>

              <p className="text-sm leading-relaxed max-w-lg mb-8" style={{ color: 'var(--text-2)' }}>
                Obsidian learns the terrain- and atmosphere-driven correction that turns IMD&apos;s 0.25° (~28 km)
                daily rainfall into a 0.05° (~5 km) field, then aggregates it to 86,103 LGD Gram Panchayats
                across 15 Indian states. Every value comes with its inputs, mapping method and metrics, plus an
                AI-written explanation of what drove it.
              </p>

              <div className="flex flex-wrap items-center gap-3 mb-8">
                <Link href="/dashboard" className="landing-cta landing-cta-primary">
                  Open the dashboard <ArrowRight className="w-3.5 h-3.5" />
                </Link>
                <a href="#how-it-works" className="landing-cta landing-cta-ghost">
                  How it works
                </a>
                <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" className="landing-cta landing-cta-ghost">
                  View source ↗
                </a>
              </div>

              <div
                className="flex items-start gap-2.5 p-3.5 rounded-xl"
                style={{ background: 'rgba(176, 141, 87, 0.05)', border: '1px solid rgba(176, 141, 87, 0.18)' }}
              >
                <TriangleAlert className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" style={{ color: 'var(--accent)' }} />
                <p className="text-[0.7rem] leading-relaxed" style={{ color: 'var(--text-2)' }}>
                  Scored against CHIRPS v2.0 — a reference product, not absolute ground truth. No accuracy
                  claims are made against real-world rainfall.
                </p>
              </div>
            </div>

            <div className="flex flex-col gap-4">
              <GridIllustration />
              <div className="grid grid-cols-2 gap-3">
                {HERO_STATS.map((s) => (
                  <div key={s.label} className="landing-card p-4">
                    <div className="flex items-center gap-2 mb-2" style={{ color: 'var(--muted)' }}>
                      {s.icon}
                      <span className="text-[0.55rem] uppercase tracking-wider font-mono">{s.label}</span>
                    </div>
                    <p className="font-mono font-bold text-xl leading-none mb-1" style={{ color: 'var(--text)' }}>{s.value}</p>
                    <p className="text-[0.6rem] font-mono" style={{ color: 'var(--muted)' }}>{s.note}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* ═══ KEY NUMBERS ════════════════════════════════════════════════ */}
        <section style={{ background: 'var(--panel)', borderTop: '1px solid var(--hairline)', borderBottom: '1px solid var(--hairline)' }}>
          <div className="max-w-6xl mx-auto px-6 py-8 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-y-6">
            {KEY_NUMBERS.map((k, i) => (
              <div key={k.label} className="px-4" style={{ borderLeft: i === 0 ? 'none' : '1px solid var(--hairline)' }}>
                <p className="font-mono font-bold text-xl leading-none mb-1.5" style={{ color: 'var(--text)' }}>{k.value}</p>
                <p className="text-[0.6rem] uppercase tracking-wider font-mono" style={{ color: 'var(--accent)' }}>{k.label}</p>
                <p className="text-[0.6rem] font-mono mt-0.5" style={{ color: 'var(--muted)' }}>{k.note}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ═══ THE PROBLEM + LADDER ═══════════════════════════════════════ */}
        <section className="max-w-6xl mx-auto px-6 py-16">
          <SectionHeader
            eyebrow="The problem"
            title="28 km is too coarse to advise a farm"
            sub="Agricultural advisories need rainfall at roughly 5 km, but the operational gridded product is 0.25°."
          />

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-10">
            {PROBLEM.map((p) => (
              <div key={p.title} className="landing-card p-5">
                <p className="font-mono font-bold text-2xl mb-1" style={{ color: 'var(--accent)' }}>{p.big}</p>
                <p className="font-semibold text-sm mb-2" style={{ color: 'var(--text)' }}>{p.title}</p>
                <p className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>{p.body}</p>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {LADDER.map((step, i) => (
              <div key={step.tag} className="relative">
                <div className="landing-card-plain p-5 h-full">
                  <span
                    className="inline-block text-[0.6rem] font-mono font-bold uppercase tracking-[0.18em] px-2 py-0.5 rounded-md mb-3"
                    style={{
                      background: i === 0 ? 'rgba(255,255,255,0.05)' : 'rgba(176, 141, 87, 0.14)',
                      color: i === 0 ? 'var(--muted)' : 'var(--accent)',
                    }}
                  >
                    {step.tag}
                  </span>
                  <p className="font-mono font-semibold text-lg mb-1" style={{ color: 'var(--text)' }}>{step.title}</p>
                  <p className="font-mono text-sm mb-3" style={{ color: 'var(--accent)' }}>{step.value}</p>
                  <p className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>{step.note}</p>
                </div>
                {i < LADDER.length - 1 && (
                  <ArrowRight
                    className="hidden md:block w-4 h-4 absolute top-1/2 -right-[0.8rem] -translate-y-1/2 z-10"
                    style={{ color: 'var(--accent)', opacity: 0.6 }}
                  />
                )}
              </div>
            ))}
          </div>
        </section>

        {/* ═══ HOW IT WORKS ═══════════════════════════════════════════════ */}
        <section id="how-it-works" className="landing-section max-w-6xl mx-auto px-6 py-16">
          <SectionHeader
            eyebrow="Pipeline"
            title="How it works"
            sub="A transparent, reproducible pipeline from IMD coarse data to panchayat-level downscaled estimates."
          />

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {PIPELINE.map((step) => (
              <div key={step.n} className="landing-card-plain p-5">
                <div className="flex items-center gap-3 mb-4">
                  <div className="step-badge">{step.n}</div>
                  <div
                    className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
                    style={{ background: 'rgba(176, 141, 87, 0.1)', color: 'var(--accent)', border: '1px solid rgba(176, 141, 87, 0.2)' }}
                  >
                    {step.icon}
                  </div>
                </div>
                <h3 className="font-bold text-sm mb-1.5" style={{ color: 'var(--text)' }}>{step.title}</h3>
                <p className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>{step.desc}</p>
              </div>
            ))}
          </div>

          <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              { icon: <Database    className="w-3.5 h-3.5" />, label: 'Rainfall channel',    value: 'imd_rain' },
              { icon: <Mountain    className="w-3.5 h-3.5" />, label: 'Terrain channel',     value: 'dem (SRTM 30 m)' },
              { icon: <Thermometer className="w-3.5 h-3.5" />, label: 'Atmosphere channels', value: 'era5_t2m · t2m_max · dewp' },
            ].map((c) => (
              <div key={c.label} className="landing-card p-4">
                <div className="flex items-center gap-2 mb-2" style={{ color: 'var(--muted)' }}>
                  {c.icon}
                  <span className="text-[0.55rem] uppercase tracking-wider font-mono">{c.label}</span>
                </div>
                <p className="font-mono text-xs" style={{ color: 'var(--text)' }}>{c.value}</p>
              </div>
            ))}
          </div>

          <div className="mt-6 p-5 rounded-xl" style={{ background: 'var(--panel)', border: '1px solid var(--hairline)' }}>
            <div className="flex items-center gap-2 mb-4">
              <Layers className="w-3.5 h-3.5" style={{ color: 'var(--muted)' }} />
              <span className="label-sm">Inputs &amp; reference data</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {SOURCES.map((s) => <span key={s} className="landing-chip">{s}</span>)}
            </div>
          </div>

          {/* Layer 2 — panchayat mapping */}
          <div className="mt-12">
            <p className="label-sm mb-1" style={{ color: 'var(--accent)' }}>Layer 2</p>
            <h3 className="font-bold text-lg mb-1" style={{ color: 'var(--text)' }}>From grid cells to panchayats</h3>
            <p className="text-sm mb-5 max-w-2xl" style={{ color: 'var(--muted)' }}>
              Every output records how it was aggregated, so the value behind any panchayat can be audited.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              {MAPPING_METHODS.map((m) => (
                <div key={m.title} className="landing-card p-4">
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <p className="font-semibold text-sm" style={{ color: 'var(--text)' }}>{m.title}</p>
                    <span className="text-[0.5rem] font-mono uppercase tracking-wider" style={{ color: 'var(--muted)' }}>{m.tag}</span>
                  </div>
                  <p className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>{m.body}</p>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {COVERAGE.map((c) => (
                <div key={c.label} className="p-4 rounded-xl text-center" style={{ background: 'var(--raised)', border: '1px solid var(--hairline)' }}>
                  <p className="font-mono font-bold text-lg" style={{ color: 'var(--text)' }}>{c.value}</p>
                  <p className="text-[0.55rem] uppercase tracking-wider font-mono mt-1" style={{ color: 'var(--muted)' }}>{c.label}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ═══ EXPLAINABLE AI ═════════════════════════════════════════════ */}
        <section id="explainable-ai" className="landing-section max-w-6xl mx-auto px-6 py-16">
          <SectionHeader
            eyebrow="Explainable AI"
            title="Every number comes with a reason"
            sub="After you downscale a panchayat, the dashboard asks a language model to explain the estimate from the model's own inputs."
          />

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            <div className="lg:col-span-3 grid grid-cols-1 sm:grid-cols-2 gap-4">
              {XAI_PARTS.map((p, i) => (
                <div key={p.title} className="landing-card-plain p-5">
                  <span className="font-mono text-[0.6rem] font-bold" style={{ color: 'var(--accent)' }}>0{i + 1}</span>
                  <p className="font-semibold text-sm mt-1 mb-1.5" style={{ color: 'var(--text)' }}>{p.title}</p>
                  <p className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>{p.body}</p>
                </div>
              ))}
            </div>

            <div className="lg:col-span-2 p-5 rounded-xl flex flex-col gap-4" style={{ background: 'var(--panel)', border: '1px solid var(--hairline)' }}>
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4" style={{ color: 'var(--accent)' }} />
                <span className="label-sm">How it stays honest</span>
              </div>
              <ul className="space-y-3 text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>
                <li>The model is given only the numbers the pipeline produced and is told not to invent new ones.</li>
                <li>Factor bars are the language model&apos;s qualitative judgement, not SHAP values.</li>
                <li>The API key stays on the server; the browser never sees it.</li>
                <li>If the language model is unreachable, the panel says so and falls back to rule-based text.</li>
              </ul>
            </div>
          </div>
        </section>

        {/* ═══ IN THE DASHBOARD ═══════════════════════════════════════════ */}
        <section className="landing-section max-w-6xl mx-auto px-6 py-16">
          <SectionHeader
            eyebrow="Dashboard"
            title="What you can do with it"
            sub="From a panchayat name to a rainfall estimate, advisory and explanation in four steps."
          />
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {DASH_FEATURES.map((f, i) => (
              <div key={f.title} className="landing-card-plain p-5">
                <div className="flex items-center justify-between mb-4">
                  <div
                    className="w-9 h-9 rounded-xl flex items-center justify-center"
                    style={{ background: 'rgba(176, 141, 87, 0.1)', color: 'var(--accent)', border: '1px solid rgba(176, 141, 87, 0.2)' }}
                  >
                    {f.icon}
                  </div>
                  <span className="font-mono text-[0.6rem]" style={{ color: 'var(--muted)' }}>0{i + 1}</span>
                </div>
                <p className="font-semibold text-sm mb-1.5" style={{ color: 'var(--text)' }}>{f.title}</p>
                <p className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>{f.body}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ═══ RESULTS ════════════════════════════════════════════════════ */}
        <section id="results" className="landing-section max-w-6xl mx-auto px-6 py-16">
          <SectionHeader
            eyebrow="Results"
            title="Ablation across the evaluation splits"
            sub="Reference: CHIRPS 0.05°. Lower MAE is better."
          />

          <div className="flex items-start gap-2.5 p-3.5 rounded-xl mb-4" style={{ background: 'rgba(176, 141, 87, 0.05)', border: '1px solid rgba(176, 141, 87, 0.18)' }}>
            <Info className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" style={{ color: 'var(--accent)' }} />
            <p className="text-[0.7rem] leading-relaxed" style={{ color: 'var(--text-2)' }}>
              These are pilot numbers (Western Ghats, 13–17° N, 2019–2022). Training on the full Deccan dataset is still pending.
            </p>
          </div>

          <div className="overflow-x-auto rounded-xl" style={{ border: '1px solid var(--hairline)' }}>
            <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--hairline)', background: 'var(--panel)' }}>
                  <th className="text-left py-3 px-4 label-sm">Model</th>
                  <th className="text-right py-3 px-4 label-sm whitespace-nowrap">Val MAE</th>
                  <th className="text-right py-3 px-4 label-sm whitespace-nowrap">Val corr</th>
                  <th className="text-right py-3 px-4 label-sm whitespace-nowrap">Test MAE</th>
                  <th className="text-right py-3 px-4 label-sm whitespace-nowrap">Test corr</th>
                  <th className="text-right py-3 px-4 label-sm whitespace-nowrap">F1 ≥25mm</th>
                </tr>
              </thead>
              <tbody>
                {RESULTS.map((row) => (
                  <tr
                    key={row.model}
                    style={{ borderBottom: '1px solid var(--hairline)', background: row.selected ? 'rgba(176, 141, 87, 0.07)' : undefined }}
                  >
                    <td className="py-2.5 px-4" style={{ color: row.selected ? 'var(--text)' : 'var(--text-2)' }}>
                      <span className="flex items-center gap-2">
                        <span>{row.model}</span>
                        {row.selected && (
                          <span className="text-[0.5rem] font-mono font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-md" style={{ background: 'rgba(176, 141, 87, 0.15)', color: 'var(--accent)' }}>
                            Selected
                          </span>
                        )}
                        {row.flagged && (
                          <span className="text-[0.5rem] font-mono uppercase tracking-wider" style={{ color: 'var(--muted)' }}>neg. result</span>
                        )}
                      </span>
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono" style={{ color: 'var(--text-2)' }}>{row.valMae}</td>
                    <td className="py-2.5 px-4 text-right font-mono" style={{ color: 'var(--text-2)' }}>{row.valCorr}</td>
                    <td className="py-2.5 px-4 text-right font-mono" style={{ color: 'var(--text-2)' }}>{row.testMae}</td>
                    <td className="py-2.5 px-4 text-right font-mono" style={{ color: 'var(--text-2)' }}>{row.testCorr}</td>
                    <td className="py-2.5 px-4 text-right font-mono" style={{ color: 'var(--text-2)' }}>{row.f1}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              { label: 'MAE vs baseline', value: '−15.9%',    note: 'selected model D on unseen 2022 data' },
              { label: 'Correlation',     value: 'B → C → D', note: 'rose monotonically across ablations' },
              { label: 'Weighted loss',   value: '−0.7 mm',   note: 'MAE cost for far better heavy-rain detection' },
            ].map((s) => (
              <div key={s.label} className="landing-card p-4">
                <p className="text-[0.55rem] uppercase tracking-wider mb-2 font-mono" style={{ color: 'var(--muted)' }}>{s.label}</p>
                <p className="font-mono font-bold text-base mb-1" style={{ color: 'var(--accent)' }}>{s.value}</p>
                <p className="text-[0.6rem] leading-relaxed" style={{ color: 'var(--muted)' }}>{s.note}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ═══ LIMITATIONS ════════════════════════════════════════════════ */}
        <section className="landing-section max-w-6xl mx-auto px-6 py-16">
          <SectionHeader
            eyebrow="Limitations"
            title="What it does not claim"
            sub="Stated plainly, so nobody has to discover them later."
          />
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {LIMITATIONS.map((l) => (
              <div key={l.title} className="flex items-start gap-3 p-4 rounded-xl" style={{ background: 'var(--raised)', border: '1px solid var(--hairline)' }}>
                <Info className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: 'var(--muted)' }} />
                <div>
                  <p className="font-semibold text-sm mb-1" style={{ color: 'var(--text)' }}>{l.title}</p>
                  <p className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>{l.body}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ═══ FAQ ════════════════════════════════════════════════════════ */}
        <section id="faq" className="landing-section max-w-3xl mx-auto px-6 py-16">
          <SectionHeader
            eyebrow="FAQ"
            title="Frequently asked questions"
            sub="The honest answers — what this system is, what it is not, and what it can and cannot claim."
          />

          <FaqSection />

          <div
            className="mt-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-5 rounded-xl"
            style={{ background: 'var(--raised)', border: '1px solid var(--hairline-strong)' }}
          >
            <div>
              <p className="font-bold text-sm mb-1" style={{ color: 'var(--text)' }}>Search a panchayat</p>
              <p className="text-xs" style={{ color: 'var(--muted)' }}>
                Explore downscaled rainfall, model metrics and the mapping behind each value.
              </p>
            </div>
            <Link href="/dashboard" className="landing-cta landing-cta-primary flex-shrink-0">
              Open the dashboard <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </section>
      </main>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer style={{ borderTop: '1px solid var(--hairline)', background: 'var(--panel)' }}>
        <div className="max-w-6xl mx-auto px-6 py-10 grid grid-cols-1 sm:grid-cols-3 gap-8">
          <div>
            <p className="font-bold text-sm mb-1" style={{ color: 'var(--text)' }}>Obsidian</p>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--muted)' }}>
              Coarse-to-fine rainfall downscaling for Gram Panchayats. Built for Smart India Hackathon 2026.
            </p>
          </div>
          <div>
            <p className="label-sm mb-2">Explore</p>
            <ul className="space-y-1.5 text-xs" style={{ color: 'var(--text-2)' }}>
              <li><Link href="/dashboard" className="no-underline" style={{ color: 'inherit' }}>Dashboard</Link></li>
              <li><a href="#how-it-works" className="no-underline" style={{ color: 'inherit' }}>How it works</a></li>
              <li><a href="#results" className="no-underline" style={{ color: 'inherit' }}>Results</a></li>
              <li><a href="#faq" className="no-underline" style={{ color: 'inherit' }}>FAQ</a></li>
            </ul>
          </div>
          <div>
            <p className="label-sm mb-2">Data &amp; credits</p>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--muted)' }}>
              Reference product: CHIRPS v2.0 · Boundaries: LGD (CC0) · Terrain: SRTM · Weather context: ERA5-Land ·
              No accuracy claims made against real-world rainfall.
            </p>
          </div>
        </div>
        <div className="py-4 text-center" style={{ borderTop: '1px solid var(--hairline)' }}>
          <p className="text-[0.6rem] font-mono uppercase tracking-widest" style={{ color: 'var(--muted)', opacity: 0.6 }}>
            Obsidian · 15 states · 86,103 panchayats · 2022 monsoon dataset
          </p>
        </div>
      </footer>
    </div>
  );
}
