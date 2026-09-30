// ── Backend request/response models ─────────────────────────────────────────

export interface QueryRequest {
  state?: string | null;
  district?: string | null;
  block_name?: string | null;
  panchayat_name?: string | null;
  lat?: number | null;
  lon?: number | null;
  date?: string | null;
  requested_metrics?: string[];
  raw_location_text?: string | null;
}

export interface QueryResponse {
  params: QueryRequest;
  prediction: {
    rainfall_mm?: number | null;
    temperature_c?: number | null;
    humidity_pct?: number | null;
    elevation_m?: number | null;
  };
  answer: string;
  model_status: 'loading' | 'ready' | 'unavailable';
}

// ── Panchayat search result (legacy - for backward compatibility) ────────────
export interface Panchayat {
  panchayat_id: number;
  panchayat_name: string;
  block_name: string;
  block_id: number;
  district: string;
  state: string;
  date: string;           // "2022-07-10"
  rainfall_mm: number;
  temperature_c: number | null;
  humidity_pct: number | null;
  elevation_m: number | null;
  n_cells: number;
  mapping_method: 'direct_grid' | 'area_weighted' | 'nearest_fallback';
  fallback_distance_m: number | null;
  lat: number | null;
  lon: number | null;
}

// ── Weather (from GET /api/weather) ─────────────────────────────────────────
export interface Weather {
  temperature_c: number | null;
  humidity_pct: number | null;
  elevation_m: number | null;
  temperature_source: string | null;
  humidity_source: string | null;
  elevation_source: string | null;
  error: string | null;
}

// ── Model metrics (from GET /api/metrics) ───────────────────────────────────
export interface ModelVariantRow {
  split: string;
  model: string;
  MAE_mm: number;
  RMSE_mm: number;
  corr: number;
}

export interface ModelMetrics {
  all_variants: ModelVariantRow[];
  test_variants: ModelVariantRow[];
  selected_model: {
    name: string;
    variant_key: string;
    test_mae_mm: number | null;
    test_rmse_mm: number | null;
    test_correlation: number | null;
  };
  baseline: {
    name: string;
    test_mae_mm: number | null;
  };
  mae_improvement_pct: number | null;
  evaluation_period: string;
  reference_product: string;
  reference_note: string;
  channels: string[];
  n_parameters: string;
}

// ── XAI breakdown assembled on the frontend ─────────────────────────────────
export interface XAIBreakdown {
  panchayat: Panchayat;
  weather: Weather | null;
  metrics: ModelMetrics | null;
}

// ── Rainfall risk classification (IMD thresholds) ───────────────────────────
export type RiskLevel = 'no_rain' | 'light' | 'moderate' | 'heavy' | 'very_heavy';

export function classifyRisk(mm: number): RiskLevel {
  if (mm < 2.5) return 'no_rain';
  if (mm < 10) return 'light';
  if (mm < 25) return 'moderate';
  if (mm < 50) return 'heavy';
  return 'very_heavy';
}

export const RISK_LABEL: Record<RiskLevel, string> = {
  no_rain: 'No Rain',
  light: 'Light',
  moderate: 'Moderate',
  heavy: 'Heavy',
  very_heavy: 'Very Heavy',
};

export const RISK_COLOR: Record<RiskLevel, string> = {
  no_rain: 'text-slate-500',
  light: 'text-blue-600',
  moderate: 'text-amber-600',
  heavy: 'text-orange-600',
  very_heavy: 'text-red-600',
};

export const RISK_BG: Record<RiskLevel, string> = {
  no_rain: 'bg-slate-100 text-slate-600',
  light: 'bg-blue-50 text-blue-700',
  moderate: 'bg-amber-50 text-amber-700',
  heavy: 'bg-orange-50 text-orange-700',
  very_heavy: 'bg-red-50 text-red-700',
};

// ── Crop advisory (from GET /api/advisory) ──────────────────────────────────
export type AdvisorySeverity = 'info' | 'watch' | 'warning' | 'alert';

export interface AdvisoryResponse {
  advisory_text: string;
  severity: AdvisorySeverity;
  actions: string[];
  evidence: {
    rainfall_mm: number;
    risk_level: RiskLevel;
    temperature_c: number | null;
    humidity_pct: number | null;
  };
  data_date: string;
  disclaimer: string;
  crop: string;
  stage: string;
  /** What the backend CHECKED before running the rules. `rainfall` is
   *  'verified_against_layer1' when the value we sent matched the stored
   *  Layer-1 field for this Panchayat and date, 'resolved_from_layer1' when the
   *  server looked it up itself, 'unverified_no_data_source' when no data layer
   *  was mounted. Temperature/humidity are always caller-supplied. */
  verification?: {
    panchayat_id: number;
    date: string;
    rainfall: 'verified_against_layer1' | 'resolved_from_layer1' | 'unverified_no_data_source';
    supplied_mm: number | null;
    expected_mm: number | null;
    source: string | null;
    cell: [number, number] | null;
    location_precision: string | null;
    reason?: string | null;
    temperature_c: 'caller_supplied' | 'absent';
    humidity_pct: 'caller_supplied' | 'absent';
    aux: string;
  };
}

export const CROP_OPTIONS = [
  { value: 'general', label: 'General Field Advisory' },
  { value: 'rice', label: 'Rice (Paddy)' },
  { value: 'wheat', label: 'Wheat' },
  { value: 'cotton', label: 'Cotton' },
  { value: 'maize', label: 'Maize (Corn)' },
  { value: 'pulses', label: 'Pulses' },
] as const;

export const STAGE_OPTIONS = [
  { value: 'general', label: 'General' },
  { value: 'sowing', label: 'Sowing / Planting' },
  { value: 'vegetative', label: 'Vegetative Growth' },
  { value: 'flowering', label: 'Flowering / Anthesis' },
  { value: 'ripening', label: 'Ripening / Maturity' },
  { value: 'harvest', label: 'Harvest' },
] as const;

export type CropType = (typeof CROP_OPTIONS)[number]['value'];
export type CropStage = (typeof STAGE_OPTIONS)[number]['value'];

// ── Regional language (Sarvam AI translation, applied server-side) ─────────
// Mirrors the `LangCode` literal in backend/main.py exactly — these are the
// only codes the backend will translate the advisory into.
export const LANGUAGE_OPTIONS = [
  { value: 'en-IN', label: 'English' },
  { value: 'hi-IN', label: 'हिंदी (Hindi)' },
  { value: 'bn-IN', label: 'বাংলা (Bengali)' },
  { value: 'ta-IN', label: 'தமிழ் (Tamil)' },
  { value: 'te-IN', label: 'తెలుగు (Telugu)' },
  { value: 'kn-IN', label: 'ಕನ್ನಡ (Kannada)' },
  { value: 'ml-IN', label: 'മലയാളം (Malayalam)' },
  { value: 'mr-IN', label: 'मराठी (Marathi)' },
  { value: 'gu-IN', label: 'ગુજરાતી (Gujarati)' },
  { value: 'od-IN', label: 'ଓଡ଼ିଆ (Odia)' },
  { value: 'pa-IN', label: 'ਪੰਜਾਬੀ (Punjabi)' },
] as const;

export type LangCode = (typeof LANGUAGE_OPTIONS)[number]['value'];

export const SEVERITY_STYLE: Record<AdvisorySeverity, { bg: string; color: string; border: string }> = {
  info:    { bg: 'rgba(47,111,143,0.08)',  color: 'var(--heat-1)', border: 'rgba(47,111,143,0.2)' },
  watch:   { bg: 'rgba(183,146,55,0.08)',  color: 'var(--heat-2)', border: 'rgba(183,146,55,0.2)' },
  warning: { bg: 'rgba(190,106,46,0.08)', color: 'var(--heat-3)', border: 'rgba(190,106,46,0.2)' },
  alert:   { bg: 'rgba(162,58,48,0.08)',  color: 'var(--heat-4)', border: 'rgba(162,58,48,0.2)' },
};

// ── Explainable AI (POST /api/explain — Groq LLM on the backend) ────────────
export interface ExplainRequest {
  panchayat_name: string;
  block_name: string;
  district: string;
  state: string;
  date: string;
  lat: number | null;
  lon: number | null;
  prediction: {
    rainfall_mm: number;
    risk_level: RiskLevel;
    temperature_c: number | null;
    humidity_pct: number | null;
    elevation_m: number | null;
  };
  mapping: {
    method: Panchayat['mapping_method'];
    n_cells: number;
    fallback_distance_m: number | null;
  };
  model: {
    name: string;
    channels: string[];
    test_mae_mm: number | null;
    baseline_mae_mm: number | null;
    mae_improvement_pct: number | null;
    reference_product: string;
  };
  question?: string | null;   // optional follow-up from the user
  language?: 'en' | 'hi';
}

export type FactorEffect = 'increases' | 'decreases' | 'neutral';

export interface ExplainFactor {
  factor: string;             // e.g. "Elevation (DEM)"
  effect: FactorEffect;       // direction of influence on the estimate
  weight: number;             // 0–1 relative importance, as judged by the LLM
  detail: string;
}

export interface ExplainResponse {
  summary: string;            // one-line takeaway
  explanation: string;        // plain-language paragraph(s)
  factors: ExplainFactor[];
  confidence: 'low' | 'medium' | 'high';
  confidence_note: string;
  answer?: string | null;     // answer to `question`, when one was asked
  provider: string;           // "groq"
  model: string;              // e.g. "llama-3.3-70b-versatile"
  generated_at: string;       // ISO timestamp
  fallback_reason?: string | null; // why rules were used instead of the LLM
}