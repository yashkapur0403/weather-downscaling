/**
 * api/backend.ts
 * ──────────────
 * Single place for all backend API calls.
 * Base URL from env; falls back to localhost:8000.
 */

import type { QueryRequest, QueryResponse, Panchayat, ModelMetrics, AdvisoryResponse, CropType, CropStage, LangCode, ExplainRequest, ExplainResponse } from '../types';

const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

/**
 * An API failure that keeps the HTTP status and parsed body, so callers can
 * tell a "you sent me the wrong value" (409) apart from "the service is down".
 * The advisory panel relies on this: a rainfall mismatch must be surfaced, and
 * a refused request must not be replaced by advice the panel made up itself.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(path: string, status: number, body: unknown, raw: string) {
    super(`API ${path} → ${status}: ${raw}`);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    let body: unknown = null;
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
    throw new ApiError(path, res.status, body, text);
  }
  return res.json() as Promise<T>;
}

// ── Main query endpoint (POST /auth/) ─────────────────────────────────────────
export async function queryWeather(request: QueryRequest): Promise<QueryResponse> {
  return apiFetch<QueryResponse>('/auth/', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

// ── Health check ──────────────────────────────────────────────────────────────
export async function checkHealth(): Promise<{ message: string }> {
  return apiFetch<{ message: string }>('/');
}

// ── Legacy endpoints (kept for backward compatibility) ───────────────────────

// ── Panchayat search ─────────────────────────────────────────────────────────
export async function searchPanchayats(q: string, limit = 20): Promise<Panchayat[]> {
  const params = new URLSearchParams({ q, limit: String(limit) });
  return apiFetch<Panchayat[]>(`/api/panchayats?${params}`);
}

export async function geocodePanchayat(panchayat: Panchayat): Promise<{ lat: number; lon: number }> {
  const params = new URLSearchParams({
    panchayat_name: panchayat.panchayat_name,
    block_name: panchayat.block_name,
    district: panchayat.district,
    state: panchayat.state,
  });
  return apiFetch<{ lat: number; lon: number }>(`/api/geocode?${params}`);
}

// ── Weather ───────────────────────────────────────────────────────────────────
// The backend now returns temperature/humidity/elevation as part of POST /auth/,
// so the separate GET /api/weather helper (and the duplicate Next.js route that
// mirrored it) was removed to avoid two sources of truth for the same quantity.

// ── Model metrics ─────────────────────────────────────────────────────────────
export async function fetchMetrics(): Promise<ModelMetrics> {
  return apiFetch<ModelMetrics>('/api/metrics');
}

// ── Crop advisory (GET /api/advisory — requires all sensor params) ────────────
// `rainfallMm` is sent so the server can CHECK it: the route resolves the
// Layer-1 value for this Panchayat + date itself and answers 409 if the two
// disagree, so a stale or wrong client value can no longer produce an advisory
// about a different place or day.
export async function fetchAdvisory(
  panchayatId: number,
  crop: CropType,
  stage: CropStage,
  rainfallMm: number,
  temperatureC: number | null,
  date: string,
  panchayatName: string,
  irrigationAvailable?: boolean,
  lang: LangCode = 'en-IN',
): Promise<AdvisoryResponse> {
  const params = new URLSearchParams({
    panchayat_id: String(panchayatId),
    crop,
    stage,
    rainfall_mm: String(rainfallMm),
    date,
    panchayat_name: panchayatName,
    lang,
  });
  if (temperatureC !== null && temperatureC !== undefined) {
    params.set('temperature_c', String(temperatureC));
  }
  if (irrigationAvailable !== undefined) {
    params.set('irrigation_available', String(irrigationAvailable));
  }
  return apiFetch<AdvisoryResponse>(`/api/advisory?${params}`);
}

// ── Explainable AI (POST /api/explain — backend calls Groq LLM) ──────────────
// The Groq API key lives only on the backend; the frontend never sees it.
export async function fetchExplanation(
  request: ExplainRequest,
  signal?: AbortSignal,
): Promise<ExplainResponse> {
  return apiFetch<ExplainResponse>('/api/explain', {
    method: 'POST',
    body: JSON.stringify(request),
    signal,
  });
}

// ── Dashboard chatbot (uses POST /api/explain/generic) ───────────────────────────────────────
export interface ChatTurn { role: 'user' | 'assistant'; content: string }

export async function sendChat(
  message: string,
  context: string,
  output_language: LangCode,
): Promise<{ answer: string }> {
  const r = await apiFetch<{ explanation: { answer: string } }>('/api/explain/generic', {
    method: 'POST',
    body: JSON.stringify({ text: message, context, input_language: 'auto', output_language }),
  });
  return { answer: r.explanation.answer };
}

// ── XAI status (GET /api/explain/status) — is the LLM configured? ────────────
export async function fetchExplainStatus(): Promise<{ enabled: boolean; provider: string; model: string }> {
  return apiFetch<{ enabled: boolean; provider: string; model: string }>('/api/explain/status');
}
