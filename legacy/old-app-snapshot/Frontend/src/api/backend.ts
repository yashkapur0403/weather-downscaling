/**
 * api/backend.ts
 * ──────────────
 * Single place for all backend API calls.
 * Base URL from env; falls back to localhost:8000.
 */

import type { QueryRequest, QueryResponse, Panchayat, Weather, ModelMetrics, AdvisoryResponse, CropType, CropStage } from '../types';

const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} → ${res.status}: ${text}`);
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

// ── Weather (temperature, humidity, elevation) ───────────────────────────────
export async function fetchWeather(
  lat: number,
  lon: number,
  date: string
): Promise<Weather> {
  const params = new URLSearchParams({ lat: String(lat), lon: String(lon), date });
  return apiFetch<Weather>(`/api/weather?${params}`);
}

// ── Model metrics ─────────────────────────────────────────────────────────────
export async function fetchMetrics(): Promise<ModelMetrics> {
  return apiFetch<ModelMetrics>('/api/metrics');
}

// ── Crop advisory (stub — waiting for backend /api/advisory endpoint) ─────────
export async function fetchAdvisory(
  panchayatId: number,
  crop: CropType,
  stage: CropStage,
  irrigationAvailable?: boolean,
): Promise<AdvisoryResponse> {
  const params = new URLSearchParams({
    panchayat_id: String(panchayatId),
    crop,
    stage,
  });
  if (irrigationAvailable !== undefined) {
    params.set('irrigation_available', String(irrigationAvailable));
  }
  return apiFetch<AdvisoryResponse>(`/api/advisory?${params}`);
}
