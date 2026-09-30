import { NextRequest, NextResponse } from 'next/server';
import https from 'node:https';

/**
 * Simple HTTPS GET using Node's built-in https module (avoids Next.js fetch quirks).
 */
function httpsGet(url: string): Promise<string> {
  return new Promise((resolve, reject) => {
    https.get(url, { timeout: 15000 }, (res) => {
      let body = '';
      res.on('data', (chunk) => (body += chunk));
      res.on('end', () => resolve(body));
    }).on('error', reject);
  });
}

/**
 * GET /api/weather?lat=...&lon=...&date=...
 *
 * Server-side proxy for weather data. Calls Open-Meteo (temperature, humidity)
 * and Open-Topo-Data (elevation) directly — avoids CORS and backend socket issues.
 */
export async function GET(req: NextRequest) {
  const { searchParams } = req.nextUrl;
  const lat = searchParams.get('lat');
  const lon = searchParams.get('lon');
  const date = searchParams.get('date') ?? '2022-07-10';

  if (!lat || !lon) {
    return NextResponse.json({ error: 'lat and lon are required' }, { status: 400 });
  }

  const result: Record<string, unknown> = {
    temperature_c: null,
    humidity_pct: null,
    elevation_m: null,
    temperature_source: null,
    humidity_source: null,
    elevation_source: null,
    error: null,
  };

  // Temperature & Humidity (Open-Meteo Archive)
  try {
    const omUrl =
      `https://archive-api.open-meteo.com/v1/archive` +
      `?latitude=${lat}&longitude=${lon}` +
      `&start_date=${date}&end_date=${date}` +
      `&daily=temperature_2m_mean,relative_humidity_2m_mean` +
      `&timezone=Asia%2FKolkata`;
    const raw = await httpsGet(omUrl);
    const data = JSON.parse(raw);
    const daily = data.daily ?? {};
    const temps: (number | null)[] = daily.temperature_2m_mean ?? [];
    const hum: (number | null)[] = daily.relative_humidity_2m_mean ?? [];
    if (temps[0] != null) {
      result.temperature_c = Math.round(temps[0] * 10) / 10;
      result.temperature_source = 'Open-Meteo Archive (ERA5 reanalysis)';
    }
    if (hum[0] != null) {
      result.humidity_pct = Math.round(hum[0] * 10) / 10;
      result.humidity_source = 'Open-Meteo Archive (ERA5 reanalysis)';
    }
  } catch (err) {
    result.error = String(err);
  }

  // Elevation (Open-Topo-Data SRTM 30m)
  try {
    const topoUrl = `https://api.opentopodata.org/v1/srtm30m?locations=${lat},${lon}`;
    const raw = await httpsGet(topoUrl);
    const data = JSON.parse(raw);
    const results = data.results ?? [];
    if (results[0]?.elevation != null) {
      result.elevation_m = Math.round(results[0].elevation * 10) / 10;
      result.elevation_source = 'Open-Topo-Data (SRTM 30m)';
    }
  } catch (err) {
    if (!result.error) result.error = String(err);
  }

  return NextResponse.json(result);
}
