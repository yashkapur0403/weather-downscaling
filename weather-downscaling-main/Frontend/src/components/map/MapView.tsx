'use client';

import { useEffect, useRef } from 'react';
import type { Panchayat } from '../../types';

interface MapViewProps {
  selected: Panchayat | null;
  onSelectLocation?: (lat: number, lon: number) => void;
}

export function MapView({ selected, onSelectLocation }: MapViewProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const markerRef = useRef<any>(null);

  // ── Initialize map (once) ──────────────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    let cancelled = false;

    import('leaflet').then((L) => {
      if (cancelled || !mapRef.current || mapInstanceRef.current) return;

      // Guard: if the container already has a Leaflet instance, remove it first
      if ((mapRef.current as any)._leaflet_id) {
        mapRef.current.innerHTML = '';
      }

      const map = L.map(mapRef.current!, {
        center: [15.5, 75.5],
        zoom: 7,
        zoomControl: false,
        attributionControl: false,
      });

      // OpenStreetMap tiles (free, no API key required)
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
      }).addTo(map);

      // Zoom control on top-right
      L.control.zoom({ position: 'topright' }).addTo(map);

      // Attribution at bottom
      L.control.attribution({ position: 'bottomleft', prefix: false })
        .addAttribution('© <a href="https://osm.org/copyright">OpenStreetMap</a> contributors')
        .addTo(map);

      mapInstanceRef.current = map;

      // Click handler to select location
      map.on('click', (e: any) => {
        if (onSelectLocation) {
          onSelectLocation(e.latlng.lat, e.latlng.lng);
        }
      });
    });

    return () => {
      cancelled = true;
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Update marker when selected panchayat changes ──────────────────────────
  useEffect(() => {
    if (!mapInstanceRef.current || !selected?.lat || !selected?.lon) return;

    import('leaflet').then((L) => {
      const map = mapInstanceRef.current;
      if (!map) return;

      // Remove old marker
      if (markerRef.current) {
        map.removeLayer(markerRef.current);
      }

      // Custom marker
      const icon = L.divIcon({
        className: '',
        html: `<div style="width:14px;height:14px;background:#DC2626;border:2px solid white;border-radius:50%;box-shadow:0 1px 4px rgba(0,0,0,0.3)"></div>`,
        iconSize: [16, 16],
        iconAnchor: [8, 8],
      });

      const marker = L.marker([selected.lat!, selected.lon!], { icon })
        .addTo(map)
        .bindPopup(
          `<div style="font-family:Inter,sans-serif;padding:4px;min-width:150px">
            <strong style="color:#0F172A">${selected.panchayat_name}</strong><br/>
            <span style="color:#94A3B8;font-size:0.7rem">${selected.block_name} · ${selected.district}</span><br/>
            <span style="color:#DC2626;font-weight:600;font-size:0.8rem">${selected.rainfall_mm.toFixed(2)} mm/day</span>
          </div>`,
          { className: 'light-popup' }
        );

      markerRef.current = marker;

      // Smooth pan to location
      map.flyTo([selected.lat!, selected.lon!], 11, { duration: 1.2 });
    });
  }, [selected]);

  return (
    <div className="relative w-full h-full gpu-layer">
      <div ref={mapRef} className="w-full h-full" style={{ background: '#050608' }} />
      {/* Map overlay text when nothing selected */}
      {!selected && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <p className="text-[0.6rem] font-mono uppercase tracking-[0.25em] mb-2" style={{ color: 'var(--muted)' }}>
              Search a location to begin
            </p>
            <p className="text-[0.6rem]" style={{ color: 'rgba(140,140,150,0.4)' }}>
              Panchayat rainfall · OpenStreetMap · ERA5 · SRTM
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
