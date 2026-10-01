import type { Metadata } from 'next';
import { LandingPage } from '@/src/views/LandingPage';

export const metadata: Metadata = {
  title: 'Obsidian — Rainfall downscaling from 28 km to 5 km',
  description:
    'IMD 0.25° rainfall refined to a 0.05° field with a residual U-Net over SRTM terrain and ERA5-Land context, aggregated to 87,735 LGD Gram Panchayats across 15 Indian states.',
};

export default function Page() {
  return <LandingPage />;
}
