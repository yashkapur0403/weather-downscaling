import type { Metadata, Viewport } from 'next';
import '../index.css';

export const metadata: Metadata = {
  title: 'Obsidian — Panchayat-level Crop Intelligence',
  description:
    'IMD rainfall downscaled from 28 km to 5 km using terrain and atmospheric context. Search 86,103 Gram Panchayats across 15 Indian states.',
  openGraph: {
    title: 'Obsidian — Panchayat-level Crop Intelligence',
    description: 'Downscaled rainfall for every Gram Panchayat.',
  },
};

export const viewport: Viewport = {
  themeColor: '#050608',
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
