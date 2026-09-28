import type { Metadata, Viewport } from 'next';
import '../index.css';

export const metadata: Metadata = {
  title: 'Obsidian — Panchayat-level Weather Intelligence',
  description:
    'IMD rainfall downscaled from 28 km to 5 km using terrain and atmospheric context. Search 8,236 Gram Panchayats across 6 Indian states.',
  openGraph: {
    title: 'Obsidian — Panchayat-level Weather Intelligence',
    description: 'Downscaled rainfall for every Gram Panchayat.',
  },
};

export const viewport: Viewport = {
  themeColor: '#EDF2F7',
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
