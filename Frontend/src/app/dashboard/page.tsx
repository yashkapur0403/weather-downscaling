import type { Metadata } from 'next';
import App from '@/src/App';

export const metadata: Metadata = {
  title: 'Dashboard — Obsidian Crop Intelligence',
  description:
    'Search 87,735 Gram Panchayats, run the downscaling projection and inspect the model metrics behind every value.',
};

export default function DashboardPage() {
  return <App />;
}
