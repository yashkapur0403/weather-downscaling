import path from 'node:path';
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // The user's home directory contains an unrelated package-lock.json that
  // confuses Next's workspace-root detection.
  outputFileTracingRoot: path.join(process.cwd()),
  
  // Enable standalone output for Docker deployment
  output: 'standalone',
};

export default nextConfig;