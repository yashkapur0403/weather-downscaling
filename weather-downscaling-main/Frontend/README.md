# Obsidian — Weather Intelligence Frontend

Next.js dashboard for panchayat-level rainfall downscaling and crop advisory.

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Create your local env file (copy the example)
cp .env.example .env.local

# 3. Start the dev server
npm run dev
```

The frontend runs at **http://localhost:3000**.

## Backend Connection

The frontend expects a FastAPI backend at `http://localhost:8000`.

```bash
# From the repo root
cd weather-downscaling-main
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
```

To point the frontend at a different backend URL, edit `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

> `.env.local` is gitignored — each developer creates their own.
> See `.env.example` for the required variables.

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start dev server (port 3000) |
| `npm run build` | Production build |
| `npm run start` | Serve production build |
| `npm run typecheck` | Run TypeScript checks |
| `npm run lint` | Run Oxlint |

## Tech Stack

- **Next.js 15** (App Router)
- **React 19** + TypeScript
- **Tailwind CSS 3**
- **Leaflet** (map)
- **Recharts** (charts)
- **Lucide React** (icons)
