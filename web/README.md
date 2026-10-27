# FAST-TRACK AI: Next-Generation Command Center

A modern, high-performance security operations command center rebuilt with **Next.js 14 (App Router)**, **TypeScript**, **Tailwind CSS**, and **Framer Motion**, featuring a deep navy glassmorphism aesthetic with soft neon indicators, smooth scrolling, and an isolated API layer.

---

## 1. Tech Stack & Architecture

- **Framework:** Next.js 14 (App Router) + React 18 + TypeScript
- **Styling:** Tailwind CSS + Glassmorphism (`backdrop-blur-xl`, `border-white/10`)
- **Animations:** Framer Motion (page transitions, count-up numbers, smooth slide-overs)
- **Smooth Scroll:** Lenis (`@studio-freight/lenis` / `lenis`)
- **Telemetry & Charts:** Recharts (severity donut chart, 24h area threat timeline)
- **Icons:** `lucide-react` (strictly no emojis)
- **Global State:** Zustand (`useAppStore` for incident triage, live camera feeds, tuning popovers)
- **Data & Mock Layer:** Self-contained high-fidelity mock data in `src/lib/mock-data.ts`
- **Isolated API Client:** `src/lib/api.ts` for clean REST/WebSocket connection to the Python backend

---

## 2. Quick Start

### Prerequisites
- Node.js 18+ (tested on Node v22)
- npm 9+

### Run Development Server
```bash
# Navigate to the web folder
cd web

# Install dependencies (if not already done)
npm install

# Start development server on port 3000
npm run dev
```

Visit **[http://localhost:3000](http://localhost:3000)** in your browser.

### Build Production Bundle
```bash
npm run build
npm run start
```

---

## 3. Design System Specs

| Token | Hex / Class | Purpose |
| :--- | :--- | :--- |
| **Background** | `#070B14` (`bg-navy-900`) | Deep navy/near-black with animated gradient mesh & faint grid |
| **Glass Card** | `bg-white/[0.035] backdrop-blur-xl border-white/10 rounded-2xl` | Elevated frosted-glass surface |
| **Neon Cyan** | `#22D3EE` | Primary, Live Status, Informational, Edge Reticles |
| **Neon Green** | `#4ADE80` | Operational status, safe threats, camera active |
| **Neon Amber** | `#FBBF24` | Elevated threats, violence score, review chips |
| **Neon Red** | `#F43F5E` | Critical weapons, active buzzer alarm, escalation |
| **Neon Violet**| `#A78BFA` | Audit logs, secondary telemetry |
| **Typography** | `Inter` (UI) & `JetBrains Mono` (KPIs, IDs, Timestamps) | Readable, high contrast, WCAG AA compliant |

---

## 4. Connecting the Python YOLO & S3D Backend

The web app is completely decoupled via `src/lib/api.ts`. By default, it operates with mock data. To connect your live Python surveillance engine (`weights/weapon_yolo.pt` + `S3D` action recognition):

### Step A: Configure Environment Variable
Create `.env.local` inside the `web/` directory:
```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

### Step B: FastAPI Backend Endpoints
Expose the following endpoints in your Python backend (e.g. `src/fasttrack/api/`):

1. **`GET /health`**
   ```python
   @app.get("/health")
   def health_check():
       return {"status": "healthy", "gpu": "RTX 3050", "cuda": True}
   ```

2. **`GET /api/incidents`**
   Returns list of recorded threat incidents matching `Incident` interface in `src/lib/types.ts`.

3. **`POST /api/incidents/{id}/status`**
   ```json
   {
     "action": "CONFIRMED_THREAT",
     "actor": "Operator_Lead",
     "notes": "Verified firearm, dispatching field response"
   }
   ```

4. **`GET /api/cameras`**
   Returns camera stream registry (`id`, `name`, `rtspUrl`, `status`, `fps`).

5. **`GET /api/metrics`**
   Returns live GPU, CPU, RAM, and SQLite WAL database metrics.

6. **`WebSocket /ws/live/{cameraId}`**
   Streams live detection bounding boxes and frame telemetry:
   ```json
   {
     "threatLevel": "CRITICAL",
     "compositeThreatScore": 0.89,
     "weaponScore": 0.94,
     "violenceScore": 0.35,
     "fps": 30.0,
     "latencyMs": 14.2,
     "boxes": [
       { "id": "b1", "label": "Gun", "confidence": 0.94, "isThreat": true, "x": 55, "y": 38, "width": 14, "height": 22 }
     ]
   }
   ```

---

## 5. Directory Structure

```
web/
├── src/
│   ├── app/
│   │   ├── globals.css              # Mesh gradient, grid patterns, glass classes
│   │   ├── layout.tsx               # Root layout: fonts, smooth scroll, sidebar, topbar
│   │   ├── page.tsx                 # 01 Dashboard: KPIs, donut chart, incidents, timeline
│   │   ├── live/
│   │   │   └── page.tsx             # 02 Live Monitor: video wall, telemetry, settings
│   │   ├── incidents/
│   │   │   └── page.tsx             # 03 Incidents: searchable registry & audit logs
│   │   ├── models/
│   │   │   └── page.tsx             # 04 AI Models: model cards, benchmarks, architecture
│   │   └── settings/
│   │       └── page.tsx             # 05 Settings: cameras, hardware telemetry, users, rules
│   ├── components/
│   │   ├── layout/
│   │   │   ├── sidebar-nav.tsx      # Slim collapsible sidebar (5 items + mobile bottom nav)
│   │   │   └── top-bar.tsx          # Sticky glass header with clock, review chip, user avatar
│   │   ├── ui/
│   │   │   ├── glass-card.tsx       # Reusable frosted glass card with lift & neon glow
│   │   │   ├── kpi-card.tsx         # Count-up KPI metric card with trends
│   │   │   ├── severity-badge.tsx   # Neon severity pill badges
│   │   │   ├── status-chip.tsx      # Plain-English status chips
│   │   │   ├── live-badge.tsx       # Pulsing neon live indicator dot
│   │   │   ├── threat-gauge.tsx     # Arc threat index meter with dynamic SVG fill
│   │   │   └── detection-settings-popover.tsx # Popover with threshold sliders & siren toggle
│   │   ├── live/
│   │   │   └── video-panel.tsx      # Video canvas, bounding box reticles, Web Audio buzzer
│   │   └── incidents/
│   │       └── incident-drawer.tsx  # Forensic slide-over drawer with chain of custody
│   ├── lib/
│   │   ├── types.ts                 # Clean TypeScript definitions
│   │   ├── mock-data.ts             # Realistic mock incidents, cameras, AI models
│   │   ├── store.ts                 # Zustand lightweight store for triage & stream state
│   │   ├── api.ts                   # Isolated API client layer with backend fallback
│   │   └── utils.ts                 # Formatters and tailwind-merge helper
│   └── providers/
│       └── smooth-scroll.tsx        # Lenis smooth scrolling provider
├── tailwind.config.ts               # Custom navy colors, neon accents, glow shadows
├── tsconfig.json
├── postcss.config.mjs
├── next.config.mjs
└── package.json
```
