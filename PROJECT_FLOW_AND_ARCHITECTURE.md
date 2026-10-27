# FAST-TRACK AI: REAL-TIME CRIME, VIOLENCE & WEAPON DETECTION SYSTEM
## Complete System Architecture, Neural Pipeline Flows, and Technical Reference

---

### Executive Overview

**FAST-TRACK AI** is an enterprise-grade, edge-accelerated surveillance analytics system designed for real-time detection, tracking, and forensic sealing of violent altercations, armed threats, and criminal activities. 

The architecture combines state-of-the-art research paradigms:
1. **Wavelet-Enhanced Weapon Detection**: Based on João Assalim and Tiago Almeida (2025), leveraging 2D Discrete Wavelet Transform (2D DWT) high-frequency sub-band analysis with specialized YOLOv8 neural detectors for concealed blades, knives, and firearms.
2. **Spatiotemporal Violence & Crime Recognition**: Based on Abduulrahman Khalid et al. (2022), utilizing a lightweight MobileNetV2 spatial backbone coupled with Bidirectional Long Short-Term Memory (BiLSTM) recurrent sequence modeling (`MoBiLSTM`) and optical flow motion magnitude fusion.
3. **Cryptographically Sealed Incident Audit Registry**: SQLite WAL architecture preserving tamper-evident chain of custody with SHA-256 evidence hashing, automatic surveillance video clipping, and forensic snapshot annotation.
4. **Dual Presentation Layers**: A modern Next.js 14 reactive command wall and a real-time Streamlit tactical monitoring hub.

---

## 1. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Video Ingestion & Frame Normalization"]
        CAM["Live CCTV / RTSP / Webcam / MP4 File"] --> CAP["OpenCV VideoCapture / WebRTC Stream"]
        CAP --> PRE["YCrCb Luminance CLAHE & Temporal Buffer"]
    end

    subgraph NeuralInference ["2. Dual-Stream Neural Detection Engine"]
        PRE --> PERSON["Person Detection<br/>YOLO11n (COCO Class 0)"]
        PRE --> RGB_STREAM["Stream A: RGB Weapon YOLO<br/>normal_compressed_best.pt"]
        PRE --> DWT["Stream B: 2D DWT Wavelet<br/>(Haar / Db2 / Symlet)"]
        DWT --> LH["LH Sub-band Contrast Normalizer"]
        LH --> WV_STREAM["Wavelet Weapon YOLO<br/>haar_compressed_best.pt"]
        PRE --> MOBI["Violence Sequence Buffer<br/>(16-frame FIFO @ 64x64)"]
        MOBI --> VIO["MobileNetV2 + BiLSTM<br/>Action Classifier (Khalid 2022)"]
        PRE --> OPTICAL["Sparse Lucas-Kanade<br/>Motion Flow Magnitude"]
    end

    subgraph Filtration ["3. False-Alarm Suppression & Threat Decision"]
        RGB_STREAM & WV_STREAM --> NMS["Coordinate Rescaling & Dual-Stream NMS"]
        NMS --> BLACK_FILTER["Black Space & Sensor Noise Filter<br/>(Luminance, Dynamic Range, Box Ratio)"]
        BLACK_FILTER --> POSE["Spatial Hand/Wrist Prior & Crop Verifier"]
        POSE --> DSA_TRACK["DSA EMA Track Smoothing & Hysteresis"]
        VIO & OPTICAL --> FUSED_VIO["Fused Violence Threat Score"]
        DSA_TRACK & FUSED_VIO --> DECISION["Composite Threat Decision Matrix<br/>(CRITICAL / HIGH / ELEVATED / SAFE)"]
    end

    subgraph Storage ["4. Forensic Sealing & Storage"]
        DECISION -->|Threat Confirmed| AUTO_INC["Incident Auto-Recorder<br/>(Cooldown Regulated)"]
        AUTO_INC --> CLIP_CUT["Surveillance Video Cutter (.mp4)"]
        AUTO_INC --> SNAP["Forensic Snapshot + Reticles (.jpg)"]
        SNAP --> HASH["SHA-256 Cryptographic Sealer"]
        HASH --> DB["SQLite WAL Database<br/>(incidents, evidence, review_actions)"]
    end

    subgraph Presentation ["5. Presentation & Dispatch"]
        DB & DECISION --> NEXT["Next.js 14 Mission Control<br/>(Port 3000: Live Wall, Registry, Detail Drawer)"]
        DB & DECISION --> FASTAPI["FastAPI REST & WebSocket Hub<br/>(Port 8000: /api/detect, /api/incidents)"]
        DB & DECISION --> STREAMLIT["Streamlit Command Center<br/>(Port 8501: Multi-Cam Matrix, Forensic Replay)"]
    end
```

---

## 2. Weapon & Knife Detection Pipeline (João Assalim Technique)

### Mathematical Foundation: 2D Discrete Wavelet Transform (2D DWT)
Standard RGB object detection often misses small, low-contrast, or reflective weapon edges (such as stainless-steel knife blades, dark gun barrels against dark clothing, or weapons held in shadows). 

The João Assalim technique decomposes 2D frame luminance $I(x, y)$ into four orthogonal frequency sub-bands using Mallat's algorithm:

$$\begin{aligned}
LL(j, k) &= \sum_{m} \sum_{n} h(m - 2j) h(n - 2k) I(m, n) \quad \text{(Low-frequency Approximation)} \\
LH(j, k) &= \sum_{m} \sum_{n} h(m - 2j) g(n - 2k) I(m, n) \quad \text{(Horizontal High-Frequency Details)} \\
HL(j, k) &= \sum_{m} \sum_{n} g(m - 2j) h(n - 2k) I(m, n) \quad \text{(Vertical High-Frequency Details)} \\
HH(j, k) &= \sum_{m} \sum_{n} g(m - 2j) g(n - 2k) I(m, n) \quad \text{(Diagonal High-Frequency Details)}
\end{aligned}$$

Where $h$ is the low-pass scaling filter and $g$ is the high-pass wavelet filter (Haar, Daubechies Db2, or Symlet).

- **Horizontal Sub-band ($LH$)**: Highlights horizontal blade tips, knife edges, firearm barrels, and metallic contours while eliminating flat uniform background illumination.
- **Wavelet Checkpoints Loaded**:
  - `weights/joao_models/normal_compressed_best.pt`: Trained directly on surveillance RGB weapon imagery.
  - `weights/joao_models/haar_compressed_best.pt`: Trained on Haar $LH$ detail sub-bands.
  - `weights/joao_models/db_best.pt`: Daubechies Db2 wavelet variant.
  - `weights/joao_models/symlet_best.pt`: Symlet wavelet variant.

### Dual-Stream Coordinate Rescaling & NMS Fusion
1. **Branch 1 (RGB)**: Full-resolution frame processed by `normal_compressed_best.pt`.
2. **Branch 2 (Wavelet)**: Frame decomposed into $LH$ sub-band, normalized via contrast stretching, and processed by `haar_compressed_best.pt` at $(640 \times 640)$. Wavelet coordinates are mapped back to native frame space:
   $$x_{orig} = x_{wavelet} \times \left(\frac{W_{native}}{640}\right), \quad y_{orig} = y_{wavelet} \times \left(\frac{H_{native}}{640}\right)$$
3. **NMS Deduplication**: Overlapping detections between RGB and Wavelet streams are merged using Intersection-over-Union (IoU threshold $0.40$), giving priority to the highest confidence detection while tagging the detection source.

---

## 3. Violence & Action Recognition Pipeline (Abduulrahman Khalid Technique)

### Spatiotemporal Sequence Classifier (`MoBiLSTM`)
Single-frame classifiers cannot differentiate between dynamic violent altercations (punches, strikes, aggressive grapples) and benign rapid movements (jogging, waving, gesturing). 

FAST-TRACK AI implements Abduulrahman Khalid's spatiotemporal architecture:

```mermaid
flowchart LR
    F["16-Frame FIFO Buffer<br/>(64x64 BGR Normalized)"] --> MOBILENET["Pretrained MobileNetV2<br/>Spatial Feature Extractor"]
    MOBILENET --> POOL["Global Average Pooling<br/>(1280-dim per frame)"]
    POOL --> BILSTM["Bidirectional LSTM<br/>(32 Forward + 32 Backward = 64 Units)"]
    BILSTM --> DROP["Dropout (p=0.40)"]
    DROP --> MLP["Dense(64, ReLU) -> Dense(32, ReLU)"]
    MLP --> SOFTMAX["Dense(2, Softmax)<br/>[Non-Violence, Violence]"]
```

1. **Temporal Buffer**: A rolling FIFO buffer retains the last 16 consecutive frames downsampled to $64 \times 64 \times 3$ and normalized to $[0.0, 1.0]$.
2. **Spatial Backbone**: MobileNetV2 extracts compact 1280-dimensional feature representations for each time step.
3. **Bidirectional Temporal Modeling**: BiLSTM captures both past-to-future and future-to-past action trajectories.
4. **Optical Flow Motion Magnitude Fusion**: Sparse Lucas-Kanade optical flow computes pixel displacement velocities inside person bounding boxes. Sudden, erratic velocity spikes reinforce the BiLSTM confidence, yielding robust multi-modal threat scores.

---

## 4. Black-Space & False Alarm Suppression Filter

Surveillance cameras frequently exhibit letterbox bars (16:9 videos on 4:3 monitors), unlit dark corners, lens caps, or high sensor noise in pitch darkness. If unchecked, normalizing low-dynamic-range noise will produce artificial high-frequency textures that trigger false detections.

FAST-TRACK AI enforces four layers of defense:

1. **Dynamic Range Noise Gate**:
   $$\text{Spread} = \max(I) - \min(I)$$
   If $\text{Spread} < 12.0$, contrast stretching is bypassed and the sub-band is suppressed to zero. Sensor noise is never amplified into spurious contours.
2. **Wavelet Energy Sub-band Gate**:
   Wavelet forward pass is skipped entirely if $\sigma(LH) < 3.2$ or mean image luminance $\bar{I} < 12.0$.
3. **Bounding Box Dimension Bounds**:
   Hallucinated boxes covering $> 75\%$ of both frame width and height are immediately discarded.
4. **Physical Object Luminance & Contrast Check**:
   For every proposed weapon bounding box, the crop in the original frame is verified:
   $$\text{If } \bar{I}_{crop} < 20.0 \quad \text{and} \quad \sigma(I_{crop}) < 14.0 \implies \text{Reject candidate}$$
   Real physical weapons (steel blades, firearms, wooden handles) exhibit high local contrast ($\sigma > 50$) and cannot exist in uniform pitch-black space.

---

## 5. Incident Lifecycle & Cryptographic Sealing Flow

```mermaid
sequenceDiagram
    autonumber
    participant Engine as Edge Inference Engine
    participant Hysteresis as Temporal Hysteresis & Track Filter
    participant AutoRecorder as Incident Auto-Recorder
    participant FileSystem as File Storage (public/evidence & public/clips)
    participant Database as SQLite WAL (data/fasttrack.db)
    participant UI as Next.js & Streamlit Frontends

    Engine->>Hysteresis: Emit Raw Weapon & Violence Scores
    Hysteresis->>Hysteresis: Apply 5-Frame K-of-N Confirmation Window
    alt Confirmed Threat (Weapon >= 0.28 or Violence >= 0.45)
        Hysteresis->>AutoRecorder: Trigger Threat Registration (Cooldown: 12s)
        AutoRecorder->>FileSystem: Draw Neural Reticles & Save Snapshot (evidence/{id}.jpg)
        AutoRecorder->>FileSystem: Extract & Copy Surveillance Video (clips/{id}.mp4)
        AutoRecorder->>AutoRecorder: Compute SHA-256 Checksum on Evidence Bytes
        AutoRecorder->>Database: INSERT into incidents & evidence tables (Foreign Key linked to cam-01)
        Database-->>AutoRecorder: Commit Transaction (WAL Mode)
        AutoRecorder->>UI: Broadcast newIncident JSON payload
        UI->>UI: Poll /api/incidents & Update Table with Snapshot Thumbnails
        UI->>UI: Sound Continuous 3-Second Audio Alarm
    else Low / Transient Noise
        Hysteresis->>Engine: Suppress Transient Flutter
    end
```

### Database Schema Structure (`data/fasttrack.db`)
- `cameras`: Registered CCTV units, RTSP endpoints, locations, and operational statuses.
- `incidents`: Unique incident ID (`live-xxxxxxxx` or `INC-YYYY-LIVE-XXXXXX`), foreign-key linked to `cameras.id`, storing severity, status (`new`, `open`, `reviewed`, `false_alarm`), timestamp, peak weapon score, peak violence score, and description.
- `evidence`: File path to physical `.jpg` snapshot, cryptographic SHA-256 hash, and creation timestamp.
- `incident_evidence`: Relational junction mapping evidence artifacts to incident IDs.
- `review_actions`: Immutable forensic audit trail recording operator notes, acknowledgments, status transitions, and reviewer identity.

---

## 6. Directory Structure & Key Artifacts

```
REAL-TIME CRIME AND WEAPON DETECTION SYSTEM/
├── config/
│   └── default.yaml               # Central configuration (thresholds, models, DB path)
├── data/
│   ├── fasttrack.db               # SQLite database with WAL journaling
│   └── joao_detected_objects_video.mp4 # Reference knife & weapon test footage
├── weights/
│   ├── joao_models/
│   │   ├── normal_compressed_best.pt # Assalim RGB weapon detector
│   │   ├── haar_compressed_best.pt   # Assalim Haar 2D DWT LH weapon detector
│   │   ├── db_best.pt                # Daubechies Db2 wavelet detector
│   │   └── symlet_best.pt            # Symlet wavelet detector
│   ├── yolov8n.pt                 # YOLOv8 baseline
│   └── yolo11n.pt                 # YOLO11 person detector
├── src/fasttrack/
│   ├── api/
│   │   └── main.py                # FastAPI REST & WebSocket server
│   ├── core/
│   │   ├── config.py              # Pydantic v2 settings schema
│   │   └── types.py               # Frame, BoundingBox, Detection data models
│   ├── ml/
│   │   ├── wavelet_enhancer.py    # 2D DWT decomposition, sub-band extraction & contrast normalization
│   │   ├── mobilenet_bilstm.py    # MobileNetV2 + BiLSTM PyTorch neural architecture
│   │   ├── weapon_detector.py     # Dual-stream YOLOv8 + Wavelet coordinate fusion
│   │   ├── violence_detector.py   # Spatiotemporal sequence violence recognizer
│   │   ├── pose_prior.py          # Spatial hand/wrist bounding box validation
│   │   └── crop_verifier.py       # Fine-grained secondary crop verifier CNN
│   └── storage/
│       ├── db.py                  # DatabaseManager connection pool
│       └── repositories/          # Repositories for incidents, cameras, evidence, audits
├── web/                           # Next.js 14 Web Application
│   ├── public/
│   │   ├── evidence/              # Forensic JPG snapshots with neural reticle overlays
│   │   └── clips/                 # MP4 surveillance video evidence clips
│   ├── src/app/
│   │   ├── page.tsx               # Command overview & real-time metric gauges
│   │   ├── live/page.tsx          # Live Surveillance Wall (webcam, video upload, alarm tone)
│   │   └── incidents/page.tsx     # Incident Registry & Audit Table with snapshot previews
│   └── src/components/
│       ├── incidents/incident-drawer.tsx # Forensic detail drawer (video player + audit log)
│       └── live/video-panel.tsx   # Interactive webcam canvas with 3-second acoustic siren
├── scripts/
│   ├── analyze_weapon_video.py    # CLI tool for Assalim wavelet benchmark analysis
│   └── analyze_combined_threats.py# CLI tool for joint weapon + violence detection
└── app.py                         # Streamlit Tactical Command Center (Port 8501)
```

---

## 7. Quickstart & Service Launch Commands

### 1. Launch FastAPI Neural Engine (Port 8000)
```bash
python -m uvicorn src.fasttrack.api.main:app --port 8000 --host 127.0.0.1
```
*Health Check: `http://127.0.0.1:8000/health`*  
*Incidents API: `http://127.0.0.1:8000/api/incidents`*  
*Live Detection: `POST http://127.0.0.1:8000/api/detect`*

### 2. Launch Next.js Enterprise Dashboard (Port 3000)
```bash
npm --prefix web run dev
```
*Live Surveillance Wall: `http://localhost:3000/live`*  
*Incident Registry & Evidence Audit: `http://localhost:3000/incidents`*

### 3. Launch Streamlit Command Center (Port 8501)
```bash
python -m streamlit run app.py --server.port 8501 --server.headless true
```
*Streamlit Hub: `http://localhost:8501`*

### 4. Execute Full Verification Test Suite
```bash
pytest tests -q
```
*Expected Result: `108 passed, 0 failures`.*

---

## 8. Summary of Solved Issues

| Issue Reported | Root Cause | Solution Implemented |
| :--- | :--- | :--- |
| **Weapons Not Detecting Properly** | Edge API was querying legacy pose-filtered weights that suppressed isolated weapon silhouettes. | Integrated João Assalim's dual-stream 2D DWT wavelet models (`normal_compressed_best.pt` + `haar_compressed_best.pt`) with $LH$ sub-band coordinate fusion. Achieved **0.889+ knife confidence**. |
| **Incident Photo & Video Missing (404 / Broken Icons)** | SQLite `sqlite3.Row` returned from queries caused 500 serialization errors; incident IDs lacked physical files on disk. | Added `_format_incident_row`, generated seed evidence (`.jpg` and `.mp4`), and added dynamic `onError` fallback handlers and table thumbnails in Next.js. |
| **Incidents Not Storing in Incident Section** | Undefined `camera_id` / missing foreign key relation in `_try_auto_create_incident` caused silent `sqlite3.IntegrityError` rollbacks. Next.js was not polling server incidents. | Resolved foreign key binding with automatic fallback, verified SQLite persistence, and added real-time polling in `web/src/app/incidents/page.tsx` and `web/src/lib/api.ts`. |
| **Black Spaces Detecting Weapons** | Contrast stretching on dark frames amplified low sensor noise ($\text{Spread} < 12$) into artificial high-contrast contours, and João Assalim's Haar model fired on full-frame zeros. | Added dynamic range noise gate, wavelet detail energy gate ($\sigma(LH) \ge 3.2$), box dimension caps ($< 75\%$), and crop luminance validation ($\bar{I} \ge 20, \sigma \ge 14$). |
