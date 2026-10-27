-- FastTrack AI Initial Database Schema V001
-- SQLite with WAL mode, foreign keys, and audit indices

-- Schema migrations tracking table
CREATE TABLE IF NOT EXISTS _schema_migrations (
    version TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    applied_utc TEXT NOT NULL
);

-- Cameras table
CREATE TABLE IF NOT EXISTS cameras (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL,          -- 'file', 'webcam', 'rtsp'
    source_ref_masked TEXT NOT NULL,    -- Passwords/tokens strictly masked
    enabled INTEGER NOT NULL DEFAULT 1,
    zones_json TEXT NOT NULL DEFAULT '[]',
    thresholds_json TEXT NOT NULL DEFAULT '{}',
    schedule_json TEXT NOT NULL DEFAULT '{}',
    created_utc TEXT NOT NULL
);

-- Evidence items (snapshots and video clips)
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,                 -- 'snapshot', 'clip'
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    encrypted INTEGER NOT NULL DEFAULT 0,
    created_utc TEXT NOT NULL,
    expires_utc TEXT
);

-- Aggregated Incidents
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    severity TEXT NOT NULL,             -- 'INFO', 'LOW', 'MEDIUM', 'HIGH'
    status TEXT NOT NULL DEFAULT 'new', -- 'new', 'reviewed', 'false_alarm', 'escalated'
    start_utc TEXT NOT NULL,
    end_utc TEXT,
    peak_weapon_score REAL NOT NULL DEFAULT 0.0,
    peak_violence_score REAL NOT NULL DEFAULT 0.0,
    summary TEXT NOT NULL DEFAULT ''
);

-- Atomic Events
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    incident_id TEXT REFERENCES incidents(id) ON DELETE SET NULL,
    camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    ts_utc TEXT NOT NULL,
    event_type TEXT NOT NULL,           -- 'weapon_detection', 'possible_violence', 'system_alert'
    class_name TEXT NOT NULL,
    raw_score REAL NOT NULL,
    calibrated_score REAL NOT NULL,
    bbox_json TEXT,                     -- [x1, y1, x2, y2]
    track_id INTEGER,
    model_version_id TEXT,
    evidence_id TEXT REFERENCES evidence(id) ON DELETE SET NULL
);

-- Operator Review Actions (Append-only audit trail)
CREATE TABLE IF NOT EXISTS review_actions (
    id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    action TEXT NOT NULL,               -- 'mark_reviewed', 'mark_false_alarm', 'escalate', 'add_note'
    note TEXT NOT NULL,
    ts_utc TEXT NOT NULL
);

-- Users & RBAC
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,                 -- 'admin', 'reviewer', 'viewer'
    active INTEGER NOT NULL DEFAULT 1,
    last_login_utc TEXT
);

-- Model Versions & Evaluation Artifacts
CREATE TABLE IF NOT EXISTS model_versions (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,                 -- 'weapon', 'violence'
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    trained_utc TEXT NOT NULL,
    metrics_path TEXT,
    active INTEGER NOT NULL DEFAULT 0
);

-- Evaluation Runs History
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY,
    model_version_id TEXT REFERENCES model_versions(id) ON DELETE SET NULL,
    dataset_name TEXT NOT NULL,
    run_utc TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    report_path TEXT
);

-- Real-Time Camera & Pipeline Health Samples
CREATE TABLE IF NOT EXISTS health_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    fps REAL NOT NULL,
    latency_ms REAL NOT NULL,
    dropped_frames INTEGER NOT NULL,
    gpu_util REAL NOT NULL DEFAULT 0.0,
    cpu_percent REAL NOT NULL DEFAULT 0.0,
    ram_percent REAL NOT NULL DEFAULT 0.0
);

-- Application Settings Key-Value Store
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_by TEXT NOT NULL,
    updated_utc TEXT NOT NULL
);

-- Performance and Audit Indices
CREATE INDEX IF NOT EXISTS idx_incidents_camera_ts ON incidents(camera_id, start_utc);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity);
CREATE INDEX IF NOT EXISTS idx_events_incident ON events(incident_id);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_utc);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_review_actions_incident ON review_actions(incident_id, ts_utc);
CREATE INDEX IF NOT EXISTS idx_health_samples_camera_ts ON health_samples(camera_id, ts_utc);
