-- FastTrack AI Normalized Surveillance Schema V002
-- Extends V001 with continuous recordings, forensic clips, audio alarms, zones, detections, and audit chain

-- 1. Continuous Segmented Recordings
CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    segment_path TEXT NOT NULL,
    start_utc TEXT NOT NULL,
    end_utc TEXT,
    duration_sec INTEGER NOT NULL DEFAULT 60,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL DEFAULT '',
    is_locked INTEGER NOT NULL DEFAULT 0, -- 1 if linked to an active/reviewed incident (never delete)
    status TEXT NOT NULL DEFAULT 'recording', -- 'recording', 'completed', 'corrupt'
    created_utc TEXT NOT NULL
);

-- 2. Forensic Incident Video Clips
CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    raw_clip_path TEXT NOT NULL,
    annotated_clip_path TEXT NOT NULL,
    thumbnail_path TEXT NOT NULL,
    keyframe_strip_path TEXT,
    sha256 TEXT NOT NULL,
    duration_sec REAL NOT NULL DEFAULT 0.0,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    codec TEXT NOT NULL DEFAULT 'h264',
    verification_status TEXT NOT NULL DEFAULT 'verified', -- 'verified', 'corrupt', 'pending'
    created_utc TEXT NOT NULL
);

-- 3. Incident Evidence Many-to-Many Linking
CREATE TABLE IF NOT EXISTS incident_evidence (
    incident_id TEXT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
    attached_utc TEXT NOT NULL,
    PRIMARY KEY (incident_id, evidence_id)
);

-- 4. Surveillance ROI Zones & Exclusion Masks
CREATE TABLE IF NOT EXISTS zones (
    id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    zone_type TEXT NOT NULL DEFAULT 'include', -- 'include', 'exclude', 'tripwire'
    polygon_json TEXT NOT NULL DEFAULT '[]', -- [[x1, y1], [x2, y2], ...]
    enabled INTEGER NOT NULL DEFAULT 1,
    created_utc TEXT NOT NULL
);

-- 5. Persistent Detections / Tracking Records
CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY,
    incident_id TEXT REFERENCES incidents(id) ON DELETE SET NULL,
    camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    ts_utc TEXT NOT NULL,
    detection_type TEXT NOT NULL DEFAULT 'weapon', -- 'weapon', 'violence', 'ambient'
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    calibrated_score REAL NOT NULL,
    bbox_json TEXT NOT NULL DEFAULT '[]', -- [x1, y1, x2, y2]
    track_id INTEGER,
    is_threat INTEGER NOT NULL DEFAULT 1
);

-- 6. Acoustic Alarm Events Audit Log
CREATE TABLE IF NOT EXISTS alarms (
    id TEXT PRIMARY KEY,
    incident_id TEXT REFERENCES incidents(id) ON DELETE SET NULL,
    camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    alarm_type TEXT NOT NULL, -- 'weapon_siren', 'violence_beep', 'manual_test'
    pattern TEXT NOT NULL DEFAULT 'continuous', -- 'continuous', 'pulsed'
    start_utc TEXT NOT NULL,
    stop_utc TEXT,
    duration_sec REAL NOT NULL DEFAULT 0.0,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    ack_by TEXT,
    ack_utc TEXT,
    notes TEXT NOT NULL DEFAULT ''
);

-- 7. Model Registry
CREATE TABLE IF NOT EXISTS model_registry (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL, -- 'object_detection', 'action_recognition', 'crop_verifier', 'pose'
    version TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    sha256 TEXT NOT NULL DEFAULT '',
    is_active INTEGER NOT NULL DEFAULT 1,
    classes_json TEXT NOT NULL DEFAULT '[]',
    created_utc TEXT NOT NULL
);

-- 8. Hash-Chained Audit Trail (Tamper-evident)
CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    ts_utc TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    details TEXT NOT NULL,
    prev_hash TEXT NOT NULL DEFAULT '',
    entry_hash TEXT NOT NULL
);

-- Indices for rapid querying
CREATE INDEX IF NOT EXISTS idx_recordings_camera_start ON recordings(camera_id, start_utc);
CREATE INDEX IF NOT EXISTS idx_recordings_locked ON recordings(is_locked);
CREATE INDEX IF NOT EXISTS idx_clips_incident ON clips(incident_id);
CREATE INDEX IF NOT EXISTS idx_incident_evidence_inc ON incident_evidence(incident_id);
CREATE INDEX IF NOT EXISTS idx_zones_camera ON zones(camera_id);
CREATE INDEX IF NOT EXISTS idx_detections_inc_ts ON detections(incident_id, ts_utc);
CREATE INDEX IF NOT EXISTS idx_detections_cam_ts ON detections(camera_id, ts_utc);
CREATE INDEX IF NOT EXISTS idx_alarms_incident ON alarms(incident_id);
CREATE INDEX IF NOT EXISTS idx_alarms_start ON alarms(start_utc);
CREATE INDEX IF NOT EXISTS idx_audit_log_ts ON audit_log(ts_utc);
