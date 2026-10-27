export type Severity = "critical" | "high" | "medium" | "low" | "safe";

export type IncidentStatus = "open" | "in_review" | "confirmed" | "false_alarm" | "resolved" | "escalated";

export interface BoundingBox {
  id: string;
  label: string;
  confidence: number;
  isThreat: boolean;
  isWarning?: boolean; // Yellow reticle for shape match / warning
  x: number; // percentage 0-100
  y: number; // percentage 0-100
  width: number; // percentage 0-100
  height: number; // percentage 0-100
}

export interface IncidentTimelineEvent {
  timestamp: string;
  description: string;
  actor: string;
}

export interface IncidentAuditEntry {
  id: string;
  timestamp: string;
  action: "CONFIRMED_THREAT" | "MARKED_FALSE_ALARM" | "ESCALATED" | "STATUS_CHANGED" | "NOTE_ADDED";
  actor: string;
  details: string;
}

export interface Incident {
  id: string;
  incidentCode: string;
  timestamp: string;
  cameraId: string;
  cameraName: string;
  location: string;
  severity: Severity;
  status: IncidentStatus;
  weaponType?: "knife" | "gun" | "grenade" | "none";
  peakWeaponScore: number;
  peakViolenceScore: number;
  compositeThreatScore: number;
  shortDescription: string;
  snapshotUrl: string;
  videoUrl?: string;
  sha256Hash: string;
  timeline: IncidentTimelineEvent[];
  auditLog: IncidentAuditEntry[];
}

export interface Camera {
  id: string;
  name: string;
  location: string;
  status: "online" | "offline" | "degraded";
  fps: number;
  resolution: string;
  activeThreatCount: number;
  thumbnailUrl: string;
  rtspUrl: string;
}

export interface DetectionTelemetry {
  threatLevel: "CRITICAL" | "HIGH" | "ELEVATED" | "NORMAL" | "SAFE";
  compositeThreatScore: number; // 0.0 - 1.0
  weaponScore: number; // 0.0 - 1.0
  violenceScore: number; // 0.0 - 1.0
  fps: number;
  latencyMs: number;
  activeDetections: BoundingBox[];
}

export interface AIModel {
  id: string;
  name: string;
  category: "Object Detection" | "Action Recognition" | "Tracking";
  version: string;
  backbone: string;
  accuracyMetric: string;
  accuracyValue: number; // percentage e.g. 94.2
  latencyMs: number;
  fps: number;
  status: "active" | "standby" | "evaluating";
  classes: string[];
  precision: number;
  recall: number;
  f1Score: number;
}

export interface DetectionSettings {
  weaponThreshold: number;
  violenceThreshold: number;
  fpsLimit: number;
  sirenEnabled: boolean;
  boxSmoothing: boolean;
  alertOnHighOnly: boolean;
}

export interface SystemMetrics {
  gpuUtilization: number;
  gpuMemoryUsedMb: number;
  gpuMemoryTotalMb: number;
  gpuName: string;
  cpuUtilization: number;
  ramUsagePercent: number;
  diskUsagePercent: number;
  sqliteWalSizeMb: number;
  uptimeSeconds: number;
}
