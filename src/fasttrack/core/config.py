"""Application configuration models and YAML/environment loaders using Pydantic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.fasttrack.core.errors import ConfigError

# Load variables from .env if present
load_dotenv()


class AppSettings(BaseModel):
    name: str = "FAST-TRACK AI"
    version: str = "1.0.0"
    environment: str = "development"
    demo_mode: bool = True
    clock: str = "system"


class StorageSettings(BaseModel):
    db_path: str = "data/fasttrack.db"
    evidence_dir: str = "evidence"
    logs_dir: str = "logs"
    weights_dir: str = "weights"
    reports_eval_dir: str = "reports/eval"
    min_disk_free_gb: float = 2.0


class LoggingSettings(BaseModel):
    level: str = "INFO"
    file_name: str = "logs/fasttrack.log"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


class WeaponDetectorSettings(BaseModel):
    model_checkpoint: str = "weights/weapon_yolo.pt"
    fallback_checkpoint: str = "weights/yolo11n.pt"
    input_size: int = 640
    default_confidence_threshold: float = 0.50
    per_class_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "knife": 0.48,
            "gun": 0.50,
            "pistol": 0.50,
            "rifle": 0.50,
            "weapon": 0.50,
            "grenade": 0.50,
            "scissors": 0.50,
            "baseball bat": 0.50,
        }
    )
    nms_iou_threshold: float = 0.45
    min_box_area_px: float = 400.0
    max_aspect_ratio: float = 8.0
    k_of_n: Dict[str, int] = Field(default_factory=lambda: {"k": 2, "n": 4})
    track_persistence_sec: float = 2.0
    tiled_inference: bool = False
    use_fp16: bool = False
    realistic_vision: bool = True
    benign_suppression: bool = True
    wavelet_mode: str = "none"  # 'none', 'haar', 'db2', 'sym2', 'composite'
    wavelet_checkpoint: Optional[str] = "weights/joao_models/haar_compressed_best.pt"
    enable_wavelet_fusion: bool = False


class ViolenceClassifierSettings(BaseModel):
    model_architecture: str = "s3d"  # 'mobilenet_bilstm', 's3d', or 'r3d_18'
    weights_path: str = "weights/s3d_kinetics400.pt"
    clip_seconds: float = 2.0
    clip_frames: int = 16
    image_size: int = 64
    stride_seconds: float = 0.5
    trigger_threshold: float = 0.65
    release_threshold: float = 0.40
    consecutive_windows: int = 2
    cooldown_seconds: float = 5.0
    ema_smoothing_alpha: float = 0.35
    temperature: float = 1.0


class EventEngineSettings(BaseModel):
    incident_merge_gap_seconds: float = 10.0
    incident_max_duration_seconds: float = 300.0
    camera_rate_limit_per_min: int = 6
    global_rate_limit_per_min: int = 24


class EvidenceSettings(BaseModel):
    pre_roll_seconds: float = 3.0
    post_roll_seconds: float = 4.0
    jpeg_quality: int = 85
    retention_days: int = 30
    encrypt_at_rest: bool = False


class UISettings(BaseModel):
    theme: str = "dark"
    display_fps: int = 10
    anonymize_faces: bool = False
    alert_sound: bool = True


class AppConfig(BaseModel):
    """Complete consolidated application configuration."""

    app: AppSettings = Field(default_factory=AppSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    weapon_detector: WeaponDetectorSettings = Field(default_factory=WeaponDetectorSettings)
    violence_classifier: ViolenceClassifierSettings = Field(default_factory=ViolenceClassifierSettings)
    event_engine: EventEngineSettings = Field(default_factory=EventEngineSettings)
    evidence: EvidenceSettings = Field(default_factory=EvidenceSettings)
    ui: UISettings = Field(default_factory=UISettings)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> AppConfig:
        """Load configuration from YAML file and apply environment variable overrides."""
        path = Path(config_path or os.getenv("FASTTRACK_CONFIG_PATH", "config/default.yaml"))
        data: Dict[str, Any] = {}

        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        data = loaded
            except Exception as e:
                raise ConfigError(f"Failed to parse config YAML at {path}: {e}")

        # Environment variable overrides
        if env_val := os.getenv("FASTTRACK_ENV"):
            data.setdefault("app", {})["environment"] = env_val
        if demo_val := os.getenv("FASTTRACK_DEMO_MODE"):
            data.setdefault("app", {})["demo_mode"] = demo_val.lower() in ("true", "1", "yes")
        if db_path := os.getenv("FASTTRACK_DATA_DIR"):
            data.setdefault("storage", {})["db_path"] = str(Path(db_path) / "fasttrack.db")
        if ev_dir := os.getenv("FASTTRACK_EVIDENCE_DIR"):
            data.setdefault("storage", {})["evidence_dir"] = ev_dir
        if log_lvl := os.getenv("FASTTRACK_LOG_LEVEL"):
            data.setdefault("logging", {})["level"] = log_lvl

        return cls(**data)


_CONFIG_INSTANCE: Optional[AppConfig] = None


def get_config(reload: bool = False, config_path: Optional[str] = None) -> AppConfig:
    """Retrieve global cached application configuration singleton."""
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None or reload or config_path is not None:
        _CONFIG_INSTANCE = AppConfig.load(config_path)
    return _CONFIG_INSTANCE
