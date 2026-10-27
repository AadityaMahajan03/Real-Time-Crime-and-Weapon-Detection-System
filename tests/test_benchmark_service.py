"""Unit and integration tests for BenchmarkService and model registry management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.fasttrack.ml.benchmark import BenchmarkService
from src.fasttrack.storage.db import DatabaseManager


@pytest.fixture
def temp_eval_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with mock evaluation JSON reports."""
    eval_dir = tmp_path / "reports" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    weapon_eval = {
        "mAP_50": 0.814,
        "mAP_50_95": 0.582,
        "precision": 0.842,
        "recall": 0.776,
        "f1_score": 0.808,
        "per_class_metrics": {
            "knife": {"precision": 0.856, "recall": 0.792, "ap_50": 0.824, "validation_threshold": 0.45, "support_count": 420}
        },
        "confusion_matrix": {
            "knife": {"true_positive": 333, "false_positive": 56, "false_negative": 87}
        },
    }
    with open(eval_dir / "weapon_eval.json", "w", encoding="utf-8") as f:
        json.dump(weapon_eval, f)

    weapon_bench = {
        "pytorch": {"p50_ms": 14.5, "fps": 68.7},
        "onnxruntime": {"p50_ms": 27.1, "fps": 35.1},
    }
    with open(eval_dir / "weapon_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(weapon_bench, f)

    violence_eval = {
        "s3d_baseline": {"roc_auc": 0.884, "precision": 0.835, "recall": 0.812},
        "r3d18_comparison": {"roc_auc": 0.841, "precision": 0.792, "recall": 0.770},
    }
    with open(eval_dir / "violence_eval.json", "w", encoding="utf-8") as f:
        json.dump(violence_eval, f)

    sys_bench = {
        "camera_benchmarks": {
            "1_cameras": {"aggregate_fps": 48.0, "latency": {"mean_ms": 20.0}}
        }
    }
    with open(eval_dir / "system_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(sys_bench, f)

    return eval_dir




def test_benchmark_service_load_reports(temp_eval_dir: Path) -> None:
    """Test loading existing reports and graceful None on missing files."""
    service = BenchmarkService(eval_dir=str(temp_eval_dir))

    weapon_eval = service.get_weapon_evaluation()
    assert weapon_eval is not None
    assert weapon_eval["mAP_50"] == 0.814
    assert "_artifact_path" in weapon_eval
    assert "_modified_utc" in weapon_eval

    weapon_bench = service.get_weapon_benchmark()
    assert weapon_bench is not None
    assert weapon_bench["pytorch"]["fps"] == 68.7

    violence_eval = service.get_violence_evaluation()
    assert violence_eval is not None
    assert violence_eval["s3d_baseline"]["roc_auc"] == 0.884

    sys_bench = service.get_system_benchmark()
    assert sys_bench is not None
    assert "1_cameras" in sys_bench["camera_benchmarks"]

    # Missing file returns None
    assert service.get_weapon_dataset_audit() is None


def test_precision_recall_curve_generation(temp_eval_dir: Path) -> None:
    """Test generating precision-recall curve points from report metrics."""
    service = BenchmarkService(eval_dir=str(temp_eval_dir))
    pr_data = service.get_precision_recall_curve_data()

    assert "recalls" in pr_data
    assert "precisions" in pr_data
    assert "thresholds" in pr_data
    assert len(pr_data["recalls"]) == len(pr_data["precisions"])
    assert pr_data["mAP_50"] == 0.814

    # Test empty fallback when report missing
    empty_service = BenchmarkService(eval_dir="nonexistent/dir")
    empty_pr = empty_service.get_precision_recall_curve_data()
    assert empty_pr["recalls"] == []
    assert empty_pr["precisions"] == []


def test_roc_curve_generation(temp_eval_dir: Path) -> None:
    """Test generating ROC curve points for S3D and R3D-18."""
    service = BenchmarkService(eval_dir=str(temp_eval_dir))
    roc_data = service.get_roc_curve_data()

    assert "fpr" in roc_data
    assert "tpr_s3d" in roc_data
    assert "tpr_r3d" in roc_data
    assert roc_data["s3d_auc"] == 0.884
    assert roc_data["r3d_auc"] == 0.841
    assert roc_data["fpr"][0] == 0.0
    assert roc_data["fpr"][-1] == 1.0

    # Test empty fallback when report missing
    empty_service = BenchmarkService(eval_dir="nonexistent/dir")
    empty_roc = empty_service.get_roc_curve_data()
    assert empty_roc["fpr"] == []


def test_sync_and_switch_models_in_db(temp_eval_dir: Path, in_memory_db: DatabaseManager) -> None:
    """Test syncing model versions to DB, listing, and switching active checkpoints."""
    service = BenchmarkService(eval_dir=str(temp_eval_dir), db_manager=in_memory_db)

    # 1. Sync
    synced = service.sync_model_versions_to_db()
    assert synced == 4

    # 2. List all
    all_models = service.list_model_versions()
    assert len(all_models) == 4

    # Filter by kind
    weapon_models = service.list_model_versions(kind="weapon")
    assert len(weapon_models) == 2
    violence_models = service.list_model_versions(kind="violence")
    assert len(violence_models) == 2

    # Check active status
    active_weapon = [m for m in weapon_models if m["active"]]
    assert len(active_weapon) == 1
    assert active_weapon[0]["id"] == "mv-yolo11n-01"

    # 3. Switch active model
    success = service.set_active_model_version("mv-yolo11n-onnx-01")
    assert success is True

    updated_weapons = service.list_model_versions(kind="weapon")
    onnx_model = next(m for m in updated_weapons if m["id"] == "mv-yolo11n-onnx-01")
    py_model = next(m for m in updated_weapons if m["id"] == "mv-yolo11n-01")
    assert onnx_model["active"] == 1
    assert py_model["active"] == 0


def test_benchmark_service_without_db() -> None:
    """Test that DB methods safely handle db_manager=None."""
    service = BenchmarkService(eval_dir="reports/eval", db_manager=None)
    assert service.sync_model_versions_to_db() == 0
    assert service.list_model_versions() == []
    assert service.set_active_model_version("any-id") is False
