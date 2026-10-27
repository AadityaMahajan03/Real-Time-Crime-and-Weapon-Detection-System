"""Unit tests for AnalyticsEngine: KPIs, MTTR distribution, trends, Pareto, and telemetry."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.fasttrack.analytics.engine import AnalyticsEngine
from src.fasttrack.core.clock import SimulatedClock
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.migrations import MigrationRunner


def test_analytics_kpis_and_mttr():
    """Verify KPI calculations, False Alarm Rate, and Mean Time to Review."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_analytics.db")
        db_mgr = DatabaseManager(db_path=db_path)
        MigrationRunner(db_mgr).run_migrations()

        clock = SimulatedClock(start_utc=datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc))
        analytics = AnalyticsEngine(db_manager=db_mgr)

        # Initially empty
        empty_kpis = analytics.get_kpis()
        assert empty_kpis["total_incidents"] == 0
        assert empty_kpis["false_alarm_rate_pct"] == 0.0
        assert empty_kpis["mttr_sec"] == 0.0

        # Seed camera
        cam_id = "cam-analytics-01"
        db_mgr.execute_non_query(
            "INSERT INTO cameras (id, name, source_type, source_ref_masked, created_utc) VALUES (?, ?, ?, ?, ?)",
            (cam_id, "Hallway Camera", "file", "test.mp4", clock.now_utc().isoformat()),
        )

        # Seed Incident 1: Reviewed (t_review = +60s)
        t1 = clock.now_utc()
        db_mgr.execute_non_query(
            """
            INSERT INTO incidents (id, camera_id, severity, status, start_utc, peak_weapon_score, peak_violence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("inc-01", cam_id, "HIGH", "reviewed", t1.isoformat(), 0.9, 0.8),
        )
        db_mgr.execute_non_query(
            "INSERT INTO review_actions (id, incident_id, user_id, action, note, ts_utc) VALUES (?, ?, ?, ?, ?, ?)",
            ("act-01", "inc-01", "op1", "mark_reviewed", "Verified threat", (t1 + timedelta(seconds=60)).isoformat()),
        )

        # Seed Incident 2: False Alarm (t_review = +120s)
        t2 = clock.now_utc() + timedelta(minutes=5)
        db_mgr.execute_non_query(
            """
            INSERT INTO incidents (id, camera_id, severity, status, start_utc, peak_weapon_score, peak_violence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("inc-02", cam_id, "MEDIUM", "false_alarm", t2.isoformat(), 0.6, 0.1),
        )
        db_mgr.execute_non_query(
            "INSERT INTO review_actions (id, incident_id, user_id, action, note, ts_utc) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "act-02",
                "inc-02",
                "op2",
                "mark_false_alarm",
                "False Alarm Reason: Maintenance ladder. Crew on site.",
                (t2 + timedelta(seconds=120)).isoformat(),
            ),
        )

        # Seed Incident 3: Active / New (not reviewed)
        t3 = clock.now_utc() + timedelta(minutes=10)
        db_mgr.execute_non_query(
            """
            INSERT INTO incidents (id, camera_id, severity, status, start_utc, peak_weapon_score, peak_violence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("inc-03", cam_id, "LOW", "new", t3.isoformat(), 0.4, 0.2),
        )

        # Verify KPIs
        kpis = analytics.get_kpis()
        assert kpis["total_incidents"] == 3
        assert kpis["reviewed_count"] == 1
        assert kpis["false_alarm_count"] == 1
        assert kpis["new_count"] == 1
        assert kpis["high_severity_count"] == 1
        # False alarm rate: 1 / (1 reviewed + 1 FA) = 50.0%
        assert kpis["false_alarm_rate_pct"] == 50.0

        # MTTR: average of 60s and 120s = 90.0s
        assert kpis["mttr_sec"] == 90.0
        assert kpis["mttr_median_sec"] == 90.0

        db_mgr.close()


def test_analytics_trends_and_pareto():
    """Verify daily trends, 24-hour diurnal distribution, and Pareto analysis."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_analytics_trends.db")
        db_mgr = DatabaseManager(db_path=db_path)
        MigrationRunner(db_mgr).run_migrations()

        clock = SimulatedClock()
        analytics = AnalyticsEngine(db_manager=db_mgr)

        cam_id = "cam-trends"
        db_mgr.execute_non_query(
            "INSERT INTO cameras (id, name, source_type, source_ref_masked, created_utc) VALUES (?, ?, ?, ?, ?)",
            (cam_id, "Perimeter Fence", "file", "test.mp4", clock.now_utc().isoformat()),
        )

        # Seed 2 incidents on 2026-10-18 at hour 14
        iso_14 = "2026-10-18T14:15:00+00:00"
        db_mgr.execute_non_query(
            "INSERT INTO incidents (id, camera_id, severity, status, start_utc) VALUES (?, ?, ?, ?, ?)",
            ("inc-t1", cam_id, "HIGH", "reviewed", iso_14),
        )
        db_mgr.execute_non_query(
            "INSERT INTO incidents (id, camera_id, severity, status, start_utc) VALUES (?, ?, ?, ?, ?)",
            ("inc-t2", cam_id, "MEDIUM", "false_alarm", iso_14),
        )

        # Seed false alarm review action
        db_mgr.execute_non_query(
            "INSERT INTO review_actions (id, incident_id, user_id, action, note, ts_utc) VALUES (?, ?, ?, ?, ?, ?)",
            ("act-fa1", "inc-t2", "op1", "mark_false_alarm", "False Alarm Reason: Optical reflection on window.", iso_14),
        )

        # 1. Trends
        trends = analytics.get_incident_trends()
        assert len(trends) == 1
        assert trends[0]["date_str"] == "2026-10-18"
        assert trends[0]["high_count"] == 1
        assert trends[0]["medium_count"] == 1
        assert trends[0]["total_count"] == 2

        # 2. Hourly distribution: 24 slots, hour 14 has count 2
        hourly = analytics.get_hourly_distribution()
        assert len(hourly) == 24
        assert hourly[14]["count"] == 2
        assert hourly[0]["count"] == 0

        # 3. Pareto analysis
        pareto = analytics.get_false_alarm_pareto()
        assert len(pareto) == 1
        assert pareto[0]["category"] == "Optical reflection on window"
        assert pareto[0]["count"] == 1
        assert pareto[0]["percentage"] == 100.0

        # 4. Camera threat summary
        cams = analytics.get_camera_threat_summary()
        assert len(cams) == 1
        assert cams[0]["camera_id"] == cam_id
        assert cams[0]["incident_count"] == 2

        db_mgr.close()
