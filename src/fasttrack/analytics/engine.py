"""Operational intelligence, triage analytics, MTTR, and trend aggregation engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.fasttrack.core.logging import get_logger
from src.fasttrack.storage.db import DatabaseManager

logger = get_logger("analytics.engine")


class AnalyticsEngine:
    """Computes security operations metrics, MTTR, false alarm rates, and incident trends."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def get_kpis(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute top-level operational KPIs across all camera streams."""
        query = """
            SELECT
                COUNT(*) as total_incidents,
                SUM(CASE WHEN status = 'new' THEN 1 ELSE 0 END) as new_count,
                SUM(CASE WHEN status = 'reviewed' THEN 1 ELSE 0 END) as reviewed_count,
                SUM(CASE WHEN status = 'false_alarm' THEN 1 ELSE 0 END) as false_alarm_count,
                SUM(CASE WHEN status = 'escalated' THEN 1 ELSE 0 END) as escalated_count,
                SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END) as high_severity_count
            FROM incidents
            WHERE 1=1
        """
        params: List[Any] = []
        if start_date:
            query += " AND start_utc >= ?"
            params.append(start_date)
        if end_date:
            query += " AND start_utc <= ?"
            params.append(end_date)

        rows = self.db.execute_query(query, tuple(params))
        row = rows[0] if rows else None

        total = row["total_incidents"] if row and row["total_incidents"] else 0
        new_cnt = row["new_count"] if row and row["new_count"] else 0
        rev_cnt = row["reviewed_count"] if row and row["reviewed_count"] else 0
        fa_cnt = row["false_alarm_count"] if row and row["false_alarm_count"] else 0
        esc_cnt = row["escalated_count"] if row and row["escalated_count"] else 0
        high_cnt = row["high_severity_count"] if row and row["high_severity_count"] else 0

        # Calculate False Alarm Rate (FAR): FA / (Reviewed + FA + Escalated)
        resolved_total = rev_cnt + fa_cnt + esc_cnt
        far_pct = round((float(fa_cnt) / float(resolved_total)) * 100.0, 1) if resolved_total > 0 else 0.0

        # Mean Time to Review (MTTR) calculation
        mttr_info = self.get_mttr_distribution(start_date=start_date, end_date=end_date)

        # Active cameras count
        cam_rows = self.db.execute_query("SELECT COUNT(*) as cnt FROM cameras WHERE enabled = 1")
        active_cams = cam_rows[0]["cnt"] if cam_rows else 0

        return {
            "total_incidents": total,
            "new_count": new_cnt,
            "reviewed_count": rev_cnt,
            "false_alarm_count": fa_cnt,
            "escalated_count": esc_cnt,
            "high_severity_count": high_cnt,
            "false_alarm_rate_pct": far_pct,
            "mttr_sec": mttr_info["mean_sec"],
            "mttr_median_sec": mttr_info["median_sec"],
            "active_cameras_count": active_cams,
        }

    def get_mttr_distribution(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute Mean Time to Review (seconds between incident start and first operator review)."""
        query = """
            SELECT
                i.id,
                i.start_utc,
                MIN(r.ts_utc) as first_action_utc
            FROM incidents i
            JOIN review_actions r ON i.id = r.incident_id
            WHERE 1=1
        """
        params: List[Any] = []
        if start_date:
            query += " AND i.start_utc >= ?"
            params.append(start_date)
        if end_date:
            query += " AND i.start_utc <= ?"
            params.append(end_date)

        query += " GROUP BY i.id"
        rows = self.db.execute_query(query, tuple(params))

        durations_sec: List[float] = []
        for r in rows:
            try:
                t_start = datetime.fromisoformat(r["start_utc"])
                t_action = datetime.fromisoformat(r["first_action_utc"])
                diff = (t_action - t_start).total_seconds()
                if diff >= 0.0:
                    durations_sec.append(diff)
            except Exception:
                continue

        if not durations_sec:
            return {"mean_sec": 0.0, "median_sec": 0.0, "p95_sec": 0.0, "sample_count": 0, "durations_sec": []}

        mean_val = round(float(np.mean(durations_sec)), 1)
        median_val = round(float(np.median(durations_sec)), 1)
        p95_val = round(float(np.percentile(durations_sec, 95)), 1)

        return {
            "mean_sec": mean_val,
            "median_sec": median_val,
            "p95_sec": p95_val,
            "sample_count": len(durations_sec),
            "durations_sec": durations_sec,
        }

    def get_incident_trends(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate daily incident frequency segmented by triage severity."""
        query = """
            SELECT
                substr(start_utc, 1, 10) as date_str,
                SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END) as high_count,
                SUM(CASE WHEN severity = 'MEDIUM' THEN 1 ELSE 0 END) as medium_count,
                SUM(CASE WHEN severity = 'LOW' THEN 1 ELSE 0 END) as low_count,
                SUM(CASE WHEN severity = 'INFO' THEN 1 ELSE 0 END) as info_count,
                COUNT(*) as total_count
            FROM incidents
            WHERE 1=1
        """
        params: List[Any] = []
        if start_date:
            query += " AND start_utc >= ?"
            params.append(start_date)
        if end_date:
            query += " AND start_utc <= ?"
            params.append(end_date)

        query += " GROUP BY substr(start_utc, 1, 10) ORDER BY date_str ASC"
        rows = self.db.execute_query(query, tuple(params))
        return [dict(r) for r in rows]

    def get_hourly_distribution(self) -> List[Dict[str, Any]]:
        """Compute 24-hour diurnal distribution of incident detection events."""
        # Query hours 00 through 23
        query = """
            SELECT
                CAST(substr(start_utc, 12, 2) AS INTEGER) as hour_int,
                COUNT(*) as count
            FROM incidents
            WHERE length(start_utc) >= 13
            GROUP BY hour_int
            ORDER BY hour_int ASC
        """
        rows = self.db.execute_query(query)
        hour_counts = {int(r["hour_int"]): int(r["count"]) for r in rows if r["hour_int"] is not None}

        # Populate all 24 hours
        result = []
        for h in range(24):
            result.append({
                "hour": h,
                "hour_label": f"{h:02d}:00",
                "count": hour_counts.get(h, 0),
            })
        return result

    def get_false_alarm_pareto(self) -> List[Dict[str, Any]]:
        """Categorize false alarm rationales and compute cumulative frequency for Pareto analysis."""
        rows = self.db.execute_query(
            "SELECT note FROM review_actions WHERE action = 'mark_false_alarm'"
        )
        if not rows:
            return []

        counts: Dict[str, int] = {}
        for r in rows:
            note = r["note"] or ""
            # Extract category name from note pattern 'False Alarm Reason: ...'
            if "False Alarm Reason:" in note:
                cat = note.split("False Alarm Reason:")[1].split(".")[0].strip()
            else:
                cat = note[:40].strip() or "Unspecified"
            counts[cat] = counts.get(cat, 0) + 1

        # Sort descending
        sorted_cats = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        total_fa = sum(counts.values())

        pareto = []
        cumulative = 0
        for cat, cnt in sorted_cats:
            cumulative += cnt
            cum_pct = round((float(cumulative) / float(total_fa)) * 100.0, 1)
            pareto.append({
                "category": cat,
                "count": cnt,
                "percentage": round((float(cnt) / float(total_fa)) * 100.0, 1),
                "cumulative_percentage": cum_pct,
            })
        return pareto

    def get_camera_threat_summary(self) -> List[Dict[str, Any]]:
        """Aggregate total threat detections and peak model scores per camera location."""
        query = """
            SELECT
                c.id as camera_id,
                c.name as camera_name,
                COUNT(i.id) as incident_count,
                SUM(CASE WHEN i.severity = 'HIGH' THEN 1 ELSE 0 END) as high_severity_count,
                MAX(i.peak_weapon_score) as max_weapon_score,
                MAX(i.peak_violence_score) as max_violence_score
            FROM cameras c
            LEFT JOIN incidents i ON c.id = i.camera_id
            GROUP BY c.id
            ORDER BY incident_count DESC
        """
        rows = self.db.execute_query(query)
        result = []
        for r in rows:
            d = dict(r)
            d["incident_count"] = d["incident_count"] or 0
            d["high_severity_count"] = d["high_severity_count"] or 0
            d["max_weapon_score"] = round(d["max_weapon_score"] or 0.0, 2)
            d["max_violence_score"] = round(d["max_violence_score"] or 0.0, 2)
            result.append(d)
        return result

    def get_telemetry_health_summary(self) -> List[Dict[str, Any]]:
        """Summarize average pipeline FPS, latency, and dropped frames from health_samples table."""
        query = """
            SELECT
                camera_id,
                AVG(fps) as avg_fps,
                AVG(latency_ms) as avg_latency_ms,
                MAX(dropped_frames) as max_dropped_frames,
                AVG(cpu_percent) as avg_cpu_pct,
                AVG(ram_percent) as avg_ram_pct,
                COUNT(*) as sample_count
            FROM health_samples
            GROUP BY camera_id
        """
        rows = self.db.execute_query(query)
        result = []
        for r in rows:
            result.append({
                "camera_id": r["camera_id"],
                "avg_fps": round(r["avg_fps"] or 0.0, 1),
                "avg_latency_ms": round(r["avg_latency_ms"] or 0.0, 1),
                "max_dropped_frames": int(r["max_dropped_frames"] or 0),
                "avg_cpu_pct": round(r["avg_cpu_pct"] or 0.0, 1),
                "avg_ram_pct": round(r["avg_ram_pct"] or 0.0, 1),
                "sample_count": int(r["sample_count"] or 0),
            })
        return result
