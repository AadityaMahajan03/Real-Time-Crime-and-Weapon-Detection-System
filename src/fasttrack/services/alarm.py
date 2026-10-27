"""Dedicated background Acoustic Alarm Service with dual-threshold hysteresis and audit logging."""

from __future__ import annotations

import enum
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.fasttrack.core.logging import get_logger
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.repositories import AlarmRepository, AuditRepository

logger = get_logger("services.alarm")

# Safe platform sound import
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AlarmState(enum.Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    COOLDOWN = "COOLDOWN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    ESCALATED = "ESCALATED"


class AlarmPattern(enum.Enum):
    CONTINUOUS_WEAPON = "continuous_weapon"  # High priority 880Hz siren
    PULSED_VIOLENCE = "pulsed_violence"      # Staccato 520Hz beep
    ESCALATION_WARBLE = "escalation_warble"  # Rapid alternating dual-pitch


class AlarmService:
    """Non-blocking background alarm state machine decoupled from video inference."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        high_threshold: float = 0.55,
        low_threshold: float = 0.35,
        cooldown_sec: float = 10.0,
        escalation_timeout_sec: float = 60.0,
        server_audio_enabled: bool = True,
    ) -> None:
        self.db = db_manager
        self.alarm_repo = AlarmRepository(db_manager)
        self.audit_repo = AuditRepository(db_manager)

        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.cooldown_sec = cooldown_sec
        self.escalation_timeout_sec = escalation_timeout_sec
        self.server_audio_enabled = server_audio_enabled

        self.state = AlarmState.IDLE
        self.pattern = AlarmPattern.CONTINUOUS_WEAPON
        self.current_alarm_id: Optional[str] = None
        self.current_incident_id: Optional[str] = None
        self.current_camera_id: Optional[str] = None

        self._last_detection_mono: float = 0.0
        self._alarm_start_mono: float = 0.0
        self._alarm_start_utc: Optional[str] = None
        self._muted_until_mono: float = 0.0

        self._lock = threading.Lock()
        self._is_running = True

        # Background state-machine tick thread
        self._tick_thread = threading.Thread(target=self._state_machine_loop, name="AlarmStateMachine", daemon=True)
        self._tick_thread.start()

        # Dedicated audio synthesis playback thread
        self._sound_thread = threading.Thread(target=self._audio_playback_loop, name="AlarmAudioPlayback", daemon=True)
        self._sound_thread.start()

    def update_threat(
        self,
        camera_id: str,
        weapon_score: float,
        violence_score: float,
        incident_id: Optional[str] = None,
    ) -> None:
        """Feed latest threat scores into the hysteresis state machine. Thread-safe & non-blocking."""
        with self._lock:
            now_mono = time.monotonic()
            is_muted = now_mono < self._muted_until_mono
            if is_muted:
                return

            max_score = max(weapon_score, violence_score)
            is_weapon = weapon_score >= violence_score

            # Trigger condition: high threshold exceeded
            if max_score >= self.high_threshold:
                self._last_detection_mono = now_mono

                if self.state in (AlarmState.IDLE, AlarmState.COOLDOWN):
                    self._start_alarm(
                        camera_id=camera_id,
                        incident_id=incident_id,
                        pattern=AlarmPattern.CONTINUOUS_WEAPON if is_weapon else AlarmPattern.PULSED_VIOLENCE,
                        now_mono=now_mono,
                    )
                elif self.state == AlarmState.ACTIVE:
                    # Update pattern if weapon supercedes violence
                    if is_weapon and self.pattern == AlarmPattern.PULSED_VIOLENCE:
                        self.pattern = AlarmPattern.CONTINUOUS_WEAPON

            # Hysteresis sustain: above lower threshold maintains active state
            elif max_score >= self.low_threshold:
                if self.state == AlarmState.ACTIVE:
                    self._last_detection_mono = now_mono

    def _start_alarm(
        self,
        camera_id: str,
        incident_id: Optional[str],
        pattern: AlarmPattern,
        now_mono: float,
    ) -> None:
        self.state = AlarmState.ACTIVE
        self.pattern = pattern
        self.current_camera_id = camera_id
        self.current_incident_id = incident_id
        self.current_alarm_id = f"alm-{uuid.uuid4().hex[:10]}"
        self._alarm_start_mono = now_mono
        self._alarm_start_utc = datetime.now(timezone.utc).isoformat()

        # Register in alarms table
        try:
            self.alarm_repo.create_alarm(
                alarm_id=self.current_alarm_id,
                incident_id=incident_id,
                camera_id=camera_id,
                alarm_type=pattern.value,
                pattern="continuous" if pattern == AlarmPattern.CONTINUOUS_WEAPON else "pulsed",
            )
        except Exception as e:
            logger.error("Failed to log alarm start to DB: %s", e)

        logger.warning("SIREN ACTIVATED: %s on camera %s (ID: %s)", pattern.value, camera_id, self.current_alarm_id)

    def acknowledge(self, operator_id: str = "Operator_Lead", notes: str = "") -> None:
        """Operator explicitly acknowledges and silences the active alarm."""
        with self._lock:
            if self.state in (AlarmState.ACTIVE, AlarmState.ESCALATED, AlarmState.COOLDOWN):
                self.state = AlarmState.ACKNOWLEDGED
                alarm_id = self.current_alarm_id
                dur = time.monotonic() - self._alarm_start_mono
                if alarm_id:
                    self.alarm_repo.acknowledge_alarm(alarm_id, operator_id, notes)
                    self.alarm_repo.stop_alarm(alarm_id, dur)
                    self.audit_repo.log_action(
                        actor=operator_id,
                        action="ALARM_ACKNOWLEDGED",
                        entity_type="alarm",
                        entity_id=alarm_id,
                        details=f"Alarm silenced by operator after {dur:.1f}s: {notes}",
                    )
                logger.info("Alarm %s acknowledged by %s", alarm_id, operator_id)

    def mute(self, duration_sec: float = 300.0) -> None:
        """Temporarily mute sirens for testing or maintenance."""
        with self._lock:
            self._muted_until_mono = time.monotonic() + duration_sec
            if self.state in (AlarmState.ACTIVE, AlarmState.ESCALATED):
                self.acknowledge(operator_id="System", notes=f"Muted for {duration_sec}s")

    def test_siren(self, duration_sec: float = 3.0, pattern: str = "weapon") -> None:
        """Emit a brief test alarm sound."""
        pat = AlarmPattern.CONTINUOUS_WEAPON if pattern == "weapon" else AlarmPattern.PULSED_VIOLENCE
        with self._lock:
            now_mono = time.monotonic()
            self._start_alarm("test_source", None, pat, now_mono)
            self._last_detection_mono = now_mono

        def _auto_stop():
            time.sleep(duration_sec)
            self.acknowledge("System", "Manual siren test completed.")

        threading.Thread(target=_auto_stop, daemon=True).start()

    def _state_machine_loop(self) -> None:
        """Evaluates cooldown and escalation timeouts every 100ms."""
        while self._is_running:
            time.sleep(0.1)
            with self._lock:
                now_mono = time.monotonic()

                if self.state == AlarmState.ACTIVE:
                    elapsed_threat = now_mono - self._last_detection_mono
                    elapsed_alarm = now_mono - self._alarm_start_mono

                    # Escalation check: unacknowledged critical threat exceeding timeout
                    if elapsed_alarm >= self.escalation_timeout_sec and self.state != AlarmState.ESCALATED:
                        self.state = AlarmState.ESCALATED
                        self.pattern = AlarmPattern.ESCALATION_WARBLE
                        logger.critical("ALARM ESCALATED: No operator acknowledgment after %ds", self.escalation_timeout_sec)

                    # Cooldown check: threat cleared for cooldown_sec
                    elif elapsed_threat >= self.cooldown_sec:
                        self.state = AlarmState.IDLE
                        dur = now_mono - self._alarm_start_mono
                        if self.current_alarm_id:
                            self.alarm_repo.stop_alarm(self.current_alarm_id, dur)
                        logger.info("Alarm %s stopped after %.1fs (threat cleared)", self.current_alarm_id, dur)

                elif self.state == AlarmState.ACKNOWLEDGED:
                    # Return to IDLE once threat is clear
                    if (now_mono - self._last_detection_mono) >= self.cooldown_sec:
                        self.state = AlarmState.IDLE

    def _audio_playback_loop(self) -> None:
        """Dedicated acoustic sound generator loop."""
        while self._is_running:
            should_play = False
            cur_pat = AlarmPattern.CONTINUOUS_WEAPON

            with self._lock:
                if self.state in (AlarmState.ACTIVE, AlarmState.ESCALATED) and self.server_audio_enabled:
                    should_play = True
                    cur_pat = self.pattern

            if should_play and HAS_WINSOUND:
                try:
                    if cur_pat == AlarmPattern.CONTINUOUS_WEAPON:
                        # Dual-tone sweeping siren: 880Hz down to 440Hz
                        winsound.Beep(880, 250)
                        winsound.Beep(587, 250)
                    elif cur_pat == AlarmPattern.PULSED_VIOLENCE:
                        # Staccato warning beep
                        winsound.Beep(520, 150)
                        time.sleep(0.15)
                    elif cur_pat == AlarmPattern.ESCALATION_WARBLE:
                        # Urgent alternating warble
                        winsound.Beep(1046, 100)
                        winsound.Beep(784, 100)
                except Exception:
                    time.sleep(0.5)
            else:
                time.sleep(0.1)

    def get_status(self) -> Dict[str, Any]:
        """Return live telemetry status for UI & Web Audio synchronization."""
        with self._lock:
            now_mono = time.monotonic()
            is_active = self.state in (AlarmState.ACTIVE, AlarmState.ESCALATED)
            dur = max(0.0, now_mono - self._alarm_start_mono) if is_active else 0.0
            return {
                "state": self.state.value,
                "is_active": is_active,
                "pattern": self.pattern.value,
                "alarm_id": self.current_alarm_id,
                "camera_id": self.current_camera_id,
                "duration_sec": round(dur, 1),
                "is_muted": now_mono < self._muted_until_mono,
                "acknowledged": self.state == AlarmState.ACKNOWLEDGED,
            }

    def stop(self) -> None:
        self._is_running = False
