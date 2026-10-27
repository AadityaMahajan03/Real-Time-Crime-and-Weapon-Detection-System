"""Unit tests for Clock abstraction (SystemClock and SimulatedClock)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from src.fasttrack.core.clock import SimulatedClock, SystemClock


def test_system_clock_returns_utc_datetime() -> None:
    clock = SystemClock()
    now = clock.now_utc()
    assert isinstance(now, datetime)
    assert now.tzinfo is not None
    assert clock.monotonic() > 0


def test_simulated_clock_advancement(sim_clock: SimulatedClock) -> None:
    t0_utc = sim_clock.now_utc()
    m0 = sim_clock.monotonic()

    # Advance virtual time by 45.5 seconds
    sim_clock.advance(45.5)

    assert sim_clock.now_utc() == t0_utc + timedelta(seconds=45.5)
    assert sim_clock.monotonic() == m0 + 45.5


def test_simulated_clock_sleep_advances_without_real_delay(sim_clock: SimulatedClock) -> None:
    t_start_real = time.monotonic()

    # Sleep 3600 simulated seconds (1 virtual hour)
    sim_clock.sleep(3600.0)

    t_elapsed_real = time.monotonic() - t_start_real
    # Real execution time must be under 50 milliseconds
    assert t_elapsed_real < 0.05
    assert sim_clock.monotonic() == 4600.0


def test_simulated_clock_set_time(sim_clock: SimulatedClock) -> None:
    custom_dt = datetime(2026, 6, 1, 8, 30, 0, tzinfo=timezone.utc)
    sim_clock.set_time(custom_dt, 500.0)

    assert sim_clock.now_utc() == custom_dt
    assert sim_clock.monotonic() == 500.0
