"""
Service expiry and daily rate limiting for ToxNav cloud deployment.

Expiry:     Hard-coded date after which the app shows a "service ended" page.
Rate limit: Max real-mode pipeline runs per day, stored in /tmp so it resets
            on container cold-start (acceptable — cold starts are rare with
            scale-to-zero and a single replica).
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

# 60 days from 2026-06-30 launch
EXPIRY_DATE = date(2026, 8, 29)

# Real-mode runs per day — keeps monthly GPT-4o spend under ~$15
DAILY_REAL_MODE_LIMIT = 10

_COUNTER_FILE = Path("/tmp/toxnav_rate.json")


# ── Expiry ────────────────────────────────────────────────────────────────────

def is_expired() -> bool:
    return date.today() > EXPIRY_DATE


def days_remaining() -> int:
    return max(0, (EXPIRY_DATE - date.today()).days)


# ── Daily counter ─────────────────────────────────────────────────────────────

def _load_counter() -> dict:
    try:
        data = json.loads(_COUNTER_FILE.read_text())
        if data.get("date") == str(date.today()):
            return data
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return {"date": str(date.today()), "count": 0}


def _save_counter(data: dict) -> None:
    try:
        _COUNTER_FILE.write_text(json.dumps(data))
    except OSError:
        pass


def real_mode_runs_today() -> int:
    return _load_counter()["count"]


def real_mode_remaining_today() -> int:
    return max(0, DAILY_REAL_MODE_LIMIT - real_mode_runs_today())


def check_and_increment() -> tuple[bool, int]:
    """Return (allowed, remaining_after).

    Atomically checks the limit and increments if allowed.
    """
    data = _load_counter()
    if data["count"] >= DAILY_REAL_MODE_LIMIT:
        return False, 0
    data["count"] += 1
    _save_counter(data)
    remaining = DAILY_REAL_MODE_LIMIT - data["count"]
    return True, remaining
