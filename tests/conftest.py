"""pytest configuration for knowledge-db tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Generator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _track_connections() -> Generator[None, None, None]:
    """Track sqlite3 connections opened during each test; assert all closed after."""
    opened: list[sqlite3.Connection] = []
    original_connect: Callable[..., sqlite3.Connection] = sqlite3.connect

    def tracking_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = original_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    sqlite3.connect = tracking_connect  # type: ignore[assignment]  # monkey-patch: tracking_connect has broader signature than sqlite3.connect
    yield
    sqlite3.connect = original_connect  # type: ignore[assignment]  # restore original after monkey-patch
    for conn in opened:
        try:
            conn.execute("SELECT 1")
            pytest.fail(f"sqlite3.Connection at {hex(id(conn))} was never closed")
        except sqlite3.ProgrammingError:
            pass  # closed — connection raises when you try to use it
