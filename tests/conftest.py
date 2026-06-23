"""pytest configuration for knowledge-db tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _track_connections() -> Generator[None, None, None]:
    """Track sqlite3 connections opened during each test; assert all closed after."""
    opened: list[sqlite3.Connection] = []
    original_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        conn = original_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    sqlite3.connect = tracking_connect  # type: ignore[assignment]
    yield
    sqlite3.connect = original_connect  # type: ignore[assignment]
    for conn in opened:
        try:
            conn.execute("SELECT 1")
            pytest.fail(f"sqlite3.Connection at {hex(id(conn))} was never closed")
        except sqlite3.ProgrammingError:
            pass  # closed — connection raises when you try to use it
