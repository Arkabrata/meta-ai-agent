"""Embedded SQLite store — file-based, no separate database server."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from meta_analyst.config import ROOT
from meta_analyst.sample_data import generate_clean_sample, generate_raw_meta_rows

DB_PATH = ROOT / "data" / "meta_analyst.sqlite"
SAMPLE_DATASET_ID = "sample_150_daily"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                row_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS dataset_rows (
                dataset_id TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (dataset_id, row_index),
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()
    ensure_sample_dataset()


def ensure_sample_dataset(n: int = 150) -> str:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id FROM datasets WHERE id = ?", (SAMPLE_DATASET_ID,)
        ).fetchone()
        if row:
            return SAMPLE_DATASET_ID

    raw = generate_raw_meta_rows(n)
    save_dataframe(
        SAMPLE_DATASET_ID,
        name="Demo Meta Ads (150 rows)",
        source="sample",
        df=raw,
        meta={"rows": n, "generic": True},
    )
    return SAMPLE_DATASET_ID


def save_dataframe(
    dataset_id: str,
    *,
    name: str,
    source: str,
    df: pd.DataFrame,
    meta: dict[str, Any] | None = None,
) -> str:
    created = datetime.now(timezone.utc).isoformat()
    records = json.loads(df.to_json(orient="records", date_format="iso"))
    with _connect() as conn:
        conn.execute("DELETE FROM dataset_rows WHERE dataset_id = ?", (dataset_id,))
        conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        conn.execute(
            """
            INSERT INTO datasets (id, name, source, row_count, created_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                name,
                source,
                len(records),
                created,
                json.dumps(meta or {}),
            ),
        )
        conn.executemany(
            """
            INSERT INTO dataset_rows (dataset_id, row_index, payload_json)
            VALUES (?, ?, ?)
            """,
            [(dataset_id, i, json.dumps(rec)) for i, rec in enumerate(records)],
        )
        conn.commit()
    return dataset_id


def list_datasets() -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, name, source, row_count, created_at, meta_json
            FROM datasets
            ORDER BY created_at DESC
            """
        ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "name": r["name"],
                "source": r["source"],
                "row_count": r["row_count"],
                "created_at": r["created_at"],
                "meta": json.loads(r["meta_json"] or "{}"),
            }
        )
    return out


def load_dataframe(dataset_id: str) -> pd.DataFrame:
    init_db()
    with _connect() as conn:
        exists = conn.execute(
            "SELECT id FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
        if not exists:
            raise KeyError(f"Dataset not found: {dataset_id}")
        rows = conn.execute(
            """
            SELECT payload_json FROM dataset_rows
            WHERE dataset_id = ?
            ORDER BY row_index
            """,
            (dataset_id,),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    records = [json.loads(r["payload_json"]) for r in rows]
    return pd.DataFrame(records)


def load_clean_dataset(dataset_id: str):
    from meta_analyst.data import load_and_clean_data

    raw = load_dataframe(dataset_id)
    if dataset_id == SAMPLE_DATASET_ID and raw.empty:
        return generate_clean_sample(150)
    return load_and_clean_data(raw, active_only=False)


def db_info() -> dict[str, Any]:
    init_db()
    return {
        "engine": "sqlite",
        "path": str(DB_PATH),
        "note": "Embedded file store — no database server required",
        "datasets": list_datasets(),
    }
