"""Local storage helpers: SQLite (primary store) + Parquet (fast analytics).

Deduplication strategy: rows are filtered against existing keys before
insert. Social posts dedupe on (source, post_hash); news on (source, url);
prices on (ticker, date).
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from config.settings import DB_PATH
from src.utils.logger import get_logger

log = get_logger(__name__)


@contextmanager
def sqlite_connect(db_path: Path | str | None = None):
    conn = sqlite3.connect(str(db_path or DB_PATH))
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def post_hash(text: str, url: str | None = None) -> str:
    """Stable identity for a post: URL when available, else normalized text."""
    if url:
        return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:32]
    normalized = " ".join((text or "").split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def _existing_keys(conn: sqlite3.Connection, table: str, keys: list[str]) -> set[tuple]:
    cur = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    if cur.fetchone() is None:
        return set()
    cols = ", ".join(keys)
    try:
        rows = conn.execute(f"SELECT {cols} FROM {table}").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {tuple("" if v is None else str(v) for v in r) for r in rows}


def save_to_sqlite(
    df: pd.DataFrame,
    table: str,
    dedup_keys: list[str] | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Append df to table, skipping rows whose dedup_keys already exist.

    Returns the number of newly inserted rows.
    """
    if df.empty:
        return 0
    with sqlite_connect(db_path) as conn:
        out = df
        if dedup_keys:
            missing = [k for k in dedup_keys if k not in df.columns]
            if missing:
                raise ValueError(f"dedup keys not in dataframe: {missing}")
            existing = _existing_keys(conn, table, dedup_keys)
            mask = [
                tuple("" if pd.isna(r[k]) else str(r[k]) for k in dedup_keys) not in existing
                for _, r in df.iterrows()
            ]
            out = df[mask]
            out = out.drop_duplicates(subset=dedup_keys, keep="first")
        inserted = len(out)
        if inserted:
            out.to_sql(table, conn, if_exists="append", index=False)
        log.info("sqlite: %d new rows -> %s (%d duplicates skipped)", inserted, table, len(df) - len(out))
        return inserted


def load_from_sqlite(
    table: str,
    where: str | None = None,
    params: tuple = (),
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    query = f"SELECT * FROM {table}"
    if where:
        query += f" WHERE {where}"
    with sqlite_connect(db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        if cur.fetchone() is None:
            return pd.DataFrame()
        return pd.read_sql_query(query, conn, params=params)


def replace_table(df: pd.DataFrame, table: str, db_path: Path | str | None = None) -> int:
    """Drop and rewrite a table atomically-ish. Used for derived data
    (e.g. sentiment annotations) that is fully recomputed each run."""
    with sqlite_connect(db_path) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        if df.empty:
            return 0
        df.to_sql(table, conn, if_exists="append", index=False)
        return len(df)


def table_exists(table: str, db_path: Path | str | None = None) -> bool:
    with sqlite_connect(db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        return cur.fetchone() is not None


def save_parquet(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path | str) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path)
