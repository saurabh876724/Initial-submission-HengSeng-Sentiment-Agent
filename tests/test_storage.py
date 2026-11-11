import pandas as pd
import pytest

from src.utils.storage import (
    load_from_sqlite,
    load_parquet,
    post_hash,
    save_parquet,
    save_to_sqlite,
    table_exists,
)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "test.sqlite"


def make_df(texts):
    return pd.DataFrame(
        {
            "source": ["lihkg"] * len(texts),
            "post_hash": [post_hash(t) for t in texts],
            "text": texts,
        }
    )


def test_save_and_load(db):
    df = make_df(["hello world", "second post"])
    assert save_to_sqlite(df, "posts", dedup_keys=["source", "post_hash"], db_path=db) == 2
    out = load_from_sqlite("posts", db_path=db)
    assert len(out) == 2
    assert set(out["text"]) == {"hello world", "second post"}


def test_dedup_on_second_save(db):
    df = make_df(["same post"])
    save_to_sqlite(df, "posts", dedup_keys=["source", "post_hash"], db_path=db)
    inserted = save_to_sqlite(df, "posts", dedup_keys=["source", "post_hash"], db_path=db)
    assert inserted == 0
    assert len(load_from_sqlite("posts", db_path=db)) == 1


def test_dedup_within_batch(db):
    df = make_df(["dupe", "dupe", "unique"])
    assert save_to_sqlite(df, "posts", dedup_keys=["source", "post_hash"], db_path=db) == 2


def test_missing_table_returns_empty(db):
    assert load_from_sqlite("nonexistent", db_path=db).empty
    assert not table_exists("nonexistent", db_path=db)


def test_where_filter(db):
    save_to_sqlite(make_df(["a", "b"]), "posts", dedup_keys=["source", "post_hash"], db_path=db)
    out = load_from_sqlite("posts", where="text = ?", params=("a",), db_path=db)
    assert len(out) == 1


def test_post_hash_url_priority():
    assert post_hash("any text", "https://x.com/1") == post_hash("other text", "https://x.com/1")
    assert post_hash("some text") != post_hash("some  other text")
    assert post_hash("same", None) == post_hash("SAME", None)


def test_parquet_roundtrip(tmp_path):
    df = make_df(["x"])
    p = tmp_path / "sub" / "out.parquet"
    save_parquet(df, p)
    out = load_parquet(p)
    assert out is not None and len(out) == 1
    assert load_parquet(tmp_path / "missing.parquet") is None


def test_dedup_keys_missing_column_raises(db):
    with pytest.raises(ValueError):
        save_to_sqlite(make_df(["x"]), "posts", dedup_keys=["nope"], db_path=db)
