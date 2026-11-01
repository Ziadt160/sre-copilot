"""One-version subset ingest of the real pinned checkout with the real gte-small model.

Skipped unless `data/sources/<newest>/` exists (run `sre-ingest fetch --versions <v>` first).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from sre_ingest import sources, store
from sre_ingest.pipeline import run_ingest
from sre_shared.config import load_corpus_config
from sre_shared.embed import get_embedder
from sre_shared.testing.plugin import DATA_TABLES

pytestmark = [pytest.mark.integration, pytest.mark.models]


@pytest.fixture
def dsn(pg_dsn: str) -> Iterator[str]:
    with psycopg.connect(pg_dsn, autocommit=True) as c:
        c.execute("truncate " + ", ".join(f"public.{t}" for t in DATA_TABLES) + " cascade")
    yield pg_dsn


def test_subset_ingest_is_idempotent_and_within_budget(dsn: str) -> None:
    cfg = load_corpus_config()
    if not cfg.versions:
        pytest.skip("config/corpus.yaml is not pinned")
    newest = cfg.versions[0]
    if not (sources.source_dir(newest) / "content").is_dir():
        pytest.skip(f"no checkout for {newest.k8s_version}")
    kwargs = {"versions": [newest.k8s_version], "limit_docs": 150, "do_fetch": False}
    first = run_ingest(cfg, dsn=dsn, models=("gte-small",), log=lambda _m: None, **kwargs)  # type: ignore[arg-type]
    assert first.chunks_inserted > 150
    second = run_ingest(cfg, dsn=dsn, models=("gte-small",), log=lambda _m: None, **kwargs)  # type: ignore[arg-type]
    assert second.corpus_version_id == first.corpus_version_id
    assert second.chunks_inserted == 0
    assert second.documents_inserted == 0

    emb = get_embedder("gte-small")
    with store.open_conn(dsn) as conn:
        rows = conn.execute(
            "select c.header, c.body, c.char_start, c.char_end, d.parsed_text, c.emb_gte_small"
            " is null as missing from public.chunks c join public.documents d on d.id = c.document_id"
        ).fetchall()
    assert len(rows) == first.chunks_total
    worst = 0
    for r in rows:
        assert r["body"] == r["parsed_text"][r["char_start"] : r["char_end"]]
        assert not r["missing"]
        worst = max(worst, emb.count_tokens(f"{r['header']}\n\n{r['body']}"))
    assert worst <= 510
    assert Path(sources.source_dir(newest)).is_dir()
