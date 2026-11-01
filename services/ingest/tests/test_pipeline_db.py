from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import psycopg
import pytest

from sre_ingest import snapshot, store
from sre_ingest.chunker import CHUNKER_VERSION
from sre_ingest.chunking_config import load_chunking
from sre_ingest.pipeline import RunResult, run_ingest
from sre_ingest.report import build_report
from sre_ingest.testsite import make_site
from sre_shared.config import CorpusConfig, CorpusSource
from sre_shared.embed.base import Vectors
from sre_shared.embed.fake import FakeEmbedder
from sre_shared.testing.plugin import DATA_TABLES

pytestmark = pytest.mark.integration

SRC = CorpusSource(
    k8s_version="1.36",
    branch="release-1.36",
    commit_sha="1" * 40,
    url_base="https://v1-36.docs.kubernetes.io",
)


@pytest.fixture
def dsn(pg_dsn: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(pg_dsn, autocommit=True) as c:
            c.execute(
                "truncate "
                + ", ".join(f"public.{t}" for t in DATA_TABLES)
                + " restart identity cascade"
            )

    wipe()
    yield pg_dsn
    wipe()


@pytest.fixture
def cfg(tmp_path: Path) -> CorpusConfig:
    make_site(tmp_path / "sources" / "1.36")
    return CorpusConfig(
        repo="https://example.test/website.git",
        license="CC-BY-4.0",
        attribution="test",
        versions=[SRC],
    )


def ingest(cfg: CorpusConfig, dsn: str, tmp_path: Path, **kw: object) -> RunResult:
    fake = FakeEmbedder("gte-small")
    return run_ingest(
        cfg,
        dsn=dsn,
        do_fetch=False,
        data_dir=tmp_path,
        embedders={"gte-small": fake},
        token_counter=fake,
        log=lambda _m: None,
        **kw,  # type: ignore[arg-type]
    )


def test_idempotent_rerun_adds_nothing(dsn: str, cfg: CorpusConfig, tmp_path: Path) -> None:
    first = ingest(cfg, dsn, tmp_path)
    assert first.created_version
    assert first.documents_seen == 4  # draft and empty glossary page skipped
    assert first.chunks_inserted > 0
    second = ingest(cfg, dsn, tmp_path)
    assert second.corpus_version_id == first.corpus_version_id
    assert not second.created_version
    assert second.chunks_inserted == 0
    assert second.documents_inserted == 0
    assert second.chunks_total == first.chunks_total
    with store.open_conn(dsn) as conn:
        row = conn.execute(
            "select count(*) filter (where emb_gte_small is null) as nulls, count(*) as n"
            " from public.chunks"
        ).fetchone()
        assert row is not None and row["nulls"] == 0 and row["n"] == first.chunks_total
        rep = build_report(conn, first.corpus_version_id)
    assert rep["unknown_shortcodes"] == {"widget": 1, "gizmo": 1}
    assert rep["status"] == "ready"
    assert rep["embedding_models"] == ["gte-small"]
    # The re-run embedded nothing: the build's embedding stats stay, last_run is the re-run.
    assert rep["embedding"]["gte-small"]["embedded"] == first.chunks_total
    assert rep["last_run"]["chunks_inserted"] == 0
    assert rep["tokens"]["max"] > 0


def test_rerun_against_the_active_version_reuses_it(
    dsn: str, cfg: CorpusConfig, tmp_path: Path
) -> None:
    """Found in the lab: an identical re-run after --activate built a second full copy."""
    first = ingest(cfg, dsn, tmp_path, activate=True)
    second = ingest(cfg, dsn, tmp_path, activate=True)
    assert second.corpus_version_id == first.corpus_version_id
    assert not second.created_version
    assert second.chunks_inserted == 0 and second.documents_inserted == 0
    with store.open_conn(dsn) as conn:
        rows = conn.execute("select id, status from public.corpus_versions").fetchall()
    assert [(r["id"], r["status"]) for r in rows] == [(first.corpus_version_id, "active")]


def test_embedding_cache_is_used_across_versions(
    dsn: str, cfg: CorpusConfig, tmp_path: Path
) -> None:
    ingest(cfg, dsn, tmp_path)
    cfg.versions = [SRC.model_copy(update={"commit_sha": "2" * 40})]
    second = ingest(cfg, dsn, tmp_path)
    assert second.created_version
    assert second.stats["embedding"]["gte-small"]["cache_hits"] == second.chunks_total


def test_version_reuse_requires_label_sources_and_chunker(dsn: str) -> None:
    with store.open_conn(dsn) as conn:
        a, created = store.get_or_create_version(conn, "L", [SRC], "1")
        assert created
        assert store.get_or_create_version(conn, "L", [SRC], "1") == (a, False)
        other_sha = SRC.model_copy(update={"commit_sha": "3" * 40})
        assert store.get_or_create_version(conn, "L", [other_sha], "1")[0] != a
        assert store.get_or_create_version(conn, "L", [SRC], "2")[0] != a
        assert store.get_or_create_version(conn, "L2", [SRC], "1")[0] != a


def test_changed_document_inside_a_version_fails_loudly(
    dsn: str, cfg: CorpusConfig, tmp_path: Path
) -> None:
    first = ingest(cfg, dsn, tmp_path)
    page = tmp_path / "sources" / "1.36" / "content/en/docs/reference/glossary-index.md"
    page.write_text("---\ntitle: Glossary index\n---\nDifferent text now.\n", encoding="utf-8")
    with pytest.raises(store.ContentChanged):
        ingest(cfg, dsn, tmp_path)
    with store.open_conn(dsn) as conn:
        assert (
            store.get_or_create_version(conn, "x", [SRC], CHUNKER_VERSION)[0]
            > first.corpus_version_id
        )


def test_activate_prune_state_machine(dsn: str, cfg: CorpusConfig, tmp_path: Path) -> None:
    a = ingest(cfg, dsn, tmp_path, activate=True).corpus_version_id
    cfg.versions = [SRC.model_copy(update={"commit_sha": "4" * 40})]
    b = ingest(cfg, dsn, tmp_path).corpus_version_id
    assert a != b

    def statuses(conn: store.Conn) -> dict[int, str]:
        rows = conn.execute("select id, status from public.corpus_versions").fetchall()
        return {int(r["id"]): r["status"] for r in rows}

    with store.open_conn(dsn) as conn:
        assert statuses(conn) == {a: "active", b: "ready"}
        store.activate(conn, a)  # already active: a no-op
        assert statuses(conn) == {a: "active", b: "ready"}
        store.activate(conn, b)
        assert statuses(conn) == {a: "retired", b: "active"}
        with pytest.raises(ValueError, match="only 'ready'"):
            store.activate(conn, a)
        conn.rollback()
        assert store.resolve_version(conn, "active") == b
        assert store.prune(conn) == 1
        assert statuses(conn) == {b: "active"}
        assert conn.execute(
            "select count(*) as n from public.chunks where corpus_version_id=%s", [a]
        ).fetchone() == {"n": 0}
        with pytest.raises(LookupError):
            store.activate(conn, 9999)


def test_snapshot_round_trip_into_fresh_database(
    dsn: str, pg_admin_dsn: str, cfg: CorpusConfig, tmp_path: Path
) -> None:
    from sre_shared.testing.pg import session_database

    cv = ingest(cfg, dsn, tmp_path).corpus_version_id
    out = tmp_path / "snap.parquet"
    with store.open_conn(dsn) as conn:
        counts = snapshot.export_snapshot(conn, cv, out)
        src_rows = conn.execute(
            "select chunk_key, ordinal, source_path, section_path, header, body, char_start,"
            " char_end, token_count, emb_gte_small::text as v from public.chunks order by chunk_key"
        ).fetchall()
        src_docs = conn.execute(
            "select source_path, content_hash, parsed_text from public.documents order by source_path"
        ).fetchall()
    assert counts["chunks"] == len(src_rows) > 0

    with session_database(pg_admin_dsn) as fresh, store.open_conn(fresh) as conn:
        new_cv, loaded = snapshot.load_snapshot(conn, out, activate=True)
        assert loaded == counts
        got = conn.execute(
            "select chunk_key, ordinal, source_path, section_path, header, body, char_start,"
            " char_end, token_count, emb_gte_small::text as v from public.chunks order by chunk_key"
        ).fetchall()
        got_docs = conn.execute(
            "select source_path, content_hash, parsed_text from public.documents order by source_path"
        ).fetchall()
        status = conn.execute(
            "select status, embedding_models from public.corpus_versions where id=%s", [new_cv]
        ).fetchone()
    assert got == src_rows
    assert got_docs == src_docs
    assert status == {"status": "active", "embedding_models": ["gte-small"]}
    assert np.isfinite([float(x) for x in got[0]["v"].strip("[]").split(",")]).all()


def test_snapshot_export_keeps_only_selected_models(
    dsn: str, cfg: CorpusConfig, tmp_path: Path
) -> None:
    """CI's snapshot needs gte-small only; the lab corpus also holds the two BGE columns."""
    import json

    import pyarrow.parquet as pq

    cv = ingest(cfg, dsn, tmp_path).corpus_version_id
    with store.open_conn(dsn) as conn:
        conn.execute(
            "update public.chunks set emb_bge_m3 = array_fill(0.1, array[1024])::halfvec"
            " where corpus_version_id = %s",
            [cv],
        )
        conn.execute(
            "update public.corpus_versions set embedding_models = %s where id = %s",
            [["gte-small", "bge-m3"], cv],
        )
        conn.commit()
        out = tmp_path / "gte.parquet"
        snapshot.export_snapshot(conn, cv, out, ["gte-small"])
        with pytest.raises(ValueError, match="no embeddings for"):
            snapshot.export_snapshot(conn, cv, tmp_path / "x.parquet", ["bge-large"])
    table = pq.read_table(out)
    kinds = table.column("kind").to_pylist()
    m3 = [
        v
        for k, v in zip(kinds, table.column("emb_bge_m3").to_pylist(), strict=True)
        if k == "chunk"
    ]
    gte = [
        v
        for k, v in zip(kinds, table.column("emb_gte_small").to_pylist(), strict=True)
        if k == "chunk"
    ]
    assert m3 and all(v is None for v in m3)
    assert all(v is not None and len(v) == 384 for v in gte)
    info = json.loads(pq.read_schema(out).metadata[b"sre_snapshot"])
    assert info["embedding_models"] == ["gte-small"]


# --- chunking variants as non-active corpus versions ---------------------------------------


class RecordingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__("gte-small")
        self.texts: list[str] = []

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> Vectors:
        self.texts.extend(texts)
        return super().embed_documents(texts, batch_size)


def add_long_page(tmp_path: Path) -> None:
    body = "\n\n".join(" ".join(f"para{j}word{i}" for i in range(80)) for j in range(14))
    page = tmp_path / "sources" / "1.36" / "content/en/docs/concepts/long-page.md"
    page.write_text(f"---\ntitle: Long page\ncontent_type: concept\n---\n\n## Body\n\n{body}\n")


def variant_ingest(
    cfg: CorpusConfig,
    dsn: str,
    tmp_path: Path,
    name: str,
    emb: FakeEmbedder | None = None,
    **kw: object,
) -> RunResult:
    emb = emb or FakeEmbedder("gte-small")
    return run_ingest(
        cfg,
        dsn=dsn,
        do_fetch=False,
        data_dir=tmp_path,
        embedders={"gte-small": emb},
        token_counter=FakeEmbedder("gte-small"),
        chunking=load_chunking(name),
        log=lambda _m: None,
        **kw,  # type: ignore[arg-type]
    )


def default_rows(conn: store.Conn, cv: int) -> list[dict[str, object]]:
    return conn.execute(
        "select id, chunk_key, body, header, char_start, char_end, token_count,"
        " emb_gte_small::text as v from public.chunks where corpus_version_id=%s order by id",
        [cv],
    ).fetchall()


def test_variants_are_separate_ready_versions_and_leave_the_default_alone(
    dsn: str, cfg: CorpusConfig, tmp_path: Path
) -> None:
    add_long_page(tmp_path)
    base = ingest(cfg, dsn, tmp_path, activate=True)
    with store.open_conn(dsn) as conn:
        before = default_rows(conn, base.corpus_version_id)
    assert before

    ids: dict[str, int] = {}
    for name in ("t256", "o64", "nohdr"):
        res = variant_ingest(cfg, dsn, tmp_path, name)
        assert res.created_version and res.corpus_version_id != base.corpus_version_id
        ids[name] = res.corpus_version_id

    with store.open_conn(dsn) as conn:
        rows = conn.execute(
            "select id, label, chunker_version, status from public.corpus_versions order by id"
        ).fetchall()
        got = {r["id"]: (r["label"], r["chunker_version"], r["status"]) for r in rows}
        n = {
            name: conn.execute(
                "select count(*) as n, max(token_count) as mx from public.chunks"
                " where corpus_version_id=%s",
                [cv],
            ).fetchone()
            for name, cv in ids.items()
        }
        assert default_rows(conn, base.corpus_version_id) == before
    assert got[base.corpus_version_id] == ("k8s-1.36-chunker1", "1", "active")
    assert got[ids["t256"]] == ("k8s-1.36-chunker1-t256", "1-t256", "ready")
    assert got[ids["o64"]] == ("k8s-1.36-chunker1-o64", "1-o64", "ready")
    assert got[ids["nohdr"]] == ("k8s-1.36-chunker1-nohdr", "1-nohdr", "ready")
    assert n["t256"] is not None and n["t256"]["mx"] <= 256
    assert n["t256"]["n"] > len(before)
    assert n["nohdr"] is not None and n["nohdr"]["n"] == len(before)


def test_variant_ingested_twice_adds_nothing(dsn: str, cfg: CorpusConfig, tmp_path: Path) -> None:
    add_long_page(tmp_path)
    ingest(cfg, dsn, tmp_path, activate=True)
    first = variant_ingest(cfg, dsn, tmp_path, "o64")
    second = variant_ingest(cfg, dsn, tmp_path, "o64")
    assert second.corpus_version_id == first.corpus_version_id and not second.created_version
    assert second.chunks_inserted == 0 and second.documents_inserted == 0
    assert second.chunks_total == first.chunks_total > 0
    with store.open_conn(dsn) as conn:
        assert conn.execute("select count(*) as n from public.corpus_versions").fetchone() == {
            "n": 2
        }
        st = conn.execute(
            "select status from public.corpus_versions where id=%s", [first.corpus_version_id]
        ).fetchone()
    assert st == {"status": "ready"}


def test_nohdr_embeds_the_body_only(dsn: str, cfg: CorpusConfig, tmp_path: Path) -> None:
    default_emb, nohdr_emb = RecordingEmbedder(), RecordingEmbedder()
    base = variant_ingest(cfg, dsn, tmp_path, "default", default_emb)
    hdr = variant_ingest(cfg, dsn, tmp_path, "nohdr", nohdr_emb)
    assert default_emb.texts and all(t.startswith("Kubernetes v1.36 | ") for t in default_emb.texts)
    assert nohdr_emb.texts and not any("Kubernetes v1.36 | " in t for t in nohdr_emb.texts)
    with store.open_conn(dsn) as conn:
        bodies = {
            r["body"]
            for r in conn.execute(
                "select body from public.chunks where corpus_version_id=%s", [hdr.corpus_version_id]
            ).fetchall()
        }
        header_kept = conn.execute(
            "select count(*) as n from public.chunks where corpus_version_id=%s and header <> ''",
            [hdr.corpus_version_id],
        ).fetchone()
        stats = store.version_stats(conn, hdr.corpus_version_id)
    assert set(nohdr_emb.texts) == bodies
    assert header_kept == {"n": hdr.chunks_total}
    assert stats["chunking"]["header_in_embedding"] is False
    # Different text, different cache key: nothing is served from the default version's vectors.
    assert hdr.stats["embedding"]["gte-small"]["cache_hits"] == 0
    assert base.stats["embedding"]["gte-small"]["embedded"] == base.chunks_total
