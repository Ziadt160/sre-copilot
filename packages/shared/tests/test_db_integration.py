"""Shared base against a real Postgres: migrations, RLS by access group, seed helper."""

from typing import Any

import numpy as np
import psycopg
import pytest
from pgvector import HalfVector

from sre_shared.db import as_user
from sre_shared.embed.fake import FakeEmbedder
from sre_shared.models import ANONYMOUS, Claims
from sre_shared.testing.seed import FixtureDoc, create_corpus_version, insert_docs

pytestmark = pytest.mark.integration

PUBLIC_DOC = FixtureDoc(
    source_path="content/en/docs/concepts/workloads/pods/_index.md",
    k8s_version="1.34",
    title="Pods",
    sections=[((), "Pods are the smallest deployable units of computing."), (("Lifecycle",), "A Pod restarts containers that exit with CrashLoopBackOff.")],
)
TEAM_A_DOC = FixtureDoc(
    source_path="internal/runbooks/payments.md",
    k8s_version="1.34",
    title="Payments runbook",
    sections=[((), "Team A secret runbook for the payments namespace.")],
    access_group="team-a",
)


async def _seed(db: Any) -> int:
    cv = await create_corpus_version(db)
    await insert_docs(db, [PUBLIC_DOC, TEAM_A_DOC], corpus_version_id=cv, embedders=[FakeEmbedder()])
    return cv


async def _visible_titles(db: Any, claims: Claims) -> set[str]:
    async with as_user(db, claims) as conn:
        rows = await (await conn.execute("select distinct title from public.chunks")).fetchall()
    return {r["title"] for r in rows}


async def test_schema_and_extension(db: Any) -> None:
    row = await (
        await db.execute("select extversion from pg_extension where extname = 'vector'")
    ).fetchone()
    major, minor, *_ = (int(x) for x in row["extversion"].split("."))
    assert (major, minor) >= (0, 8)
    cols = await (
        await db.execute(
            "select column_name, format_type(atttypid, atttypmod) as t"
            " from information_schema.columns c"
            " join pg_attribute a on a.attname = c.column_name"
            "  and a.attrelid = 'public.chunks'::regclass"
            " where table_name = 'chunks' and column_name like 'emb_%'"
        )
    ).fetchall()
    assert {c["column_name"]: c["t"].removeprefix("extensions.") for c in cols} == {
        "emb_gte_small": "halfvec(384)",
        "emb_bge_large": "halfvec(1024)",
        "emb_bge_m3": "halfvec(1024)",
    }


async def test_rls_limits_rows_to_caller_groups(db: Any, user_a: Claims, user_b: Claims) -> None:
    await _seed(db)
    assert await _visible_titles(db, user_a) == {"Pods", "Payments runbook"}
    assert await _visible_titles(db, user_b) == {"Pods"}
    assert await _visible_titles(db, ANONYMOUS) == {"Pods"}


async def test_user_groups_helper(db: Any, user_a: Claims) -> None:
    async with as_user(db, user_a) as conn:
        row = await (await conn.execute("select sre.user_groups() as g")).fetchone()
    assert row["g"] == ["team-a", "public"]
    async with as_user(db, ANONYMOUS) as conn:
        row = await (await conn.execute("select sre.user_groups() as g")).fetchone()
    assert row["g"] == ["public"]


async def test_callers_cannot_write(db: Any, user_a: Claims) -> None:
    await _seed(db)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with as_user(db, user_a) as conn:
            await conn.execute("delete from public.chunks")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with as_user(db, user_a) as conn:
            await conn.execute("select count(*) from public.eval_runs")


async def test_one_active_corpus_version(db: Any) -> None:
    await create_corpus_version(db, label="a", status="active")
    with pytest.raises(psycopg.errors.UniqueViolation):
        await create_corpus_version(db, label="b", status="active")


async def test_halfvec_round_trip_and_active_version(db: Any) -> None:
    cv = await _seed(db)
    row = await (await db.execute("select sre.active_corpus_version() as id")).fetchone()
    assert row["id"] == cv
    text = "Kubernetes v1.34 | Pods\n\nPods are the smallest deployable units of computing."
    expected = FakeEmbedder().embed_documents([text])[0]
    row = await (
        await db.execute("select emb_gte_small from public.chunks where ordinal = 0 and title = 'Pods'")
    ).fetchone()
    stored = row["emb_gte_small"]
    assert isinstance(stored, HalfVector)
    np.testing.assert_allclose(stored.to_numpy(), expected, atol=1e-3)
    # The dense operator works on the stored type (negative inner product).
    row = await (
        await db.execute(
            "select (emb_gte_small <#> %s) as d from public.chunks where ordinal = 0 and title = 'Pods'",
            [HalfVector(expected)],
        )
    ).fetchone()
    assert row["d"] == pytest.approx(-1.0, abs=1e-2)


async def test_keyword_columns_are_generated(db: Any) -> None:
    await _seed(db)
    rows = await (
        await db.execute(
            "select title from public.chunks"
            " where tsv_simple @@ websearch_to_tsquery('simple', 'CrashLoopBackOff')"
        )
    ).fetchall()
    assert [r["title"] for r in rows] == ["Pods"]
