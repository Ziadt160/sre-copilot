"""Fetch run data and render `docs/results.md` (details behind the README table)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from psycopg import AsyncConnection

from sre_evals.report.readme import run_model
from sre_shared.embed.base import SPECS

RUN_COLUMNS = (
    "id, experiment, loop, config, stamps, golden_version, corpus_version_id, summary, started_at"
)


async def latest_runs(conn: AsyncConnection[Any]) -> list[dict[str, Any]]:
    """Latest `done` retrieval run per experiment (dict rows)."""
    cur = await conn.execute(
        f"select distinct on (experiment) {RUN_COLUMNS} from public.eval_runs"
        " where kind = 'retrieval' and status = 'done'"
        " order by experiment, started_at desc, id desc"
    )
    return [dict(r) for r in await cur.fetchall()]


async def latest_sweep_runs(conn: AsyncConnection[Any]) -> list[dict[str, Any]]:
    """Latest `done` run per sweep point (kinds `sweep` and `hnsw`), oldest first, each with
    `folds`: recall@5 and MRR@10 on the two fixed halves of the golden set."""
    cur = await conn.execute(
        f"select * from (select distinct on (experiment) kind, {RUN_COLUMNS}"
        " from public.eval_runs where kind in ('sweep', 'hnsw') and status = 'done'"
        " order by experiment, started_at desc, id desc) latest order by id"
    )
    runs = [dict(r) for r in await cur.fetchall()]
    for r in runs:
        cur = await conn.execute(
            "select item_id, metric, value from public.eval_scores where run_id = %s"
            " and ((metric = 'recall' and k = 5) or (metric = 'mrr' and k = 10))",
            [r["id"]],
        )
        per_item: dict[str, dict[str, float]] = {}
        for row in await cur.fetchall():
            per_item.setdefault(row["item_id"], {})[row["metric"]] = float(row["value"])
        r["folds"] = fold_means(per_item)
    return runs


def fold_of(item_ids: Sequence[str]) -> dict[str, str]:
    """Two fixed halves: items alternate A/B in id order within each stratum (the id prefix,
    `fac` or `exs`), so each half keeps the stratum mix and the split never depends on scores."""
    groups: dict[str, list[str]] = {}
    for iid in sorted(item_ids):
        groups.setdefault(iid.split("-", 1)[0], []).append(iid)
    return {iid: "AB"[i % 2] for ids in groups.values() for i, iid in enumerate(ids)}


def fold_means(per_item: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean recall and MRR per fold, from {item_id: {"recall": x, "mrr": y}}."""
    folds = fold_of(list(per_item))
    out: dict[str, dict[str, float]] = {}
    for fold in "AB":
        rows = [v for iid, v in per_item.items() if folds[iid] == fold]
        if rows:
            out[fold] = {m: sum(r.get(m, 0.0) for r in rows) / len(rows) for m in ("recall", "mrr")}
    return out


async def storage_stats(
    conn: AsyncConnection[Any], corpus_version_id: int | None
) -> dict[str, dict[str, Any]]:
    """Per model: MB used by its vector column (sum of pg_column_size), vector count, and the
    embedding seconds the ingest recorded (`stats.embedding.<model>.seconds`, as written by
    sre_ingest; `embed_seconds` / `ingest_seconds` keyed by model are also read)."""
    if corpus_version_id is None:
        return {}
    crow = await (
        await conn.execute(
            "select stats from public.corpus_versions where id = %s", [corpus_version_id]
        )
    ).fetchone()
    stats: dict[str, Any] = (crow["stats"] if crow else None) or {}
    out: dict[str, dict[str, Any]] = {}
    for name, spec in SPECS.items():
        col = spec.column  # whitelisted: comes from SPECS, never from user input
        row = await (
            await conn.execute(
                f"select count({col}) as n, coalesce(sum(pg_column_size({col})), 0) as bytes"
                " from public.chunks where corpus_version_id = %s",
                [corpus_version_id],
            )
        ).fetchone()
        assert row is not None
        if not row["n"]:
            continue
        seconds = None
        embedding = stats.get("embedding")
        if isinstance(embedding, dict) and isinstance(embedding.get(name), dict):
            raw = embedding[name].get("seconds")
            seconds = float(raw) if isinstance(raw, int | float) else None
        for key in ("embed_seconds", "ingest_seconds"):
            per_model = stats.get(key)
            if seconds is None and isinstance(per_model, dict) and name in per_model:
                seconds = float(per_model[name])
        out[name] = {
            "vectors": int(row["n"]),
            "mb": int(row["bytes"]) / 1_000_000,
            "ingest_seconds": seconds,
        }
    return out


def _f(x: Any) -> str:
    return f"{float(x):.3f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _params(p: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())


def _fold(r: dict[str, Any], metric: str) -> str:
    f = r.get("folds") or {}
    if "A" not in f or "B" not in f:
        return "-"
    return f"{f['A'][metric]:.3f} / {f['B'][metric]:.3f}"


def _hnsw_key(r: dict[str, Any]) -> tuple[int, int, int]:
    p = r["summary"]["params"]
    return (int(p["m"]), int(p["ef_construction"]), int(p["ef_search"]))


def render_sweeps(sweeps: Sequence[dict[str, Any]]) -> list[str]:
    """Sections for parameter sweeps (kind `sweep`) and HNSW build sweeps (kind `hnsw`)."""
    if not sweeps:
        return []
    lines = [
        "",
        "## Sweeps",
        "",
        "Each sweep point is scored on the same items it is chosen on, so the best point "
        "looks better than it would on new questions. Folds A / B are two fixed halves of "
        "the golden set (items alternate within each stratum): a point that wins on both "
        "halves is more than noise.",
    ]
    by_sweep: dict[str, list[dict[str, Any]]] = {}
    for r in sweeps:
        by_sweep.setdefault(str(r["summary"].get("sweep", r["experiment"])), []).append(r)
    for name, runs in by_sweep.items():
        kind = runs[0]["kind"]
        lines += ["", f"### {name} ({kind})", ""]
        rows: list[list[str]] = []
        if kind == "hnsw":
            for r in sorted(runs, key=_hnsw_key):
                s, h = r["summary"], r["summary"].get("hnsw", {})
                m, efc, ef = _hnsw_key(r)
                ann = s.get("ann_recall") or {}
                rows.append([
                    str(m), str(efc), f"{h.get('build_seconds', 0):.1f}",
                    f"{h.get('index_mb', 0):.1f}", str(ef),
                    _f(next(iter(ann.values()))) if ann else "-",
                    _f(s["recall@5"]), _f(s["mrr@10"]), _fold(r, "recall"),
                    f"{s['latency_ms']['p50']:.0f}", f"{s['latency_ms']['p95']:.0f}",
                ])  # fmt: skip
            header = [
                "m", "ef_construction", "build s", "index MB", "ef_search", "ANN recall@10",
                "recall@5", "MRR@10", "recall@5 A / B", "p50 ms", "p95 ms",
            ]  # fmt: skip
        else:
            for r in runs:
                s = r["summary"]
                ann = s.get("ann_recall") or {}
                rows.append([
                    _params(s.get("params", {})), _f(s["recall@5"]), _f(s["mrr@10"]),
                    _f(s["ndcg@10"]), _fold(r, "recall"), _fold(r, "mrr"),
                    _f(next(iter(ann.values()))) if ann else "-",
                    f"{s['latency_ms']['p95']:.0f}",
                ])  # fmt: skip
            header = [
                "parameters", "recall@5", "MRR@10", "nDCG@10", "recall@5 A / B",
                "MRR@10 A / B", "ANN recall@10", "p95 ms",
            ]  # fmt: skip
        lines += _table(header, rows)
        base = str(runs[0]["experiment"]).split("[", 1)[0]
        lines += ["", f"Base experiment: `{base}`. Golden version: `{runs[0]['golden_version']}`."]
    return lines


def render_details(
    runs: Sequence[dict[str, Any]],
    storage: dict[str, dict[str, Any]] | None = None,
    sweeps: Sequence[dict[str, Any]] = (),
) -> str:
    """Markdown for docs/results.md. Deterministic for the same inputs (no timestamps)."""
    runs = sorted(runs, key=lambda r: r["experiment"])
    lines = [
        "# Retrieval results",
        "",
        "Generated by `sre-eval report` from the latest `done` run of each experiment. "
        "Do not edit by hand.",
        "",
    ]
    if not runs:
        return "\n".join([*lines, "_No eval runs yet._", ""])

    shas = sorted({str(r["stamps"].get("git_sha", "unknown")) for r in runs})
    goldens = sorted({str(r["golden_version"]) for r in runs})
    lines += [
        f"Git SHA: {', '.join(f'`{s}`' for s in shas)}. "
        f"Golden version: {', '.join(f'`{g}`' for g in goldens)}.",
        "",
        "## All experiments",
        "",
    ]
    rows = []
    for r in runs:
        s = r["summary"]
        rows.append(
            [
                f"`{r['experiment']}`",
                run_model(r["config"]),
                _f(s["recall@3"]) if "recall@3" in s else "-",
                _f(s["recall@5"]),
                _f(s["recall@10"]) if "recall@10" in s else "-",
                _f(s["mrr@10"]),
                _f(s["ndcg@10"]),
                _f(s["empty_result_rate"]),
                f"{s['latency_ms']['p50']:.0f}",
                f"{s['latency_ms']['p95']:.0f}",
                f"{s['n_scored']}/{s['n_items']}",
            ]
        )
    lines += _table(
        [
            "Experiment", "Model", "recall@3", "recall@5", "recall@10", "MRR@10", "nDCG@10",
            "empty rate", "p50 ms", "p95 ms", "scored/items",
        ],
        rows,
    )  # fmt: skip

    lines += ["", "## Per stratum (recall@5 / MRR@10 / nDCG@10)", ""]
    strata = sorted({st for r in runs for st in r["summary"].get("by_stratum", {})})
    if strata:
        rows = []
        for r in runs:
            by = r["summary"].get("by_stratum", {})
            cells = [f"`{r['experiment']}`"]
            for st in strata:
                g = by.get(st)
                cells.append(
                    f"{_f(g['recall@5'])} / {_f(g['mrr@10'])} / {_f(g['ndcg@10'])} (n={g['n']})"
                    if g
                    else "-"
                )
            rows.append(cells)
        lines += _table(["Experiment", *strata], rows)
    else:
        lines.append("_No per-stratum data._")

    sweep = [
        r
        for r in runs
        if r["config"].get("legs") == ["dense", "keyword"]
        and r["config"].get("embedding_model") == "gte-small"
        and r["config"].get("weights", {}).get("dense") == 1.0
    ]
    lines += ["", "## Weight sweep (gte-small hybrid, w_dense = 1)", ""]
    if len(sweep) > 1:
        sweep.sort(key=lambda r: float(r["config"]["weights"]["keyword"]))
        lines += _table(
            ["w_keyword", "Experiment", "recall@5", "MRR@10", "nDCG@10"],
            [
                [
                    f"{float(r['config']['weights']['keyword']):g}",
                    f"`{r['experiment']}`",
                    _f(r["summary"]["recall@5"]),
                    _f(r["summary"]["mrr@10"]),
                    _f(r["summary"]["ndcg@10"]),
                ]
                for r in sweep
            ],
        )
    else:
        lines.append("_Sweep not run yet._")

    ann = [r for r in runs if r["summary"].get("ann_recall")]
    lines += ["", "## ANN recall (index vs exact scan, dense-only top 10)", ""]
    if ann:
        efs = sorted({int(ef) for r in ann for ef in r["summary"]["ann_recall"]})
        lines += _table(
            ["Experiment", "Sample", *[f"ef_search={ef}" for ef in efs]],
            [
                [
                    f"`{r['experiment']}`",
                    str(r["summary"].get("ann_sample", "-")),
                    *[
                        _f(r["summary"]["ann_recall"][str(ef)])
                        if str(ef) in r["summary"]["ann_recall"]
                        else "-"
                        for ef in efs
                    ],
                ]
                for r in ann
            ],
        )
    else:
        lines.append("_ANN recall not measured yet (`sre-eval retrieval --ann-recall`)._")

    lines += ["", "## Embedding models: quality and cost", ""]
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for r in runs:
        legs = r["config"].get("legs", [])
        if "dense" in legs:
            kind = "hybrid" if "keyword" in legs else "dense"
            if kind == "hybrid" and r["config"]["weights"]["keyword"] != 1.0:
                continue
            by_model.setdefault(r["config"]["embedding_model"], {})[kind] = r["summary"]
    storage = storage or {}
    if by_model:
        rows = []
        for model in SPECS:
            if model not in by_model:
                continue
            d, h = by_model[model].get("dense"), by_model[model].get("hybrid")
            st = storage.get(model, {})
            secs = st.get("ingest_seconds")
            rows.append(
                [
                    model,
                    str(SPECS[model].dims),
                    _f(d["recall@5"]) if d else "-",
                    _f(h["recall@5"]) if h else "-",
                    _f(d["mrr@10"]) if d else "-",
                    _f(h["mrr@10"]) if h else "-",
                    f"{st['mb']:.1f}" if st else "-",
                    f"{secs:.0f}" if secs is not None else "-",
                    SPECS[model].backend,
                ]
            )
        lines += _table(
            [
                "Model", "Dims", "dense recall@5", "hybrid recall@5", "dense MRR@10",
                "hybrid MRR@10", "vector column MB", "embed s", "embed backend",
            ],
            rows,
        )  # fmt: skip
        lines += [
            "",
            "Vector column MB is the sum of `pg_column_size` over the model's halfvec column "
            "for the corpus version of the run (index size not included). Embed s is the "
            "document-embedding time the ingest recorded for the whole corpus; `onnx` runs on "
            "the CPU and `sentence-transformers` on the GPU embedding server, so the column "
            "compares machines as much as models.",
        ]
    else:
        lines.append("_No dense experiments yet._")
    lines += render_sweeps(sweeps)
    return "\n".join(lines) + "\n"
