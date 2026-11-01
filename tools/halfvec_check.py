"""What does halfvec cost versus fp32? An offline check for the design doc's storage table.

Re-embeds one Kubernetes version's chunk texts with OnnxGteSmall (fp32), then compares exact
(brute force) inner-product rankings from the fp32 vectors with rankings from the same vectors
cast to fp16, which is what a `halfvec` column stores. Queries are cast the same way, as the
search SQL casts its query parameter to halfvec. Reports, for golden-set questions of that
version and for N random chunk-header queries:

  * overlap@10: share of the fp32 top-10 that the fp16 top-10 also holds
  * recall@5 of the gold passage under each (golden questions; header queries use the chunk
    itself as gold)

and prints analytic bytes per row (fp32 vs fp16) with an HNSW index size estimate.

Read-only on the database. The lead runs it, e.g.:

    uv run python tools/halfvec_check.py --database-url postgresql://... --version 1.37

The metric parts are pure functions (tested in evals/tests/tools/test_halfvec_check.py).
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]

# pgvector on-disk sizes: vector = 4-byte varlena header + 4 (dim, unused) + 4 * dims;
# halfvec = the same 8-byte header + 2 * dims. Heap rows are padded to 8 bytes; TOAST does not
# apply at these sizes.
VECTOR_HEADER_BYTES = 8
HNSW_ELEMENT_HEADER_BYTES = 16  # element tuple fields before the vector data
HNSW_NEIGHBOR_BYTES = 8  # one neighbor slot in a neighbor tuple (tid + padding)


def cast_fp16(vecs: Matrix) -> Matrix:
    """Round-trip through float16, as a halfvec column does, back to float32 for scoring."""
    return vecs.astype(np.float16).astype(np.float32)


def top_k(queries: Matrix, docs: Matrix, k: int) -> NDArray[np.int64]:
    """Exact top-k doc indexes per query by inner product, best first.

    Ties break toward the lower doc index (stable), so identical inputs give identical rankings.
    """
    scores = queries @ docs.T
    order = np.argsort(-scores, axis=1, kind="stable")
    return order[:, :k].astype(np.int64)


def overlap_at_k(a: Sequence[int], b: Sequence[int], k: int) -> float:
    """Share of the first k of `a` that is also in the first k of `b`."""
    top_a, top_b = list(a[:k]), set(b[:k])
    if not top_a:
        return 1.0
    return sum(1 for x in top_a if x in top_b) / len(top_a)


def hit_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    """1.0 when any gold doc index is in the first k of `ranked`, else 0.0."""
    return 1.0 if any(i in gold for i in ranked[:k]) else 0.0


def mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


@dataclass(frozen=True)
class Comparison:
    n_queries: int
    overlap_at_10: float | None
    recall_at_5_fp32: float | None
    recall_at_5_fp16: float | None
    queries_with_changed_top10: int


def compare(
    queries: Matrix,
    docs: Matrix,
    gold: Sequence[set[int]],
    *,
    k_overlap: int = 10,
    k_recall: int = 5,
) -> Comparison:
    """fp32 vs fp16 rankings. `gold[i]` holds the doc indexes that count as hits for query i;
    queries with no gold are skipped for recall but still count for overlap."""
    if queries.shape[0] != len(gold):
        raise ValueError("one gold set per query")
    top32 = top_k(queries, docs, k_overlap)
    top16 = top_k(cast_fp16(queries), cast_fp16(docs), k_overlap)
    overlaps = [overlap_at_k(top32[i], top16[i], k_overlap) for i in range(len(gold))]
    changed = sum(1 for i in range(len(gold)) if list(top32[i]) != list(top16[i]))
    r32 = [hit_at_k(top32[i], g, k_recall) for i, g in enumerate(gold) if g]
    r16 = [hit_at_k(top16[i], g, k_recall) for i, g in enumerate(gold) if g]
    return Comparison(len(gold), mean(overlaps), mean(r32), mean(r16), changed)


@dataclass(frozen=True)
class StorageEstimate:
    dims: int
    rows: int
    fp32_row_bytes: int
    fp16_row_bytes: int
    fp32_total_bytes: int
    fp16_total_bytes: int
    fp32_index_bytes: int
    fp16_index_bytes: int


def _align8(n: int) -> int:
    return (n + 7) & ~7


def storage_per_row(dims: int) -> tuple[int, int]:
    """Analytic bytes of one vector value: (fp32, fp16), 8-byte aligned."""
    return (
        _align8(VECTOR_HEADER_BYTES + 4 * dims),
        _align8(VECTOR_HEADER_BYTES + 2 * dims),
    )


def hnsw_index_bytes(rows: int, dims: int, *, bytes_per_dim: int, m: int = 16) -> int:
    """Rough HNSW size: each element holds its vector plus 2*m level-0 neighbor slots, and about
    1/(m-1) of elements also sit on upper layers with m slots. An estimate for the table,
    not a measurement (compare with pg_relation_size on the lab index)."""
    vec = _align8(VECTOR_HEADER_BYTES + bytes_per_dim * dims)
    per_element = HNSW_ELEMENT_HEADER_BYTES + vec + 2 * m * HNSW_NEIGHBOR_BYTES
    upper = (m * HNSW_NEIGHBOR_BYTES) / max(m - 1, 1)
    return int(rows * (per_element + upper))


def storage_estimate(rows: int, dims: int, m: int = 16) -> StorageEstimate:
    r32, r16 = storage_per_row(dims)
    return StorageEstimate(
        dims=dims,
        rows=rows,
        fp32_row_bytes=r32,
        fp16_row_bytes=r16,
        fp32_total_bytes=r32 * rows,
        fp16_total_bytes=r16 * rows,
        fp32_index_bytes=hnsw_index_bytes(rows, dims, bytes_per_dim=4, m=m),
        fp16_index_bytes=hnsw_index_bytes(rows, dims, bytes_per_dim=2, m=m),
    )


def fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GiB"  # unreachable


# --- the part that touches the database and the model -------------------------------------


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _print_comparison(label: str, c: Comparison) -> None:
    print(
        f"{label:<22} n={c.n_queries:<5} overlap@10={_fmt(c.overlap_at_10)} "
        f"recall@5 fp32={_fmt(c.recall_at_5_fp32)} fp16={_fmt(c.recall_at_5_fp16)} "
        f"top10-order-changed={c.queries_with_changed_top10}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--database-url", required=True, help="Postgres DSN to read chunks from")
    ap.add_argument("--version", default="1.37", help="Kubernetes version to check")
    ap.add_argument("--golden", default="v1", help="golden set version under evals/golden/")
    ap.add_argument("--n-header-queries", type=int, default=200)
    ap.add_argument("--corpus-version-id", type=int, default=None, help="default: the active one")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # Imported here so the pure functions import without the model stack.
    from sre_evals.metrics.core import match_grade
    from sre_shared.db import connect_sync
    from sre_shared.embed.onnx_gte import OnnxGteSmall
    from sre_shared.golden_io import golden_path, load_golden

    with connect_sync(args.database_url) as conn:
        cv = args.corpus_version_id
        if cv is None:
            row = conn.execute(
                "select id from public.corpus_versions where status = 'active'"
            ).fetchone()
            if row is None:
                print("no active corpus version; pass --corpus-version-id", file=sys.stderr)
                return 2
            cv = int(row["id"])
        chunks = conn.execute(
            "select id, source_path, k8s_version, header, body from public.chunks "
            "where corpus_version_id = %s and k8s_version = %s order by id",
            (cv, args.version),
        ).fetchall()
    if not chunks:
        print(f"no chunks for version {args.version} in corpus version {cv}", file=sys.stderr)
        return 2

    embedder = OnnxGteSmall()
    texts = [f"{c['header']}\n\n{c['body']}" if c["header"] else c["body"] for c in chunks]
    print(f"embedding {len(texts)} chunks of {args.version} (corpus version {cv}) in fp32 ...")
    docs = embedder.embed_documents(texts)

    # Golden questions whose gold passages are in this version.
    items = [
        i
        for i in load_golden(golden_path(args.golden), accepted_only=True)
        if i.gold_passages and all(p.k8s_version == args.version for p in i.gold_passages)
    ]
    q_golden = embedder.embed_queries([i.question for i in items])
    gold_golden: list[set[int]] = []
    for item in items:
        gold_golden.append(
            {
                idx
                for idx, c in enumerate(chunks)
                if any(match_grade(_Chunk(c), p) >= 1 for p in item.gold_passages)
            }
        )

    rng = random.Random(args.seed)  # noqa: S311 - sampling, not security
    picks = rng.sample(range(len(chunks)), min(args.n_header_queries, len(chunks)))
    q_header = embedder.embed_queries([chunks[i]["header"] for i in picks])
    gold_header = [{i} for i in picks]

    print()
    if items:
        _print_comparison("golden questions", compare(q_golden, docs, gold_golden))
    else:
        print(f"no accepted golden items of version {args.version}")
    _print_comparison("chunk-header queries", compare(q_header, docs, gold_header))

    est = storage_estimate(len(chunks), embedder.spec.dims)
    print()
    print(f"storage, {est.rows} rows x {est.dims} dims (vector value only, analytic)")
    print(f"  per row : fp32 {est.fp32_row_bytes} B   fp16 {est.fp16_row_bytes} B")
    print(
        f"  column  : fp32 {fmt_bytes(est.fp32_total_bytes)}   "
        f"fp16 {fmt_bytes(est.fp16_total_bytes)}"
    )
    print(
        f"  HNSW est: fp32 {fmt_bytes(est.fp32_index_bytes)}   "
        f"fp16 {fmt_bytes(est.fp16_index_bytes)}   (m=16; estimate, verify with pg_relation_size)"
    )
    return 0


class _Chunk:
    """Adapts a chunk row to the metrics' ChunkLike."""

    def __init__(self, row: dict[str, str]) -> None:
        self.source_path = row["source_path"]
        self.k8s_version = row["k8s_version"]
        self.body = row["body"]


if __name__ == "__main__":
    raise SystemExit(main())
