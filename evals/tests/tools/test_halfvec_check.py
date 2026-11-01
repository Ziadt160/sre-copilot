"""Pure metric functions of tools/halfvec_check.py, on synthetic vectors (no DB, no model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[3] / "tools" / "halfvec_check.py"
_spec = importlib.util.spec_from_file_location("halfvec_check", _PATH)
assert _spec and _spec.loader
hv = importlib.util.module_from_spec(_spec)
sys.modules["halfvec_check"] = hv
_spec.loader.exec_module(hv)


def test_identical_rankings_give_overlap_one() -> None:
    # Orthogonal directions: scores are far apart at fp16 precision.
    docs = np.eye(8, dtype=np.float32)
    queries = np.eye(8, dtype=np.float32)
    c = hv.compare(queries, docs, [{i} for i in range(8)])
    assert c.overlap_at_10 == 1.0
    assert c.recall_at_5_fp32 == 1.0 and c.recall_at_5_fp16 == 1.0
    assert c.queries_with_changed_top10 == 0


def test_overlap_at_k_basics() -> None:
    assert hv.overlap_at_k([1, 2, 3], [3, 2, 1], 3) == 1.0
    assert hv.overlap_at_k([1, 2, 3, 4], [1, 2, 9, 8], 4) == 0.5
    assert hv.overlap_at_k([1, 2, 3, 4], [4, 3, 2, 1], 2) == 0.0
    assert hv.hit_at_k([5, 6, 7], {7}, 3) == 1.0
    assert hv.hit_at_k([5, 6, 7], {7}, 2) == 0.0


def test_top_k_orders_by_inner_product_with_stable_ties() -> None:
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    assert hv.top_k(q, docs, 3)[0].tolist() == [0, 2, 1]


def test_cast_fp16_rounds() -> None:
    x = np.array([[1.0 + 2.0**-12]], dtype=np.float32)  # below fp16 resolution near 1.0
    assert hv.cast_fp16(x)[0, 0] == 1.0


def test_crafted_near_tie_flips_under_fp16() -> None:
    # Doc 0 beats doc 1 by 2**-13 in fp32; fp16 rounds both to 1.0 so the stable tie-break
    # (lower index first) would keep the order, hence doc 1 is made the fp32 winner instead.
    docs = np.array([[1.0], [1.0 + 2.0**-12]], dtype=np.float32)
    q = np.array([[1.0]], dtype=np.float32)
    assert hv.top_k(q, docs, 2)[0].tolist() == [1, 0]  # fp32: doc 1 wins by a hair
    c = hv.compare(q, docs, [{1}], k_overlap=2, k_recall=1)
    assert c.queries_with_changed_top10 == 1  # fp16 ties and the stable order puts doc 0 first
    assert c.recall_at_5_fp32 == 1.0
    assert c.recall_at_5_fp16 == 0.0
    assert c.overlap_at_10 == 1.0  # same two docs, different order


def test_compare_skips_queries_without_gold_for_recall() -> None:
    docs = np.eye(3, dtype=np.float32)
    c = hv.compare(np.eye(3, dtype=np.float32), docs, [{0}, set(), {2}])
    assert c.n_queries == 3
    assert c.recall_at_5_fp32 == 1.0


def test_storage_numbers() -> None:
    r32, r16 = hv.storage_per_row(384)
    assert (r32, r16) == (1544, 776)  # 8 + 4*384 = 1544; 8 + 2*384 = 776
    est = hv.storage_estimate(1000, 384)
    assert est.fp32_total_bytes == 1000 * 1544
    assert est.fp16_total_bytes == 1000 * 776
    assert est.fp16_index_bytes < est.fp32_index_bytes
    assert hv.fmt_bytes(1536) == "1.5 KiB"
