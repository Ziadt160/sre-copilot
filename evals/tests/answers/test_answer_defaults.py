"""Experiments inherit config/answer.yaml, and answers that reached max_tokens are counted
(gpt-oss spends hidden reasoning against the same budget: decision 34)."""

from __future__ import annotations

from sre_evals.answers.cli import effective_answer_config
from sre_evals.answers.runner import ItemRun, summarize
from sre_evals.answers.scoring import score_answer
from sre_evals.answers.testing import W, chunk, golden, result
from sre_evals.runner.config import ExperimentConfig
from sre_shared.config import load_answer_config


def test_an_experiment_block_overrides_only_the_keys_it_sets() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "answer-t", "legs": ["dense"], "answer": {"k": 8, "compression": "sentence"}}
    )
    base = load_answer_config()
    got = effective_answer_config(cfg.answer)
    assert (got.k, got.compression) == (8, "sentence")
    # Unset keys come from the yaml, not from the model defaults.
    assert (got.max_tokens, got.reasoning_effort) == (base.max_tokens, base.reasoning_effort)
    assert effective_answer_config(None) == base


def test_answers_that_reached_max_tokens_are_counted() -> None:
    item = golden("fac-0001", "factual", W(1, 12))
    ctx = [chunk(1, W(1, 12))]
    runs = []
    for completion in (20, 512, 600):
        res = result(item.question, "A [1].", ctx, [1])
        res = res.model_copy(update={"completion_tokens": completion})
        runs.append(ItemRun(item, res, score_answer(res, item)))
    assert summarize(runs, 3)["n_hit_max_tokens"] == 2  # result() uses max_tokens 512


def test_the_answer_experiments_differ_from_the_baseline_only_where_named() -> None:
    from sre_evals.answers.cli import load_answer_experiments

    exps = load_answer_experiments()
    base = effective_answer_config(exps["answer-k5"].answer)
    want = {
        "answer-k3": {"k": 3},
        "answer-k8": {"k": 8},
        "answer-k5-sentence": {"compression": "sentence"},
    }
    for name, diff in want.items():
        got = effective_answer_config(exps[name].answer)
        assert got == base.model_copy(update=diff), name
