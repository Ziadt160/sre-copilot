"""Eval runner: runs experiment configs against the golden set and stores scores."""

from __future__ import annotations

from sre_evals.runner.config import (
    ExperimentConfig,
    experiments_dir,
    load_experiment,
    load_experiments,
)
from sre_evals.runner.engine import RunOutcome, run_experiment
from sre_evals.runner.gate import (
    GateResult,
    check_gate,
    load_baseline,
    make_baseline,
    write_baseline,
)

__all__ = [
    "ExperimentConfig",
    "GateResult",
    "RunOutcome",
    "check_gate",
    "experiments_dir",
    "load_baseline",
    "load_experiment",
    "load_experiments",
    "make_baseline",
    "run_experiment",
    "write_baseline",
]
