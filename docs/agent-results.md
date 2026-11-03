# Agent results

The report `sre-eval agent run --report` writes, from runs 55 (kind `agent`) and 56 (kind
`conversation`) in the eval database. Git SHA `4d5468334723`, task set `agent-v1-d9528ec0eb9825d3`
(66 accepted items), rig `sre_evals.agent.live:lab`: the real agent graph and mcp-ops server over
the simulator, every role on `ollama:qwen2.5:7b`, with `--faults`. Reproduce with
`SRE_EVAL_DATABASE_URL=... sre-eval agent run --rig sre_evals.agent.live:lab --faults --store`.


### Agent (agent-v1, full set, rig sre_evals.agent.live:lab)

Injected faults: dup_write, redeliver, double_approval. Items: 45. Agent crashes (error events): 0.

| stratum | n | task success | tool | args | refusal | injection resisted | steps | duplicates |
|---|---|---|---|---|---|---|---|---|
| action | 25 | 0.88 [0.76, 1.00] | 0.84 [0.68, 0.96] | 0.84 [0.68, 0.96] | - | - | 1.9 | 0 |
| injection | 10 | 0.80 [0.50, 1.00] | 0.80 [0.40, 1.00] | 0.80 [0.40, 1.00] | - | 1.00 [1.00, 1.00] | 1.4 | 0 |
| permission | 10 | 0.80 [0.50, 1.00] | - | - | 0.80 [0.50, 1.00] | - | 0.8 | 0 |
| all | 45 | 0.84 [0.73, 0.93] | 0.83 [0.70, 0.97] | 0.83 [0.70, 0.97] | 0.80 [0.50, 1.00] | 1.00 [1.00, 1.00] | 1.5 | 0 |

Duplicate side effects: **0** (gate: 0). Gate met.

Cells are the item mean with a 95% bootstrap interval; `-` means the metric does not apply to that stratum. `steps` is the mean of proposed tool calls plus final answers.

### Agent (agent-v1, full set, rig sre_evals.agent.live:lab)

Injected faults: dup_write, redeliver, double_approval. Items: 21. Agent crashes (error events): 0.

| stratum | n | task success | rewrite recall@5 | steps | duplicates |
|---|---|---|---|---|---|
| multi_turn | 21 | 0.14 [0.00, 0.29] | 0.44 [0.31, 0.57] | 1.4 | 0 |
| all | 21 | 0.14 [0.00, 0.29] | 0.44 [0.31, 0.57] | 1.4 | 0 |

Duplicate side effects: **0** (gate: 0). Gate met.

Cells are the item mean with a 95% bootstrap interval; `-` means the metric does not apply to that stratum. `steps` is the mean of proposed tool calls plus final answers.
