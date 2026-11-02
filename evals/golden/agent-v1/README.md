# agent-v1: the agent task set

66 accepted items (70 drafted) that `sre-eval agent run` plays against the agent over the
simulator: whole tasks scored by end state, not single answers.

| Stratum | Accepted | What it checks |
|---|---|---|
| action | 25 | the right tool and arguments, the approval when one is needed, the cluster's end state |
| multi_turn | 21 | follow-ups rewritten into standalone questions (retrieval recall of the rewrite, required terms) and grounded answers per turn |
| permission | 10 | a caller without the scope, namespace or thread is refused, or the action waits for an admin |
| injection | 10 | instructions planted in logs or passages produce no tool call |

`--faults` also replays every write with the first reply dropped, delivers the task body again,
and sends two approvals at once; the Phase 3 gate is zero duplicate side effects under all three.

## Who reviewed it

Models, not a person with Kubernetes or SRE experience, under `review-rubric.md` and the same
procedure as golden v1 (decision 27). Neither reviewer saw the agent's code or any run.

- Pass A: Claude Opus 5.5 (`review.pass-a.jsonl`): 53 accept, 13 revise, 4 reject.
- Pass B: Claude Sonnet 5.5, independently (`review.pass-b.jsonl`): 56 accept, 10 revise, 4 reject.
- Agreement on accept / revise / reject: 60 of 70 (85.7%, Cohen's kappa 0.61); on keep versus
  reject: 66 of 70 (94.3%, kappa 0.47; rejects are rare, which keeps kappa low).
- Disagreements were settled against the scenario, the policy and the passages; each item's
  `notes` records both decisions and, where they differed, the settlement and why.

Rejected (kept in `items.jsonl` with `status: rejected`, not run): mt-003 and mt-025 (repeat
other conversations' turns), mt-009 (another item's gold pair in reverse), mt-013 (its follow-up
contradicts itself). Revisions mostly loosened `rewrite_terms` that a correct rewrite could miss
and made follow-ups lean on the earlier turn.
