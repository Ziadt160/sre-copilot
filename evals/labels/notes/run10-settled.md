# Run 10: labelling passes and settled disagreements

Sample: run 10 (gpt-oss-20b, provider-default reasoning, 512 tokens), all 28 labellable
answers (2 of 30 were empty), labelled under [the rubric](../faithfulness-rubric.md).

| | supported | partly | not supported |
|---|---|---|---|
| Pass A (Claude Opus 5.5) | 22 | 5 | 1 |
| Pass B (Claude Sonnet 5.5) | 26 | 2 | 0 |
| Final | 23 | 4 | 1 |

Pass A vs pass B: 3-class accuracy 0.821, kappa 0.307; binary (supported vs not) accuracy
0.857, kappa 0.440. Kappa is low next to accuracy because about 80% of answers are supported,
so chance agreement is high. Against the final labels: pass A accuracy 0.964 (kappa 0.891),
pass B 0.857 (kappa 0.371). Pass B was the more lenient: three of its four changed labels were
claims it noted itself but kept as supported.

Settled by the lead (Claude Opus 5.5), re-reading each answer against its sources:

| Item | A | B | Final | Why |
|---|---|---|---|---|
| fac-0017 | partly | supported | partly | "meets policy or schema requirements": no source says admission checks schema; source 2 places schema validation after admission |
| exs-0036 | partly | supported | partly | offers "move the rule to a more granular level" as a fix; source 1 says the item-level rule has the same estimated cost |
| exs-0026 | not supported | partly | not supported | the "because" clause that answers the question says an OOM kill removes the filtering rules; source 1 says they are not cleaned up (the rubric's rule for the claim that answers the question) |
| exs-0019 | partly | supported | partly | "value may be any JSON type" goes beyond source 5's list; "reference the request object via the `Object` type" misstates it (`Object` is the type for building values) |
| exs-0022 | partly | supported | supported | source 5: a schedulerName mismatch makes the scheduler reject the group "as unschedulable"; with source 1's reasons, `Unschedulable` follows from the two sentences together |

The packet the passes read (question, answer and sources as the model saw them, no scores or
gold) is rebuilt byte for byte from run 10's stored rows with
`tools/label_packet.py export --run 10`; `merge` reproduces the agreement figures above.
