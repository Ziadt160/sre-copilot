# Faithfulness labelling rubric (labels v1)

A faithfulness label says whether an answer's claims are supported by the sources the model
was given for that answer. It is a reading task. It does not ask whether the answer is true,
complete, well written or correctly cited; those are measured separately (citation hit,
answer relevancy). The judge models are calibrated against these labels, so the rubric
follows the judge metric's definition: DeepEval faithfulness, the share of the answer's
claims that the sources support, with ambiguous claims counted against
(`penalize_ambiguous_claims`). A judge score maps to a label with `stats.score_to_label`
(>= 0.9 supported, >= 0.5 partly, else not supported).

## What counts as a claim

A claim is any statement about the world that the answer makes: a fact, a cause, a step to
take, a command, a flag, a field name, a default value, a version, a limit. A command or a
YAML snippet is a claim that this command or field exists and does what the answer says.

These are not claims and are ignored:

* citation markers such as `[2]` (whether they point at the right source is not judged here)
* formatting, headings, politeness, restating the question
* statements about the sources themselves ("the sources do not say how to ..."), as long as
  they are accurate

## What counts as supported

A claim is supported when a source states it, or it follows directly from what a source
states (a paraphrase, or a straightforward combination of two sentences). Any of the sources
given may support it, cited or not.

A claim is **not** supported when it:

* is absent from every source, even if it is true Kubernetes knowledge (grounding, not truth)
* contradicts a source
* goes beyond the sources: a specific flag, value, field, version or step the sources do not
  contain, or a generalisation the sources do not make
* is ambiguous enough that you cannot tell whether the sources support it

## The three labels

| Label | When |
|---|---|
| `supported` | Every claim is supported (at most a trivial, harmless gap, such as an obvious connective step) |
| `partly` | At least half of the claims are supported, and one or more is not |
| `not_supported` | Fewer than half of the claims are supported, or the claim that answers the question is unsupported or contradicted |

An answer that is cut off mid-sentence is judged on what it says. An answer that correctly
says the sources do not cover the question makes no unsupported claim and is `supported`.

## Procedure

Two independent passes label the same sample without seeing each other's labels, the judges'
scores or the gold passages: Claude Opus 5.5 and Claude Sonnet 5.5, each from this rubric and
a packet holding only the question, the answer and the sources. Agreement between the two
passes is reported (accuracy, Cohen's kappa). Where they disagree, the lead (Claude Opus 5.5)
re-reads the answer against the sources and settles the label. These are model labels and are
called that everywhere: the owner does not have the Kubernetes background to judge them
(see decision 27 for the golden set, reviewed the same way).
