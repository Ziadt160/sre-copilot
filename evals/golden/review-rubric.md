# Golden-set review rubric

Every candidate gets one decision under these rules. Each item records who decided (`reviewer`)
and why (`notes`). Golden v1 was reviewed by a model, not by a person with Kubernetes
experience (see "Who reviewed v1" in `README.md`).

Judge each item from its question, expected answer and gold passage (`quote`) only. Do not use
outside knowledge to make an item answerable, and do not look at the retrieval system or its
results: the golden set must not be tuned to the system it measures.

## Reject when any of these holds

1. The passage does not contain the answer, or only part of it, or the answer needs another page.
2. The question is vague or generic, or has many right answers ("What are some resources to
   learn about metrics?").
3. It asks about the documentation itself (what a page or section covers, how to contribute).
4. It is several questions in one and cannot be cut down to one answerable question.
5. `exact_string` only: the literal is not a real Kubernetes identifier as used in the passage
   (a placeholder such as `--meaningless-flag`, a domain name read as a field path such as
   `metadata.resource.k8s.io`), or the answer does not involve the literal.
6. It is a near-duplicate of an item already accepted.
7. The question gives away its own answer.

## Rewrite

Aim for at least 35% of accepted items in each stratum. Rewrite a question that is answerable
but reads like the documentation:

- Write it the way an engineer types into a search box or asks a teammate: plain, usually 6 to
  20 words, often task-first or symptom-first.
- Paraphrase. Avoid the passage's distinctive phrases and use an operator's words. Keep the
  standard Kubernetes terms (Pod, Deployment, kubelet, taint); do not obscure them.
- `exact_string`: keep the literal character for character; everything else may change.
- Do not add "in Kubernetes v1.xx" unless the question is about a version change: each item is
  already scoped to its version.
- One question, answerable from the passage alone.

Also rewrite a question whose content is fine but whose form is not (grammar, two questions
that can be cut to one).

## Expected answer

Fix it when it is wrong, incomplete or not supported by the passage: one to three sentences,
grounded in the passage.

## Quote

Trim it to the shortest contiguous span that holds the answer, usually one to three sentences
and at least about 12 words. Copy it exactly from the quote (whitespace may differ). For
`exact_string`, keep the literal inside the span when it is part of the answer. Leave the quote
whole when the whole passage is the answer (a short list, a small table).

## Quantity

Aim for about 40 accepted `factual` and about 30 accepted `exact_string` items, but never accept
a bad item to reach a number.
