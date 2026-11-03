# Semantic cache calibration

`tools/cache_calibrate.py`, 2026-11-03. Positives: 109 paraphrases of golden v1 questions (written
by qwen2.5:7b, 138 drafts, kept by a blind Claude Opus review: same question, same answer, literals
unchanged). Negatives: 50 reviewed contrast questions (same topic, a different answer) and all 2,346
pairs of distinct golden questions. Similarity is gte-small cosine exactly as the cache computes it
(stored question normalised, lookup as asked), with the cache's literal guard applied.

| threshold | paraphrase hit rate | wrong hits |
|---|---|---|
| 0.80 | 0.991 (108/109) | 156 / 2396 |
| 0.81 | 0.991 (108/109) | 123 / 2396 |
| 0.82 | 0.991 (108/109) | 96 / 2396 |
| 0.83 | 0.991 (108/109) | 78 / 2396 |
| 0.84 | 0.991 (108/109) | 64 / 2396 |
| 0.85 | 0.991 (108/109) | 48 / 2396 |
| 0.86 | 0.991 (108/109) | 44 / 2396 |
| 0.87 | 0.991 (108/109) | 38 / 2396 |
| 0.88 | 0.991 (108/109) | 34 / 2396 |
| 0.89 | 0.982 (107/109) | 30 / 2396 |
| 0.90 | 0.963 (105/109) | 28 / 2396 |
| 0.91 | 0.954 (104/109) | 24 / 2396 |
| 0.92 (current) | 0.936 (102/109) | 23 / 2396 |
| 0.93 | 0.927 (101/109) | 20 / 2396 |
| 0.94 | 0.917 (100/109) | 16 / 2396 |
| 0.95 | 0.881 (96/109) | 13 / 2396 |
| 0.96 | 0.817 (89/109) | 8 / 2396 |
| 0.97 | 0.670 (73/109) | 4 / 2396 |
| 0.98 | 0.486 (53/109) | 3 / 2396 |

No threshold in the range avoids every wrong hit.


Every wrong hit is a contrast pair; no two distinct golden questions ever cross 0.80. At 0.97 the
four left are: enable vs disable a scheduler setting, gang vs non-gang PodGroup, what a field does
vs which field to use, diagnose vs resolve an `ImagePullBackOff`. One word flips the answer and
barely moves the embedding, so the semantic tier is off (`cache.semantic_enabled: false`,
decision 48); the exact tier stays.
