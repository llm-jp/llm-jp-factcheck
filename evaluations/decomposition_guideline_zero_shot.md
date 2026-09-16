# Guideline decomposition evaluation

Completed on 2026-09-16T12:01:26.612874+00:00.

## Configuration

- Model: `gpt-5.4-2026-03-05` through the configured Factchecker connection.
- Dataset: the original AIO test split, 210 generated texts and 1,169 gold claims.
- Five independent live prediction runs, 1,050 completed predictions; concurrency limited to eight.
- Two requests in the final run stalled for more than four minutes. The process was restarted from saved predictions and only those two missing documents were requested again. This count describes completed predictions, not total HTTP attempts.
- Guideline with zero additional few-shot demonstrations. The source guideline explanations and their embedded examples are retained.
- Generated text only: no question, conversation context, gold claim, check-worthiness filtering, or evidence retrieval.
- Structured Outputs with the application claim schema; provider-default sampling parameters.
- Reference repository revision: `e43e3898959860caa0feaf1708e820ecec330584`.
- Prompt SHA-256: `6934204c9b1d45aaf8d88cdbd84b9b15870f42816e1bd4962fda6fbfcc627eed`.
- Gold SHA-256: `a9d9ca91f3df01262d9625837e8c505d202f667741f8513bddb8ea0e22d3ab50`.

## Results

Scores range from 0 to 1. Each run macro-averages document-level precision, recall, and F1 separately. The table reports the mean and population standard deviation across the five runs.

| Matching | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Exact | 0.2071 ± 0.0086 | 0.1828 ± 0.0076 | 0.1916 ± 0.0079 |
| Fuzzy, content words | 0.5368 ± 0.0113 | 0.4564 ± 0.0127 | 0.4857 ± 0.0120 |

| Run | Exact P | Exact R | Exact F1 | Fuzzy P | Fuzzy R | Fuzzy F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2022 | 0.1824 | 0.1895 | 0.5470 | 0.4709 | 0.4987 |
| 2 | 0.2168 | 0.1904 | 0.2001 | 0.5380 | 0.4576 | 0.4873 |
| 3 | 0.2024 | 0.1783 | 0.1869 | 0.5434 | 0.4624 | 0.4918 |
| 4 | 0.2176 | 0.1916 | 0.2011 | 0.5405 | 0.4582 | 0.4877 |
| 5 | 0.1964 | 0.1713 | 0.1804 | 0.5151 | 0.4327 | 0.4632 |

## Metric validation

The saved predictions were also re-scored when the evaluator adopted the reference Hungarian tie resolution during the eight-shot comparison. All zero-shot scores remain unchanged; updated implementation hashes are recorded in the JSON report.

The evaluator follows [Masano et al. (LREC 2026), §6.2](https://aclanthology.org/2026.lrec-1.186/). Exact matches require identical strings. Fuzzy matches use Jaccard similarity over MeCab/UniDic-lite content-word sets (nouns, verbs, adjectives, adverbs). Maximum-weight one-to-one assignment is computed before accepting pairs at similarity ≥ 0.8, as in the reference code. Token forms use UniDic field 10 with surface-form fallback. Gold whitespace and punctuation are preserved.

All 2,100 document/metric combinations (210 documents × five runs × two metrics) were compared against the original evaluator's Hungarian implementation at the pinned revision. Precision, recall, and F1 agree in every case. Unit tests also cover duplicate predictions, empty outputs, the threshold boundary, assignment before thresholding, macro averaging, missing predictions, and altered snapshots.

## Differences from the published experiment

- The paper used GPT-4o; this evaluation uses the configured GPT-5.4 model and provider-default sampling. It is not an exact reproduction of the paper's published scores.
- The source prompt is `scripts/prompts/decomposition/manual_rule+example+expl/manual_rule+example+expl.txt`. All 13 guideline sections are retained. Its final standalone input/output demonstration is removed, and the output instruction is adapted from a plain array to `{"claims": [...]}`. No additional few-shot demonstrations are supplied.
- The application adds reference-resolution and separate check-worthiness instructions. No external context is supplied during evaluation.
- The pinned branch contains the AIO test split but no CBA test split. CBA is therefore not evaluated; no replacement split is invented.

## Reproduce or inspect

```bash
uv run --locked --group evaluation python scripts/evaluate_decomposition.py \
  --dataset-repo /path/to/llm-jp-evidence-verification-dataset \
  --prompt prompts/decomposition.json \
  --source-manifest evaluations/decomposition_source.json \
  --output result/decomposition-guideline-zero-shot
```

Append `--score-only` to recompute metrics from completed saved predictions without making API calls. Reusing the same settings resumes missing predictions; use a new output directory for a different prompt or model. The complete protocol, run scores, and aggregate scores are in [decomposition_guideline_zero_shot.json](decomposition_guideline_zero_shot.json). Raw predictions, gold text, assignments, and the prompt snapshot are stored locally under `result/decomposition-guideline-zero-shot/` (ignored by Git).

Validation: 131 of 132 application and evaluation tests passed. The existing `test_chat.ChatTests.test_preserves_full_history_without_metadata_or_input_mutation` still expects a system message that the current chat implementation no longer sends. This unrelated mismatch was left unchanged. Ruff lint/format, lockfile validation, and the saved-prediction re-scoring check passed.
