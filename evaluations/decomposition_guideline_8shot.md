# Guideline + 8-shot decomposition evaluation

Completed on 2026-09-16T12:15:25.261577+00:00.

## Setup

- Model: `gpt-5.4-2026-03-05`; the same Factchecker connection and provider-default sampling as the zero-shot evaluation.
- AIO test: 210 generated texts and 1,169 gold claims; five independent runs (1,050 completed predictions).
- Metrics: exact string matching and content-word Jaccard matching at similarity ≥ 0.8 after maximum-weight one-to-one assignment. P/R/F1 are macro-averaged over documents, then averaged across runs.
- Same gold data, evaluator, Structured Outputs schema, and generated-text-only input as the zero-shot baseline.
- Reference revision: `e43e3898959860caa0feaf1708e820ecec330584`.
- Prompt SHA-256: `ca29eba0d7381fa16130d3e5bf553702fd0ac6f0af8f5fabd22e5b34ca46e7ce`.
- Gold SHA-256: `a9d9ca91f3df01262d9625837e8c505d202f667741f8513bddb8ea0e22d3ab50`.

## Eight-shot prompt

This is **guideline + 8-shot**, using the existing zero-shot guideline and eight additional input/output demonstrations. The app now defaults to the eight-shot prompt in [prompts/decomposition_8shot.json](../prompts/decomposition_8shot.json).

The reference file named `manual_rule+example+expl_8shot.txt` contains only six demonstrations. The complete `scripts/prompts/decomposition/fewshot/8shot.txt` contains eight, and its first six match those in the guideline file. All eight are imported from that complete file, preserving their text and claim strings while adapting output arrays to the application’s `{"claims": [...]}` schema. The guideline rules, application instructions, and user template are unchanged from the zero-shot evaluation.

All eight example inputs were checked against the development set and are absent from the test set. Their development IDs are: `AIO02-0044`, `AIO02-0038`, `AIO02-0081`, `AIO02-0092`, `AIO02-0053`, `AIO02-0060`, `AIO02-0069`, `AIO02-0074`.

## Five-run results

Values are mean ± population standard deviation on a 0–1 scale.

| Setting | Matching | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| Guideline + 0-shot | Exact | 0.2071 ± 0.0086 | 0.1828 ± 0.0076 | 0.1916 ± 0.0079 |
| Guideline + 0-shot | Fuzzy, content words | 0.5368 ± 0.0113 | 0.4564 ± 0.0127 | 0.4857 ± 0.0120 |
| Guideline + 8-shot | Exact | 0.2660 ± 0.0085 | 0.2364 ± 0.0086 | 0.2468 ± 0.0085 |
| Guideline + 8-shot | Fuzzy, content words | 0.6062 ± 0.0094 | 0.5289 ± 0.0129 | 0.5567 ± 0.0117 |

Eight-shot minus zero-shot:

| Matching | Δ Precision | Δ Recall | Δ F1 |
| --- | ---: | ---: | ---: |
| Exact | +0.0589 | +0.0536 | +0.0552 |
| Fuzzy, content words | +0.0694 | +0.0726 | +0.0710 |

These differences describe this model and dataset; no statistical significance test was performed.

| Run | Exact P | Exact R | Exact F1 | Fuzzy P | Fuzzy R | Fuzzy F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2660 | 0.2345 | 0.2457 | 0.5975 | 0.5209 | 0.5487 |
| 2 | 0.2729 | 0.2399 | 0.2516 | 0.6069 | 0.5232 | 0.5535 |
| 3 | 0.2771 | 0.2502 | 0.2595 | 0.6236 | 0.5543 | 0.5795 |
| 4 | 0.2609 | 0.2332 | 0.2429 | 0.5988 | 0.5194 | 0.5477 |
| 5 | 0.2531 | 0.2240 | 0.2343 | 0.6041 | 0.5268 | 0.5540 |

## Validation and reproducibility

Initial validation found two tied-assignment cases (AIO02-0362 in runs 3 and 5) where SciPy and the reference Hungarian code chose different equally optimal assignments. The evaluator now preserves the reference tie resolution. Both configurations were re-scored from saved predictions without new model calls; the zero-shot scores are unchanged. The corrected eight-shot results above use the reference assignments.

All 2,100 document/metric combinations were compared against the original reference evaluator, with identical precision, recall, and F1 throughout. Re-scoring saved predictions without API calls also reproduced every run exactly. The prompt-rendering and evaluation CLI tests passed, including custom source-manifest recording. Ruff and the lockfile checks passed. The full test suite passed 134 of 135 tests; the pre-existing chat-history test still expects a system message that the chat implementation no longer sends.

```bash
uv run --locked --group evaluation python scripts/evaluate_decomposition.py \
  --dataset-repo /path/to/llm-jp-evidence-verification-dataset \
  --prompt prompts/decomposition_8shot.json \
  --source-manifest evaluations/decomposition_8shot_source.json \
  --model gpt-5.4-2026-03-05 \
  --output result/decomposition-guideline-8shot
```

Append `--score-only` to recompute metrics from saved predictions. The complete protocol and numerical results are in [decomposition_guideline_8shot.json](decomposition_guideline_8shot.json), and the example provenance is in [decomposition_8shot_source.json](decomposition_8shot_source.json). Raw predictions, assignments, gold text, and the prompt snapshot are saved locally under `result/decomposition-guideline-8shot/` (ignored by Git).

As in the zero-shot evaluation, this follows the [paper’s evaluation method](https://aclanthology.org/2026.lrec-1.186/) but uses GPT-5.4 and Structured Outputs, so it does not reproduce the paper’s GPT-4o scores. Only AIO is evaluated because the pinned experiment revision does not contain the CBA test split.
