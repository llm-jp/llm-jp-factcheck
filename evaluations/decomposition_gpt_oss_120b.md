# GPT-OSS-120B guideline + 8-shot decomposition evaluation

Completed on 2026-09-17T07:46:00.287909+00:00.

## Setup

- Model: `gpt-oss-120b`, using the configured OpenAI-compatible Factchecker endpoint.
- Guideline + 8-shot; the same instructions and eight development examples as the earlier GPT-5.4 evaluation.
- AIO test: 210 generated texts and 1,169 gold claims, evaluated in five runs (1,050 completed document predictions).
- Generated text is the only model input; no question, conversation context, gold claims, check-worthiness filtering, or evidence retrieval is supplied.
- Exact string matching and content-word Jaccard matching at similarity ≥ 0.8 after maximum-weight one-to-one assignment, preserving the reference tie resolution.
- MeCab with UniDic-lite extracts nouns, verbs, adjectives, and adverbs, using feature index 10 with surface fallback. Precision, recall, and F1 are macro-averaged separately across documents, then averaged across runs; variation is population standard deviation.
- At most eight concurrent decomposition requests, a 120-second request timeout, and the SDK default of two retries for retryable failures. Verification was evaluated concurrently in a separate process.
- Reference revision: `e43e3898959860caa0feaf1708e820ecec330584`.
- Prompt SHA-256: `dea1d5cd4199313edc64bbd9fb48c59492f44c4a7d471c2853c5a13c9ebfcbda`.
- Gold SHA-256: `a9d9ca91f3df01262d9625837e8c505d202f667741f8513bddb8ea0e22d3ab50`.

The earlier prompt snapshot was JSON, while the current snapshot is YAML. Their parsed system/user templates are identical. The decomposition implementation differs only in its default prompt path; both evaluations explicitly supply their eight-shot snapshot. The evaluator, schema, gold data, and dependency versions are unchanged. No prompt tuning was performed on this test set.

The earlier GPT-5.4 run used Azure, while this run uses an OpenAI-compatible deployment. Neither run overrides temperature, top_p, seed, or reasoning effort. Provider defaults and server-side model revision, quantization, and inference settings were not independently controlled. The results compare configured deployments and do not isolate model weights from serving settings.

## Five-run results

Values are mean ± population standard deviation on a 0–1 scale.

| Model | Matching | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| GPT-5.4 | Exact | 0.2660 ± 0.0085 | 0.2364 ± 0.0086 | 0.2468 ± 0.0085 |
| GPT-5.4 | Fuzzy, content words | 0.6062 ± 0.0094 | 0.5289 ± 0.0129 | 0.5567 ± 0.0117 |
| GPT-OSS-120B | Exact | 0.1779 ± 0.0079 | 0.1600 ± 0.0064 | 0.1660 ± 0.0071 |
| GPT-OSS-120B | Fuzzy, content words | 0.5428 ± 0.0101 | 0.4752 ± 0.0095 | 0.4984 ± 0.0099 |

GPT-OSS minus GPT-5.4:

| Matching | Δ Precision | Δ Recall | Δ F1 |
| --- | ---: | ---: | ---: |
| Exact | -0.0881 | -0.0764 | -0.0808 |
| Fuzzy, content words | -0.0634 | -0.0537 | -0.0583 |

These are descriptive differences; no statistical significance test was performed.

| Run | Exact P | Exact R | Exact F1 | Fuzzy P | Fuzzy R | Fuzzy F1 | Predicted claims |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1898 | 0.1707 | 0.1774 | 0.5552 | 0.4881 | 0.5112 | 975 |
| 2 | 0.1668 | 0.1509 | 0.1561 | 0.5346 | 0.4617 | 0.4874 | 967 |
| 3 | 0.1820 | 0.1617 | 0.1687 | 0.5544 | 0.4831 | 0.5084 | 959 |
| 4 | 0.1727 | 0.1576 | 0.1625 | 0.5314 | 0.4693 | 0.4884 | 966 |
| 5 | 0.1779 | 0.1590 | 0.1654 | 0.5382 | 0.4739 | 0.4965 | 965 |

## Validation and reproduction

All 2,100 document/metric combinations match the pinned original reference evaluator exactly. Re-scoring saved predictions with API access blocked reproduces every run. All 210 document IDs are present exactly once in each of the five runs; no missing or failed document is silently dropped. The runner stops on an unsuccessful request rather than scoring an incomplete dataset. A separate two-document smoke test is excluded from the reported scores.

Configure the Factchecker connection for the GPT-OSS deployment, then run:

```bash
uv run --locked --group evaluation python scripts/evaluate_decomposition.py \
  --dataset-repo /path/to/llm-jp-evidence-verification-dataset \
  --prompt prompts/decomposition_8shot.yaml \
  --source-manifest evaluations/decomposition_8shot_source.json \
  --model gpt-oss-120b \
  --max-concurrency 8 \
  --output result/decomposition-gpt-oss-120b
```

Append `--score-only` to recompute metrics without model calls. The [JSON report](decomposition_gpt_oss_120b.json) contains the protocol, metrics, reference parity audit, prediction counts, and comparison. Raw predictions, matching assignments, gold text, and the frozen prompt are saved under `result/decomposition-gpt-oss-120b/` (ignored by Git). The [GPT-5.4 eight-shot evaluation](decomposition_guideline_8shot.md) remains unchanged.

This follows the [LREC paper’s evaluation method](https://aclanthology.org/2026.lrec-1.186/). Only AIO is evaluated because the pinned experiment revision does not contain the CBA test split. The current prompt and Structured Outputs adaptation are documented in [the example source manifest](decomposition_8shot_source.json).
