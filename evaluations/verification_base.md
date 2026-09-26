# Base verdict prediction evaluation

Completed on 2026-09-17T01:31:50.064271+00:00.

## Configuration

- Model: `gpt-5.4-2026-03-05`, using the configured Factchecker connection and provider-default sampling.
- Three independent runs on 5,682 AIO test claim-evidence pairs (17,046 scheduled classifications), with at most 32 concurrent requests.
- Base instructions only: task, claim definition, and six label definitions. No few-shot demonstrations or reasoning/rationale instructions.
- Labels: fully supported, inferentially supported, partially supported, fully refuted, inferentially refuted, and not enough information.
- Source: `feat/verdict_prediction` at `cecf2a1d2c05a71dbc8460a1476ba051d18db807`.
- Prompt: `scripts/prompts/verdict_prediction/6class_base_1224.txt`.
- Prompt SHA-256: `a2e7888b4799e7756535c682d373f016eeea079d01388b686a86eb8f455975fb`.
- Gold SHA-256: `a9218c1b75ac5ae5794ecbc46d2d74cbcce1ae2fd81bbc225b3799c3e994b49b`.

## Evaluation method and data

The protocol follows [Masano et al. (ACL SRW 2026), §5.1](https://aclanthology.org/2026.acl-srw.99/): three AIO test runs, pair accuracy, and macro precision/recall/F1 over the fixed six labels. Macro F1 is the mean of per-class F1 scores, not the harmonic mean of macro precision and recall. Undefined class scores are zero. The table reports the mean and population standard deviation across runs.

The original test file contains 6,824 pairs. In line with the paper’s exclusion of original-input evidence, 1,142 pairs whose evidence equals the corresponding input question after trimming outer whitespace are removed. All 5,682 remaining pairs are retained. Gold refutation labels using “矛盾” are normalized to the corresponding “否定” category. Gold verdicts are never sent to the model, and no new decomposition or retrieval is performed.

## Aggregate results

| Setting | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Current GPT-5.4 base | 0.6279 ± 0.0016 | 0.3990 ± 0.0019 | 0.4857 ± 0.0031 | 0.4000 ± 0.0009 |
| Archived GPT-4o base, re-scored | 0.6691 | 0.4084 | 0.4748 | 0.4175 |
| Published GPT-4o base (Table 2) | 0.669 | 0.408 | 0.475 | 0.418 |

| Run | Accuracy | Macro precision | Macro recall | Macro F1 | Failed predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6297 | 0.4004 | 0.4819 | 0.3989 | 4 |
| 2 | 0.6283 | 0.3962 | 0.4894 | 0.4000 | 4 |
| 3 | 0.6258 | 0.4003 | 0.4859 | 0.4011 | 4 |

## Per-label results

| Label | Gold count | Precision | Recall | F1 | Archived GPT-4o F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fully supported | 1874 | 0.7024 | 0.8351 | 0.7630 | 0.7523 |
| Inferentially supported | 591 | 0.1986 | 0.3841 | 0.2618 | 0.1604 |
| Partially supported | 264 | 0.2323 | 0.1604 | 0.1895 | 0.1935 |
| Fully refuted | 87 | 0.2493 | 0.7356 | 0.3724 | 0.4404 |
| Inferentially refuted | 33 | 0.0537 | 0.2121 | 0.0857 | 0.1418 |
| Not enough information | 2833 | 0.9574 | 0.5869 | 0.7277 | 0.8167 |

## Errors and comparability

- API errors remain in the evaluation denominator as incorrect predictions, with an explicit error column in the confusion matrix. Each run had four failures (IDs 2581, 5089, 7454, and 7459), all identified as Azure content-filter responses (HTTP 400) in separate diagnostics. The failed IDs and error types are preserved in the JSON report. Separate diagnostic requests were used only to identify provider errors and never replaced the scored predictions.
- The reference branch’s three saved base runs contain 5,681 pairs. Pair ID 7006 is missing; evidence differs from the current test data for IDs 5772, 11775, and 12049. Recorded gold labels otherwise agree. Archived runs are re-scored on their own recorded inputs and gold labels, while the current evaluation retains all valid retrieved-evidence pairs.
- The paper and archived runs use GPT-4o; the current evaluation uses GPT-5.4, provider-default sampling, and Structured Outputs. The paper’s labels and Japanese definitions are preserved, but its plain-label output is adapted to a one-field JSON object. Literal input placeholders and the reference runner’s generic three-way user wrapper are replaced with the actual claim and evidence in the user message.
- These model, format, sampling, and small data differences mean this is a base evaluation using the paper’s method, not a controlled model comparison or an exact replication. No prompt tuning was performed on the test outcomes.

## Validation and reproduction

The aggregate metrics, each class’s precision/recall/F1/support, and confusion matrices agree with scikit-learn’s reference metric functions for every run (absolute tolerance 1e-12). Re-scoring saved predictions without API requests reproduced all scores. Unit tests cover label normalization, fixed-label macro averaging, error accounting, exact dataset coverage, input-evidence exclusion, checkpoint recovery, Japanese-to-English output mapping, and rendering results without a rationale.

This evaluation used the six-label base prompt without rationales. The application has since added short rationales to [prompts/verification.yaml](../prompts/verification.yaml). The original prompt snapshot and protocol remain under `result/verification-base/`; these scores describe that earlier configuration.

```bash
uv run --locked --group evaluation python scripts/evaluate_verification.py \
  --dataset-repo /path/to/llm-jp-evidence-verification-dataset \
  --model gpt-5.4-2026-03-05 \
  --max-concurrency 32 \
  --output result/verification-base
```

This is the historical command, using the label-only implementation at application commit `9a7f4589c81970d946c88700cadf211324c24d37`. The current implementation uses a different prompt and output schema, so reproducing or re-scoring this baseline requires that earlier implementation. With it, append `--score-only` to compute metrics from completed saved predictions without model requests. The protocol, aggregate and per-label metrics, reference scores, and error records are in [verification_base.json](verification_base.json). Raw predictions, input snapshots, excluded IDs, per-run confusion matrices, and error diagnostics are under `result/verification-base/` (ignored by Git).

Validation: 141 of 142 tests passed. The pre-existing chat-history test expects a system message that the current chat implementation no longer sends. Ruff lint/format, lockfile validation, and the relevant verification/UI/evaluation tests passed.
