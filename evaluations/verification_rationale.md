# Verdict prediction with short rationales

Completed on 2026-09-17T02:04:16.310180+00:00.

## Change

The base prompt now asks for a brief rationale in one English sentence of at most 40 words, identifying relevant evidence or missing information and any inference needed for an inferential label. The original Japanese task and six label definitions, user template, model, and sampling settings are unchanged. No few-shot examples were added. The JSON schema requests `label` first and `rationale` second; both fields are required. Validation rejects empty rationales but does not enforce or truncate to the word budget.

This evaluates the combined prompt and output-schema change. Verdict labels are scored; rationale factuality, faithfulness, and sentence-count compliance are not graded. Word counts below use whitespace splitting as a descriptive length check.

## Evaluation protocol

- Model: `gpt-5.4-2026-03-05`, using the same configured Factchecker endpoint and provider-default sampling as the baseline.
- Three independent runs over the same 5,682 AIO test pairs (17,046 scheduled classifications), with at most 32 concurrent requests.
- The original 6,824-pair test file has 1,142 original-input evidence pairs excluded by the same question/evidence text comparison as before.
- Accuracy is computed per claim-evidence pair. Precision, recall, and F1 are macro averages over the fixed six labels, with zero-division scores set to zero. Run means and population standard deviations are reported.
- API failures and invalid outputs count as incorrect predictions; every gold ID must have exactly one prediction or explicit error record.
- The cohort, source revision, gold-data hash, model, sampling, aggregation, and error policy match the saved baseline protocol.
- Source revision: `cecf2a1d2c05a71dbc8460a1476ba051d18db807`.
- Prompt SHA-256: `17189a1c43bb10824405702c01a1a751066c5cb10ecf689b84ae00ca5a105f9d`.
- Gold SHA-256: `a9218c1b75ac5ae5794ecbc46d2d74cbcce1ae2fd81bbc225b3799c3e994b49b`.

The evaluation follows the three-run, six-class AIO method in [Masano et al. (ACL SRW 2026)](https://aclanthology.org/2026.acl-srw.99/). The comparison below uses our earlier GPT-5.4 run without rationales, not the paper’s GPT-4o results. It is a descriptive comparison; no significance test or claim of statistical significance is made.

## Aggregate results

| Setting | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Base without rationale | 0.6279 ± 0.0016 | 0.3990 ± 0.0019 | 0.4857 ± 0.0031 | 0.4000 ± 0.0009 |
| Base with short rationale | 0.6256 ± 0.0021 | 0.4019 ± 0.0027 | 0.4968 ± 0.0035 | 0.4028 ± 0.0025 |
| Change | -0.0023 | +0.0030 | +0.0111 | +0.0028 |

| Run | Accuracy | Macro precision | Macro recall | Macro F1 | Failed predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6269 | 0.4006 | 0.4998 | 0.4026 | 7 |
| 2 | 0.6227 | 0.3994 | 0.4919 | 0.3998 | 6 |
| 3 | 0.6272 | 0.4057 | 0.4987 | 0.4060 | 6 |

## Per-label results

| Label | Gold count | Precision | Recall | F1 | Base F1 | Change in F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fully supported | 1874 | 0.7066 | 0.8367 | 0.7662 | 0.7630 | +0.0032 |
| Inferentially supported | 591 | 0.2074 | 0.4055 | 0.2745 | 0.2618 | +0.0127 |
| Partially supported | 264 | 0.2462 | 0.1856 | 0.2116 | 0.1895 | +0.0221 |
| Fully refuted | 87 | 0.2300 | 0.7471 | 0.3517 | 0.3724 | -0.0208 |
| Inferentially refuted | 33 | 0.0592 | 0.2323 | 0.0943 | 0.0857 | +0.0087 |
| Not enough information | 2833 | 0.9619 | 0.5737 | 0.7187 | 0.7277 | -0.0090 |

## Rationale length and failures

| Run | Successful outputs | Mean words | Maximum words | Outputs over 40 words |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5675 | 23.7 | 38 | 0 |
| 2 | 5676 | 23.7 | 38 | 0 |
| 3 | 5676 | 23.7 | 39 | 0 |

- Run 1 failed IDs: 2568 (ValueError), 2581 (content_filter, HTTP 400), 3034 (ValueError), 3037 (ValueError), 5089 (content_filter, HTTP 400), 7454 (content_filter, HTTP 400), 7459 (content_filter, HTTP 400).
- Run 2 failed IDs: 2568 (ValueError), 2581 (content_filter, HTTP 400), 3037 (ValueError), 5089 (content_filter, HTTP 400), 7454 (content_filter, HTTP 400), 7459 (content_filter, HTTP 400).
- Run 3 failed IDs: 2567 (ValueError), 2581 (content_filter, HTTP 400), 3037 (ValueError), 5089 (content_filter, HTTP 400), 7454 (content_filter, HTTP 400), 7459 (content_filter, HTTP 400).

Failures were retained as incorrect predictions. Separate diagnostic requests were used only to investigate validation failures and never replaced the scored predictions. SDK-default retries for transient failures remain enabled. No prompt tuning was performed on these test results.

In three separate diagnostic requests, ID 3037 returned `finish_reason="content_filter"`, while IDs 2568 and 3034 completed successfully. The original failed outputs were not retained, so these diagnostics do not establish the cause of every recorded `ValueError`. Their outcomes are included in the JSON report.

## Validation and reproduction

The 99 relevant application, endpoint, prompt, UI, pipeline, and evaluation tests passed, as did Ruff lint and format checks. The unrelated pre-existing chat-history test was outside this targeted test run.

All aggregate metrics, per-class precision/recall/F1/support, and confusion matrices match scikit-learn for each run (absolute tolerance 1e-12). Re-scoring the saved predictions without API requests reproduced all scores. Every successful output has a nonempty rationale. The existing UI renders these rationales with the verdicts and still handles historical label-only results.

```bash
uv run --locked --group evaluation python scripts/evaluate_verification.py \
  --dataset-repo /path/to/llm-jp-evidence-verification-dataset \
  --model gpt-5.4-2026-03-05 \
  --max-concurrency 32 \
  --output result/verification-rationale
```

Append `--score-only` to re-score a completed run without API requests. The prompt is editable in [prompts/verification.yaml](../prompts/verification.yaml). The [JSON report](verification_rationale.json) contains the protocol, metrics, baseline comparison, errors, and length statistics. Per-pair labels and rationales, the frozen YAML prompt, gold data, and per-run scores are preserved under `result/verification-rationale/` (ignored by Git). The earlier [base scores](verification_base.json) and raw predictions are retained unchanged as the comparison data.

The output schema uses the required-field contract described in the [official OpenAI Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs).
