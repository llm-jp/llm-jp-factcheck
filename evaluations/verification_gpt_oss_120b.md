# GPT-OSS-120B verdict prediction evaluation

Completed on 2026-09-17T07:38:15.926731+00:00.

## Evaluation protocol

This evaluates `gpt-oss-120b` through the configured OpenAI-compatible Factchecker endpoint, using the same base prompt with a short rationale as the earlier `gpt-5.4-2026-03-05` evaluation. The YAML prompt, JSON Schema, cohort, evaluator, and OpenAI SDK version match the saved GPT-5.4 protocol. No prompt or inference-setting tuning was performed on the test results.

- Three runs over 5,682 AIO test claim-evidence pairs (17,046 scheduled classifications).
- Exclude 1,142 original-input evidence pairs from the 6,824-pair test file by comparing evidence and the corresponding question after trimming outer whitespace.
- Follow the three-run, six-class AIO evaluation in [Masano et al. (ACL SRW 2026)](https://aclanthology.org/2026.acl-srw.99/): pair accuracy and macro precision, recall, and F1 over the fixed six labels, with zero division set to zero. Report run means and population standard deviations.
- Use at most 32 concurrent requests, a 60-second request timeout, and the SDK default of two retries for retryable failures. Failed predictions are retained as incorrect; no pair is silently excluded and no explicit error-retry pass is used.
- Request a Japanese label followed by a short English rationale in one sentence of at most 40 words. Only verdict labels are scored. Rationale factuality, faithfulness, and sentence-count compliance are not graded; whitespace word counts are descriptive.
- Decomposition, check-worthiness, and retrieval are not part of this evaluation; the dataset supplies the claims and evidence.
- Source revision: `cecf2a1d2c05a71dbc8460a1476ba051d18db807`.
- Prompt SHA-256: `17189a1c43bb10824405702c01a1a751066c5cb10ecf689b84ae00ca5a105f9d`.
- Gold SHA-256: `a9218c1b75ac5ae5794ecbc46d2d74cbcce1ae2fd81bbc225b3799c3e994b49b`.

The models use different serving environments: the prior GPT-5.4 evaluation used Azure, and this evaluation uses an OpenAI-compatible endpoint. Neither run overrides temperature, top_p, seed, or reasoning effort. Provider defaults can differ, and the server-side model revision, quantization, and inference configuration were not independently verified. The comparison therefore measures the configured deployments, without isolating model weights from serving settings. These are descriptive differences, with no claim of statistical significance.

## Aggregate results

| Model (short rationale) | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.4 | 0.6256 ± 0.0021 | 0.4019 ± 0.0027 | 0.4968 ± 0.0035 | 0.4028 ± 0.0025 |
| GPT-OSS-120B | 0.6537 ± 0.0024 | 0.4069 ± 0.0046 | 0.4803 ± 0.0092 | 0.4169 ± 0.0058 |
| Change (GPT-OSS minus GPT-5.4) | +0.0281 | +0.0050 | -0.0165 | +0.0141 |

| Run | Accuracy | Macro precision | Macro recall | Macro F1 | Failed predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6572 | 0.4104 | 0.4831 | 0.4206 | 1 |
| 2 | 0.6521 | 0.4099 | 0.4900 | 0.4213 | 0 |
| 3 | 0.6519 | 0.4005 | 0.4679 | 0.4087 | 0 |

## Comparison with archived GPT-4o results

To compare all three models on identical inputs and gold labels, restrict each run to the same 5,678 pairs. Exclude IDs [5772, 7006, 11775, 12049]: the archived GPT-4o results omit ID 7006 and change the evidence for the other three IDs. All scores below are three-run means. GPT-4o uses the archived label-only base setting, while GPT-5.4 and GPT-OSS use short rationales. This output-format difference remains a limitation.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| gpt-4o (archived base) | 0.6691 | 0.4084 | 0.4748 | 0.4175 |
| gpt-5.4-2026-03-05 | 0.6255 | 0.4019 | 0.4969 | 0.4028 |
| gpt-oss-120b | 0.6536 | 0.4069 | 0.4804 | 0.4169 |

All nine scored runs in this matched comparison agree with scikit-learn within absolute tolerance 1e-12. These scores use a slightly smaller cohort than the primary evaluation above.

## Per-label results

| Label | Gold count | Precision | Recall | F1 | GPT-5.4 F1 | Change in F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fully supported | 1874 | 0.6754 | 0.8903 | 0.7681 | 0.7662 | +0.0019 |
| Inferentially supported | 591 | 0.2032 | 0.2814 | 0.2360 | 0.2745 | -0.0385 |
| Partially supported | 264 | 0.2757 | 0.2753 | 0.2755 | 0.2116 | +0.0639 |
| Fully refuted | 87 | 0.2838 | 0.6782 | 0.4002 | 0.3517 | +0.0485 |
| Inferentially refuted | 33 | 0.0499 | 0.1414 | 0.0737 | 0.0943 | -0.0206 |
| Not enough information | 2833 | 0.9535 | 0.6154 | 0.7480 | 0.7187 | +0.0292 |

## Rationale length and failures

| Run | Successful outputs | Mean words | Maximum words | Outputs over 40 words |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5681 | 23.5 | 40 | 0 |
| 2 | 5682 | 23.5 | 41 | 2 |
| 3 | 5682 | 23.6 | 41 | 2 |

- Run 1 failed IDs: 6326 (ValueError).
- Run 2 failed IDs: none.
- Run 3 failed IDs: none.

The word budget is a prompt instruction, not a hard validation constraint. Nonempty rationales longer than 40 words remain valid predictions, following the unchanged evaluation implementation.

## Validation and reproduction

Every gold ID has exactly one prediction or explicit failure record in each run. Aggregate metrics, per-class precision/recall/F1/support, and confusion matrices match scikit-learn to absolute tolerance 1e-12. Re-scoring the saved predictions with API access blocked reproduced all scores.

Configure the Factchecker connection for the GPT-OSS deployment, then run:

```bash
uv run --locked --group evaluation python scripts/evaluate_verification.py \
  --dataset-repo /path/to/llm-jp-evidence-verification-dataset \
  --model gpt-oss-120b \
  --max-concurrency 32 \
  --output result/verification-gpt-oss-120b
```

Append `--score-only` to re-score a completed evaluation without model requests. The model also defaults to `FACTCHECKER_MODEL` when `--model` is omitted. A separate three-pair smoke test verified output compatibility before the full evaluation and is not included in these scores.

The [JSON report](verification_gpt_oss_120b.json) contains the protocol, per-run metrics and confusion matrices, comparison, errors, and rationale lengths. Predictions, rationales, the frozen YAML prompt, gold data, and per-run scores are preserved in `result/verification-gpt-oss-120b/` (ignored by Git). The [GPT-5.4 rationale report](verification_rationale.md) and its raw predictions remain unchanged.
