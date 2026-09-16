# LLM-jp Factcheck

A Streamlit application for chatting with an LLM and checking its responses using evidence retrieved from its training data. Each response is decomposed into claims, check-worthy claims are selected, and relevant passages are retrieved from the indexed training corpus. Each claim–evidence pair receives a verdict and rationale. Users can inspect the evidence passages, their sources, and the associated training steps as results arrive.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run all commands from the project root. The project uses Python 3.12, selected by `.python-version`. Dependencies are declared in `pyproject.toml` and resolved in `uv.lock`.

```bash
uv sync --locked
```

### Model connections

Create `.env` in the project root using [`.env.example`](.env.example) as a template. Configure chat generation with `CHATBOT_` variables and decomposition, check-worthiness, and verification with `FACTCHECKER_` variables. Each role can independently use an OpenAI-compatible endpoint or Azure OpenAI.

| Setting | Chat generation | Decomposition, check-worthiness, and verification | Required |
| --- | --- | --- | --- |
| API type | `CHATBOT_API_TYPE` | `FACTCHECKER_API_TYPE` | `openai` or `azure` |
| Endpoint | `CHATBOT_ENDPOINT` | `FACTCHECKER_ENDPOINT` | Yes |
| API key | `CHATBOT_API_KEY` | `FACTCHECKER_API_KEY` | Yes; use a dummy value for a local server without authentication |
| API version | `CHATBOT_API_VERSION` | `FACTCHECKER_API_VERSION` | Only when API type is `azure` |
| Model | `CHATBOT_MODEL` | `FACTCHECKER_MODEL` | Optional; see model selection below |

With `API_TYPE=openai`, supply the complete API base URL, including its path: `https://api.openai.com/v1` for OpenAI, `http://localhost:8000/v1` for a compatible local server, or `https://your-resource.openai.azure.com/openai/v1` for an [Azure v1 endpoint](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle). The role's `API_VERSION` is ignored in this mode.

With `API_TYPE=azure`, supply the Azure resource root, such as `https://your-resource.openai.azure.com`, and an API version supported by that resource. Model arguments are Azure deployment names in this mode. These modes use the SDK's `OpenAI` and `AzureOpenAI` clients, respectively; see the [official Python SDK reference](https://developers.openai.com/api/reference/python).

For example, use a local chatbot and an Azure fact-checker:

```dotenv
CHATBOT_API_TYPE="openai"
CHATBOT_ENDPOINT="http://localhost:8000/v1"
CHATBOT_API_KEY="unused"
CHATBOT_MODEL="your-local-model-id"

FACTCHECKER_API_TYPE="azure"
FACTCHECKER_ENDPOINT="https://your-resource.openai.azure.com"
FACTCHECKER_API_KEY="your-factchecker-api-key"
FACTCHECKER_API_VERSION="your-supported-api-version"
FACTCHECKER_MODEL="your-factcheck-deployment-name"
```

The chatbot endpoint must support streaming chat completions. The fact-checker endpoint, API version, and selected model must support [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) through Chat Completions with `response_format.type=json_schema` and `strict=true`. This requirement applies to OpenAI-compatible servers and Azure deployments as well. Clients are created when first used and cached until the process exits. Restart Streamlit after editing connection or model settings in `.env` or the environment.

Decomposition, check-worthiness, and verification each send a JSON Schema and read the JSON object from the assistant message content. All schema fields are required, extra fields are forbidden, and verification labels are constrained to the five supported values. Refusals, incomplete responses, invalid JSON, and invalid values are reported as errors. Unsupported endpoints are not retried with function calling or a weaker output format.

For compatibility, a role with none of its four prefixed connection variables set uses the shared legacy `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `AZURE_OPENAI_API_VERSION` settings. Setting any prefixed connection variable requires a complete configuration for that role; missing values are never filled from the shared settings. Model variables alone do not disable this legacy connection fallback. Global `OPENAI_*` variables are not used as a fallback.

### Retrieval

Retrieval requires an existing Elasticsearch index of the LLM's training data containing `token_ids` (space-separated token IDs), `dataset_name` (source), and `iteration` (training step). Source details come directly from each search result.

Configure retrieval with environment variables or `.env`:

| CLI argument | Environment variable | Default |
| --- | --- | --- |
| `--tokenizer_name` | `TOKENIZER_NAME` | `llm-jp/llm-jp-3-13b` |
| `--es_host` | `ES_HOST` | `http://10.2.73.12:9200` |
| `--es_dump_index` | `ES_DUMP_INDEX` | `llm-jp-corpus-v3` |

CLI arguments take precedence over environment variables, followed by the defaults above. Existing process environment variables take precedence over `.env`; blank or whitespace-only values use the defaults. Restart Streamlit after changing these settings.

Set `TOKENIZER_NAME` or `--tokenizer_name` to the tokenizer used to build the index. The first fact-check downloads the tokenizer from Hugging Face if it is not cached.

Retrieval follows the `v3.0` branch: encode each claim without special tokens, search the `token_ids` field with a match query, and request up to `--num_evidences` hits. Each hit is decoded into a complete evidence passage in Elasticsearch result order, with its source and training step preserved. Passages are not split or reranked locally. Empty passages are skipped.

## Run

With the connection and model settings configured in `.env`, run:

```bash
uv run --locked streamlit run src/serve.py
```

Model selection uses the following precedence:

| Role | Highest to lowest priority |
| --- | --- |
| Fact-checker | `--engine`, then `FACTCHECKER_MODEL`, then `gpt-5.4-2026-03-05` |
| Chatbot | `--chat-engine`, then `CHATBOT_MODEL`, then the resolved fact-checker model |

The app loads the project-root `.env` without overriding existing process environment variables. Blank or whitespace-only model variables count as unset. Use model IDs accepted by the endpoint in `openai` mode and deployment names in `azure` mode. Azure v1 endpoints also expect deployment names.

The default Factchecker model is [`gpt-5.4-2026-03-05`](https://developers.openai.com/api/docs/models/gpt-5.4). Override it with `FACTCHECKER_MODEL` or `--engine`. For Azure, use your deployment name, which may differ from the model ID.

Override model settings and supply retrieval options on the command line:

```bash
uv run --locked streamlit run src/serve.py -- \
  --engine your-factcheck-deployment-name \
  --chat-engine your-local-model-id \
  --es_host http://10.2.73.12:9200 \
  --es_dump_index llm-jp-corpus-v3 \
  --tokenizer_name llm-jp/llm-jp-3-13b \
  --num_evidences 3 \
  --max-concurrency 8
```

## Mock mode

After `uv sync --locked`, try the interface without API credentials, Elasticsearch, or tokenizer downloads:

```bash
uv run --locked streamlit run src/serve.py -- --mock
```

`--mock` replaces chat and the entire fact-checking pipeline with bundled, deterministic fictional fixtures. It supports multi-turn chat, streamed response fragments, stage progress, all five verdict labels, a non-check-worthy claim that is skipped, and per-response checks. The interface identifies mock mode and synthetic results. The canned content demonstrates the interface; it does not evaluate a model or real-world claims.

Artificial pauses make response generation and verification progress visible: approximately two seconds per chat response and three seconds per fact-check with the default evidence count and concurrency, excluding UI overhead. Mock check-worthiness, retrieval, and verification use the same bounded parallel execution as live requests. Adjust the delay constants in `src/mock_backend.py` to change these timings.

Mock mode does not load `.env` and ignores environment-based configuration. It also ignores model, endpoint, retrieval, and prompt-selection settings. `--num_evidences` selects up to two available synthetic evidence passages per claim, and `--max-concurrency` limits simultaneous mock tasks. Fixtures can be edited in `src/mock_backend.py`. Mock mode is off by default; omit `--mock` to use the configured live services.

## Workflow

Send messages through the chat input to continue a conversation. Select **Fact-check response** below any assistant response to check it. Only the selected response is decomposed; its preceding conversation history is supplied automatically as context for resolving references and omitted information.

Each reply displays the Chatbot model or Azure deployment name used to generate it, including during generation. The name is saved with the reply so it remains consistent if the model configuration changes. Mock responses display **Mock model**.

Results remain associated with their response as the conversation continues. Claims, evidence, and verdicts are kept outside the chat history sent to the LLM. **New chat** clears the conversation and its verification results.

The first fact-check starts immediately. Checking the same response again opens a confirmation modal: **Run again** replaces its existing results, while **Cancel**, the close button, or Escape keeps them. The progress indicator disappears when the run ends. Expand **Evidence passage** to read the passage together with its source name and training step.

During a fact-check, its button changes to **Pause fact-check**. Pausing stops new requests from starting while retaining the current pipeline, progress, and intermediate results. Requests already sent may finish and their responses are retained for resumption. Select **Resume fact-check** to continue without repeating completed requests. The progress bar stays visible while paused; activity icons stop. **New chat** discards paused or running checks and cancels work that has not started.

After decomposition, the Factchecker model assesses each claim independently in its own check-worthiness request. These requests run concurrently. Non-check-worthy claims remain visible with a skipped message and receive no verification verdict. If every claim is skipped, neither the tokenizer nor Elasticsearch is initialized. The app searches Elasticsearch concurrently for all check-worthy claims, then submits the individual claim–evidence verification requests concurrently across all claims. Tokenization and decoding happen on the calling thread. Results retain the original claim order and Elasticsearch evidence order. When retrieval returns no usable evidence, the app explicitly displays that condition with `Not enough information` and skips the verification API call.

`--max-concurrency` (default: `8`) limits simultaneous HTTP requests in each of the check-worthiness, Elasticsearch retrieval, and verification stages; use `1` for sequential execution. These are parallel requests to Chat Completions and Elasticsearch, not asynchronous OpenAI Batch API jobs. The claim list appears immediately after decomposition. Check-worthiness decisions, retrieved passages and source details, and individual verification verdicts appear as each request finishes, without waiting for earlier claims or other evidence pairs. Claims and evidence stay in their original display positions. Progress counts completed work, regardless of completion order. API and retrieval failures are reported as errors rather than verdicts. Available intermediate results and completed verdicts remain visible after an error or interruption; queued work is cancelled on failure and already-running requests are allowed to finish.

## Replace prompts

Prompts are UTF-8 JSON files containing `system` and `user` strings. Edit these strings to replace instructions, guidelines, or few-shot examples.

| Task | Default file | Required placeholders | Optional placeholders |
| --- | --- | --- | --- |
| Decomposition | `prompts/decomposition_8shot.json` | `{{document}}` | `{{context}}` |
| Check-worthiness | `prompts/checkworthiness.json` | `{{claim}}` | None |
| Verification | `prompts/verification.json` | `{{claim}}`, `{{evidence}}` | None |

The check-worthiness template receives one claim in `{{claim}}`. Edit its `system` and `user` strings to change selection criteria or examples; files are reread on every call. Existing custom check-worthiness templates must replace `{{claims}}` with `{{claim}}` and request a single boolean `label`. The bundled criteria follow `v3.0`, with its subjective-opinion example corrected to `false` to match the stated criteria.

For example, a minimal decomposition prompt is:

```json
{
  "system": "Decompose the response into minimal claims that can each be understood independently.",
  "user": "Context:\n{{context}}\nResponse:\n{{document}}"
}
```

Ordinary JSON braces do not need escaping. Supported `{{...}}` placeholders are expanded once; placeholder-like text inside inputs is preserved. Files are reread on each call, so edits take effect on subsequent calls. Select alternate files without changing Python code:

```bash
uv run --locked streamlit run src/serve.py -- \
  --engine your-factcheck-model-or-deployment \
  --chat-engine your-chat-model-or-deployment \
  --decomposition-prompt /path/to/decomposition.json \
  --checkworthiness-prompt /path/to/checkworthiness.json \
  --verification-prompt /path/to/verification.json
```

Python callers can set `prompt_path` on each function. Run scripts with `PYTHONPATH=src uv run --locked python your_script.py`:

```python
from checkworthy import identify_checkworthiness
from decompose import decompose_document_into_claims
from verify import verify_claim

claims = decompose_document_into_claims(
    "London is the capital of England.",
    model="your-factcheck-model-or-deployment",
    prompt_path="prompts/decomposition_8shot.json",
)
for claim in claims:
    is_checkworthy = identify_checkworthiness(
        claim,
        model="your-factcheck-model-or-deployment",
        prompt_path="prompts/checkworthiness.json",
    )
    if not is_checkworthy:
        continue
    result = verify_claim(
        claim,
        "London is the capital of England.",
        model="your-factcheck-model-or-deployment",
        prompt_path="prompts/verification.json",
    )
    # The result contains "label" and "rationale".
```

JSON Schemas in Python define the output contract: decomposition returns a list of claim strings, check-worthiness reads a single boolean `label` and returns that boolean, and verification returns `label` and `rationale`. Application validation also rejects invalid values. Prompt files control instructions and examples independently of these schemas. Custom prompts should request JSON output rather than function calls. Verification accepts exactly five labels:

- `Supported`
- `Partially supported`
- `Partially refuted`
- `Refuted`
- `Not enough information`

The default decomposition prompt uses **guideline + 8-shot**, based on the rule/example/explanation guidelines and eight development examples from the paper's `experiment/lrec2026` branch. The 13 guideline sections and their instructional examples are retained in Japanese to preserve the source wording. Demonstration outputs use the application's JSON Schema contract. Conversation context is used only to resolve references; check-worthiness remains a separate step. The source revision, example IDs, and file paths are recorded in [evaluations/decomposition_8shot_source.json](evaluations/decomposition_8shot_source.json). Edit `prompts/decomposition_8shot.json` to change the default instructions. The zero-shot variant remains available in `prompts/decomposition.json`.

The paper does not define this application's five-label verification scheme. The bundled verification criteria remain provisional and can be replaced independently.

## Evaluate claim decomposition

See the completed [guideline/zero-shot evaluation](evaluations/decomposition_guideline_zero_shot.md) and [guideline/8-shot comparison](evaluations/decomposition_guideline_8shot.md) for five-run AIO test results with `gpt-5.4-2026-03-05`.

The evaluation follows §6.2 of [Masano et al. (LREC 2026)](https://aclanthology.org/2026.lrec-1.186/): exact string matching and fuzzy matching with content-word Jaccard similarity. A maximum-weight one-to-one assignment matches predicted and gold claims before applying the threshold. Fuzzy matches use `similarity >= 0.8`, following the reference code. MeCab with UniDic-lite extracts nouns, verbs, adjectives, and adverbs using the reference evaluator's field 10, with surface-form fallback. Precision, recall, and F1 are calculated per generated text and macro-averaged separately. The reported result averages five independent prediction runs.

Install the optional evaluation dependencies and evaluate the default **guideline + 8-shot** prompt:

```bash
uv sync --locked --group evaluation
uv run --locked --group evaluation python scripts/evaluate_decomposition.py \
  --dataset-repo /path/to/llm-jp-evidence-verification-dataset \
  --output result/decomposition-guideline-8shot
```

The runner reads the pinned experiment revision directly from the dataset repository using `git show`; it does not change that repository's checkout. It uses the original AIO test split (210 texts, 1,169 gold claims), including non-check-worthy claims. It neither filters claims nor passes gold claims or the dataset's questions to the model. Only the generated text is decomposed, matching the reference experiment. The CBA test split is absent from this experiment revision and is not recreated or substituted with the verification dataset's split.

To evaluate **guideline + 0-shot** instead, select both its prompt and source manifest:

```bash
uv run --locked --group evaluation python scripts/evaluate_decomposition.py \
  --dataset-repo /path/to/llm-jp-evidence-verification-dataset \
  --prompt prompts/decomposition.json \
  --source-manifest evaluations/decomposition_source.json \
  --model gpt-5.4-2026-03-05 \
  --output result/decomposition-guideline-zero-shot
```

The eight demonstrations come from the pinned branch's `scripts/prompts/decomposition/fewshot/8shot.txt`. Its `manual_rule+example+expl_8shot.txt` actually contains only six demonstrations; those six are identical to the first six in the complete few-shot file. The eight-shot prompt combines the existing zero-shot guideline with all eight examples and adapts their output wrappers to `{"claims": [...]}`. All eight inputs were verified against the development set and have no overlap with the test set. Their document IDs and source paths are recorded in the source manifest. The app and evaluation runner use eight-shot by default. To select zero-shot in the app, pass `--decomposition-prompt prompts/decomposition.json` when starting Streamlit.

This command sends live requests through the Factchecker endpoint configured in `.env`. Model precedence is `--model`, then `FACTCHECKER_MODEL`, then `gpt-5.4-2026-03-05`. `--prompt` changes the editable prompt file; `--runs` defaults to `5` and `--max-concurrency` to `8`. `--request-timeout` sets the API request timeout in seconds (default: `120`, with the SDK's standard retries). `--limit N` runs a clearly marked smoke-test subset. Each completed prediction is saved; rerunning the same command resumes missing predictions. A changed model, prompt, dataset, generation implementation, or dependency version requires a different output directory. Incomplete runs fail instead of silently dropping missing documents.

The output directory contains the prompt snapshot, gold data, protocol and SHA-256 hashes, predictions for each run, per-document scores and match assignments, and `summary.json` with the run means and population standard deviations. Files under `result/` are ignored by Git. Recompute metrics without API requests using the same command with `--score-only`.

Matching uses the reference Hungarian implementation, including its tie resolution. Different optimal assignments can have the same total similarity but different numbers of pairs above the threshold. If only `src/claim_metrics.py` changes, use `--score-only --refresh-metrics` to re-score existing predictions; the previous scoring protocol is preserved in `previous_scoring_protocol.json`. Changes to the model, prompt, dataset, generation code, or dependency versions still require a separate output directory.

These results evaluate the application's configured model and Structured Outputs. The paper used GPT-4o, while this application defaults to GPT-5.4 and provider-default sampling. The prompt's output wrapper and handling of application context also differ from the original experimental runner. Treat these as an evaluation using the paper's guidelines and metrics, not an exact replication of its published model scores. Exact matching preserves the dataset's original whitespace and punctuation; the application's returned predictions have outer whitespace stripped as in normal use.

## Development

`uv sync --locked` includes the development group with Ruff and pre-commit. Run the checks with:

```bash
PYTHONPATH=src uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv lock --check
```

To include the evaluation metric tests, use `PYTHONPATH=src uv run --locked --group evaluation python -m unittest discover -s tests -v`. These tests need no API access. Without that dependency group, metric tests are skipped.

To enable the Git hooks, run `uv run --locked pre-commit install`. The hooks use the project's locked Ruff version and check that `uv.lock` is up to date.

Tests mock LLM calls, retrieval, and tokenizer loading. SDK requests also run against a mock HTTP transport to check independent endpoint routing, authentication, and Azure API versions for all provider combinations. Tests cover multi-turn chat, response-specific verification and result persistence, decomposition context, conversation reset, prompt replacement, output validation, pair-level verdicts, progress, and error handling. They do not send live API requests. Decomposition accuracy evaluation uses the live Factchecker model and does not require Elasticsearch.

Manage dependencies with `uv add`, `uv add --dev`, and `uv remove`. Commit changes to both `pyproject.toml` and `uv.lock` together.
