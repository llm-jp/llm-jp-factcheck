# LLM-jp Factcheck

A Streamlit application for chatting with an LLM and checking individual responses against a searchable corpus. Each response is decomposed into claims, check-worthy claims are selected, and each selected claim–evidence pair receives a verdict and rationale. The interface uses a restrained white theme with progress messages throughout generation and verification.

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

Retrieval requires an existing Elasticsearch search index containing `token_ids` (space-separated token IDs), `dataset_name` (source), and `iteration` (training step). Source details come directly from each search result.

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

Results remain associated with their response as the conversation continues. Claims, evidence, and verdicts are kept outside the chat history sent to the LLM. **New chat** clears the conversation and its verification results.

The first fact-check starts immediately. Checking the same response again shows an inline confirmation: **Run again** replaces its existing results, while **Cancel** keeps them. The progress indicator disappears when the run ends. Expand **Evidence passage** to read a passage and **Source details** to see its source name and training step.

After decomposition, the Factchecker model assesses each claim independently in its own check-worthiness request. These requests run concurrently. Non-check-worthy claims remain visible with a skipped message and receive no verification verdict. If every claim is skipped, neither the tokenizer nor Elasticsearch is initialized. The app searches Elasticsearch concurrently for all check-worthy claims, then submits the individual claim–evidence verification requests concurrently across all claims. Tokenization and decoding happen on the calling thread. Results retain the original claim order and Elasticsearch evidence order. When retrieval returns no usable evidence, the app explicitly displays that condition with `Not enough information` and skips the verification API call.

`--max-concurrency` (default: `8`) limits simultaneous HTTP requests in each of the check-worthiness, Elasticsearch retrieval, and verification stages; use `1` for sequential execution. These are parallel requests to Chat Completions and Elasticsearch, not asynchronous OpenAI Batch API jobs. Progress messages report completed claims, searches, or pairs as results are collected in input order. API and retrieval failures are reported as errors rather than verdicts. Completed claim results remain visible; queued work is cancelled on failure and already-running requests are allowed to finish.

## Replace prompts

Prompts are UTF-8 JSON files containing `system` and `user` strings. Edit these strings to replace instructions, guidelines, or few-shot examples.

| Task | Default file | Required placeholders | Optional placeholders |
| --- | --- | --- | --- |
| Decomposition | `prompts/decomposition.json` | `{{document}}` | `{{context}}` |
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
    prompt_path="prompts/decomposition.json",
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

The bundled decomposition prompt is provisional and follows the decomposition guidelines in §4 of [Masano et al. (LREC 2026)](https://aclanthology.org/2026.lrec-1.186/). It does not reproduce the paper's complete experimental prompt or its eight selected examples. That paper does not define this five-label verification scheme, so the bundled verification criteria are also provisional. Replace both files when the intended prompts are available.

## Development

`uv sync --locked` includes the development group with Ruff and pre-commit. Run the checks with:

```bash
PYTHONPATH=src uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
uv lock --check
```

To enable the Git hooks, run `uv run --locked pre-commit install`. The hooks use the project's locked Ruff version and check that `uv.lock` is up to date.

Tests mock LLM calls, retrieval, and tokenizer loading. SDK requests also run against a mock HTTP transport to check independent endpoint routing, authentication, and Azure API versions for all provider combinations. Tests cover multi-turn chat, response-specific verification and result persistence, decomposition context, conversation reset, prompt replacement, output validation, pair-level verdicts, progress, and error handling. They do not send live API requests. Accuracy evaluation requires the actual models and indexes.

Manage dependencies with `uv add`, `uv add --dev`, and `uv remove`. Commit changes to both `pyproject.toml` and `uv.lock` together.
