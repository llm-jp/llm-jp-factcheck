# LLM-jp Factcheck

A Streamlit research demo for chatting with an LLM and checking individual responses against a searchable corpus. Each response is decomposed into claims, and each claim–evidence pair receives a verdict and rationale. The interface uses a restrained white theme with progress messages throughout generation and verification.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run all commands from the project root. The project uses Python 3.12, selected by `.python-version`. Dependencies are declared in `pyproject.toml` and resolved in `uv.lock`.

```bash
uv sync --locked
```

### Model connections

Create `.env` in the project root using [`.env.example`](.env.example) as a template. Configure chat generation with `CHATBOT_` variables and both decomposition and verification with `FACTCHECKER_` variables. Each role can independently use an OpenAI-compatible endpoint or Azure OpenAI.

| Setting | Chat generation | Decomposition and verification | Required |
| --- | --- | --- | --- |
| API type | `CHATBOT_API_TYPE` | `FACTCHECKER_API_TYPE` | `openai` or `azure` |
| Endpoint | `CHATBOT_ENDPOINT` | `FACTCHECKER_ENDPOINT` | Yes |
| API key | `CHATBOT_API_KEY` | `FACTCHECKER_API_KEY` | Yes; use a dummy value for a local server without authentication |
| API version | `CHATBOT_API_VERSION` | `FACTCHECKER_API_VERSION` | Only when API type is `azure` |

With `API_TYPE=openai`, supply the complete API base URL, including its path: `https://api.openai.com/v1` for OpenAI, `http://localhost:8000/v1` for a compatible local server, or `https://your-resource.openai.azure.com/openai/v1` for an [Azure v1 endpoint](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle). The role's `API_VERSION` is ignored in this mode.

With `API_TYPE=azure`, supply the Azure resource root, such as `https://your-resource.openai.azure.com`, and an API version supported by that resource. Model arguments are Azure deployment names in this mode. These modes use the SDK's `OpenAI` and `AzureOpenAI` clients, respectively; see the [official Python SDK reference](https://developers.openai.com/api/reference/python).

For example, use a local chatbot and an Azure fact-checker:

```dotenv
CHATBOT_API_TYPE="openai"
CHATBOT_ENDPOINT="http://localhost:8000/v1"
CHATBOT_API_KEY="unused"

FACTCHECKER_API_TYPE="azure"
FACTCHECKER_ENDPOINT="https://your-resource.openai.azure.com"
FACTCHECKER_API_KEY="your-factchecker-api-key"
FACTCHECKER_API_VERSION="your-supported-api-version"
```

The chatbot endpoint must support streaming chat completions. The fact-checker endpoint, API version, and selected model must support tool calling. Clients are created when first used and cached until the process exits. Restart Streamlit after editing connection settings in `.env` or the environment.

For compatibility, a role with none of its four prefixed connection variables set uses the shared legacy `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `AZURE_OPENAI_API_VERSION` settings. Setting any prefixed connection variable requires a complete configuration for that role; missing values are never filled from the shared settings. Global `OPENAI_*` variables are not used as a fallback. Model selection uses CLI arguments, not environment variables.

### Retrieval

Retrieval requires Elasticsearch with existing search and metadata indexes:

| Index | Document fields |
| --- | --- |
| Search | `token_ids`: space-separated token IDs; `dataset_name` and `iteration`: source details |
| Metadata | `token_ids`: space-separated token IDs; `meta`: a JSON string or object |

Set `--tokenizer_name` to the tokenizer used to build these indexes. The first fact-check downloads the tokenizer and embedding model from Hugging Face if they are not cached.

## Run

```bash
uv run --locked streamlit run src/serve.py -- \
  --engine your-factcheck-deployment-name \
  --chat-engine your-local-model-id
```

`--engine` selects the model for decomposition and verification through the fact-checker connection. `--chat-engine` selects the model for the chatbot connection; its value defaults to `--engine`. Use model IDs accepted by the endpoint in `openai` mode and deployment names in `azure` mode. Azure v1 endpoints also expect deployment names. The command above matches the mixed configuration example.

Retrieval options can be supplied alongside the model arguments:

```bash
uv run --locked streamlit run src/serve.py -- \
  --engine your-factcheck-deployment-name \
  --chat-engine your-local-model-id \
  --es_host http://localhost:9200 \
  --es_dump_index llm-jp-search-v1.0 \
  --es_meta_index llm-jp-search-for-meta-v1.0 \
  --tokenizer_name llm-jp/llm-jp-13b-v1.0 \
  --num_evidences 3
```

## Mock mode

After `uv sync --locked`, try the interface without API credentials, Elasticsearch, or model downloads:

```bash
uv run --locked streamlit run src/serve.py -- --mock
```

`--mock` replaces chat and the entire fact-checking pipeline with bundled, deterministic fictional fixtures. It supports multi-turn chat, streamed response fragments, stage progress, all five verdict labels, and per-response checks. The interface identifies mock mode and synthetic results. The canned content demonstrates the interface; it does not evaluate a model or real-world claims.

Artificial pauses make response generation and verification progress visible: approximately two seconds per chat response and ten seconds per fact-check with the default evidence count, excluding UI overhead. Adjust the delay constants in `src/mock_backend.py` to change these timings.

Mock mode ignores model, endpoint, retrieval, and prompt-selection settings, except `--num_evidences`, which selects up to two available synthetic evidence passages per claim. Fixtures can be edited in `src/mock_backend.py`. Mock mode is off by default; omit `--mock` to use the configured live services.

## Workflow

Send messages through the chat input to continue a conversation. Select **Fact-check response** below any assistant response to check it. Only the selected response is decomposed; its preceding conversation history is supplied automatically as context for resolving references and omitted information.

Results remain associated with their response as the conversation continues. Claims, evidence, and verdicts are kept outside the chat history sent to the LLM. **New chat** clears the conversation and its verification results.

The first fact-check starts immediately. Checking the same response again shows an inline confirmation: **Run again** replaces its existing results, while **Cancel** keeps them. The progress indicator disappears when the run ends. Expand **Evidence passage** to read a passage and **Source details** to see its source name and training step.

For each extracted claim, the app retrieves and ranks evidence passages, then verifies each claim–evidence pair independently. Every extracted claim is processed. When retrieval returns no evidence, the app explicitly displays that condition with `Not enough information` and skips the verification API call. API and retrieval failures are reported as errors rather than verdicts.

## Replace prompts

Prompts are UTF-8 JSON files containing `system` and `user` strings. Edit these strings to replace instructions, guidelines, or few-shot examples.

| Task | Default file | Required placeholders | Optional placeholders |
| --- | --- | --- | --- |
| Decomposition | `prompts/decomposition.json` | `{{document}}` | `{{context}}` |
| Verification | `prompts/verification.json` | `{{claim}}`, `{{evidence}}` | None |

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
  --verification-prompt /path/to/verification.json
```

Python callers can set `prompt_path` on either function. Run scripts with `PYTHONPATH=src uv run --locked python your_script.py`:

```python
from decompose import decompose_document_into_claims
from verify import verify_claim

claims = decompose_document_into_claims(
    "London is the capital of England.",
    model="your-factcheck-model-or-deployment",
    prompt_path="prompts/decomposition.json",
)
if claims:
    result = verify_claim(
        claims[0],
        "London is the capital of England.",
        model="your-factcheck-model-or-deployment",
        prompt_path="prompts/verification.json",
    )
    # The result contains "label" and "rationale".
```

Python tool schemas define the output contract: decomposition returns a list of claim strings, and verification returns `label` and `rationale`. Invalid outputs raise errors. Verification accepts exactly five labels:

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

Tests mock LLM calls, retrieval, and model loading. SDK requests also run against a mock HTTP transport to check independent endpoint routing, authentication, and Azure API versions for all provider combinations. Tests cover multi-turn chat, response-specific verification and result persistence, decomposition context, conversation reset, prompt replacement, output validation, pair-level verdicts, progress, and error handling. They do not send live API requests. Accuracy evaluation requires the actual models and indexes.

Manage dependencies with `uv add`, `uv add --dev`, and `uv remove`. Commit changes to both `pyproject.toml` and `uv.lock` together.
