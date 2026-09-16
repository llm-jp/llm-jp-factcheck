"""Fact-checking workflow, with progress events independent of the UI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

from decompose import decompose_document_into_claims
from retrieval import create_elasticsearch_client, search_documents
from verify import verify_claim


@dataclass(frozen=True)
class PipelineConfig:
    engine: str = "gpt-4-0613"
    tokenizer_name: str = "llm-jp/llm-jp-3-13b"
    es_host: str = "http://localhost:9200"
    es_dump_index: str = "llm-jp-corpus-v3"
    num_evidences: int = 1
    decomposition_prompt: str | None = None
    verification_prompt: str | None = None

    def __post_init__(self) -> None:
        if self.num_evidences < 1:
            raise ValueError("num_evidences must be at least 1.")


@dataclass
class PipelineEvent:
    stage: str
    message: str
    current_claim: int = 0
    total_claims: int = 0
    result: dict | None = None
    claims: list[str] | None = None


@lru_cache(maxsize=4)
def get_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


@lru_cache(maxsize=4)
def get_search_client(host: str):
    return create_elasticsearch_client(host)


def run_factcheck(document: str, context: str | None, config: PipelineConfig) -> Iterator[PipelineEvent]:
    """Decompose text, retrieve evidence, and verify each claim–evidence pair.

    Yield before slow operations so callers can show the current activity.
    An empty retrieval is reported separately from a model's pair verdict.
    Errors propagate to the caller; they are never turned into verdicts.
    """
    if not document.strip():
        yield PipelineEvent("complete", "No text to check.", claims=[])
        return

    yield PipelineEvent("decomposition", "Decomposing the response into claims…")
    claims = decompose_document_into_claims(
        document,
        model=config.engine,
        context=context,
        prompt_path=config.decomposition_prompt,
    )
    total = len(claims)
    yield PipelineEvent("decomposition", f"Claims extracted: {total}.", total_claims=total, claims=claims)
    if not claims:
        yield PipelineEvent("complete", "No claims found.", claims=[])
        return

    yield PipelineEvent(
        "preparation",
        "Preparing retrieval. The first run may require a tokenizer download…",
        total_claims=total,
    )
    tokenizer = get_tokenizer(config.tokenizer_name)
    yield PipelineEvent("preparation", "Connecting to the evidence database…", total_claims=total)
    es = get_search_client(config.es_host)

    for index, claim in enumerate(claims, 1):
        prefix = f"Claim {index} / {total}"
        yield PipelineEvent("retrieval", f"{prefix}: searching for evidence…", index, total)
        claim_token_ids = tokenizer.encode(claim, add_special_tokens=False)
        hits = search_documents(
            es,
            config.es_dump_index,
            body={"query": {"match": {"token_ids": " ".join(map(str, claim_token_ids))}}},
            size=config.num_evidences,
            max_concurrent_shard_requests=64,
        )
        evidences = []
        for hit in hits:
            source = hit["_source"]
            passage = tokenizer.decode(list(map(int, source["token_ids"].split()))).strip()
            if passage:
                evidences.append(
                    {
                        "passage": passage,
                        "dataset": source.get("dataset_name", ""),
                        "training_step": source.get("iteration"),
                    }
                )

        for evidence_index, evidence in enumerate(evidences, 1):
            yield PipelineEvent(
                "verification",
                f"{prefix}: verifying against evidence {evidence_index} / {len(evidences)}…",
                index,
                total,
            )
            evidence["verification"] = verify_claim(
                claim,
                evidence["passage"],
                model=config.engine,
                prompt_path=config.verification_prompt,
            )

        result = {"claim": claim, "evidences": evidences, "no_evidence": not evidences}
        yield PipelineEvent(
            "claim_complete",
            f"{prefix}: verification complete." if evidences else f"{prefix}: no evidence found.",
            index,
            total,
            result=result,
        )

    yield PipelineEvent("complete", f"Fact-check complete. Claims processed: {total}.", total, total)
