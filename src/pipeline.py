"""Fact-checking workflow, with progress events independent of the UI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Iterator

from batching import request_batch
from checkworthy import identify_checkworthiness
from decompose import decompose_document_into_claims
from retrieval import create_elasticsearch_client, search_documents
from verify import verify_claim


@dataclass(frozen=True)
class PipelineConfig:
    engine: str = "gpt-5.4-2026-03-05"
    tokenizer_name: str = "llm-jp/llm-jp-3-13b"
    es_host: str = "http://localhost:9200"
    es_dump_index: str = "llm-jp-corpus-v3"
    num_evidences: int = 1
    max_concurrency: int = 8
    decomposition_prompt: str | None = None
    checkworthiness_prompt: str | None = None
    verification_prompt: str | None = None

    def __post_init__(self) -> None:
        if self.num_evidences < 1:
            raise ValueError("num_evidences must be at least 1.")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")


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
    """Decompose text, select check-worthy claims, retrieve evidence, and verify pairs.

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

    yield PipelineEvent("checkworthiness", f"Assessing check-worthiness: 0 / {total} claims…", total_claims=total)
    requests = [
        partial(identify_checkworthiness, claim, model=config.engine, prompt_path=config.checkworthiness_prompt)
        for claim in claims
    ]
    labels = []
    with request_batch(requests, config.max_concurrency) as futures:
        for index, future in enumerate(futures, 1):
            labels.append(future.result())
            yield PipelineEvent(
                "checkworthiness", f"Check-worthiness assessed: {index} / {total} claims.", index, total
            )
    if any(labels):
        yield PipelineEvent(
            "preparation",
            "Preparing retrieval. The first run may require a tokenizer download…",
            total_claims=total,
        )
        tokenizer = get_tokenizer(config.tokenizer_name)
        yield PipelineEvent("preparation", "Connecting to the evidence database…", total_claims=total)
        es = get_search_client(config.es_host)

    results = [
        {"claim": claim, "is_checkworthy": label, "evidences": [], "no_evidence": False}
        for claim, label in zip(claims, labels, strict=True)
    ]
    search_targets = [(index, result) for index, result in enumerate(results, 1) if result["is_checkworthy"]]
    if search_targets:
        yield PipelineEvent(
            "retrieval", f"Searching for evidence: 0 / {len(search_targets)} claims…", total_claims=total
        )
        requests = []
        # Keep tokenizer operations on the calling thread; workers only perform searches.
        for _, result in search_targets:
            claim_token_ids = tokenizer.encode(result["claim"], add_special_tokens=False)
            requests.append(
                partial(
                    search_documents,
                    es,
                    config.es_dump_index,
                    body={"query": {"match": {"token_ids": " ".join(map(str, claim_token_ids))}}},
                    size=config.num_evidences,
                    max_concurrent_shard_requests=64,
                )
            )
        with request_batch(requests, config.max_concurrency) as futures:
            for completed, ((index, result), future) in enumerate(zip(search_targets, futures, strict=True), 1):
                evidences = []
                for hit in future.result():
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
                result["evidences"] = evidences
                result["no_evidence"] = not evidences
                yield PipelineEvent(
                    "retrieval",
                    f"Evidence searches completed: {completed} / {len(search_targets)} claims.",
                    index,
                    total,
                )

    requests = [
        partial(
            verify_claim,
            result["claim"],
            evidence["passage"],
            model=config.engine,
            prompt_path=config.verification_prompt,
        )
        for result in results
        for evidence in result["evidences"]
    ]
    if requests:
        yield PipelineEvent("verification", f"Verifying claim–evidence pairs: 0 / {len(requests)}…", total_claims=total)
    with request_batch(requests, config.max_concurrency) as futures:
        pending = iter(futures)
        completed_pairs = 0
        for index, result in enumerate(results, 1):
            for evidence in result["evidences"]:
                evidence["verification"] = next(pending).result()
                completed_pairs += 1
                yield PipelineEvent(
                    "verification", f"Verified claim–evidence pairs: {completed_pairs} / {len(requests)}.", index, total
                )
            if not result["is_checkworthy"]:
                message = "not check-worthy; retrieval and verification skipped."
            elif result["no_evidence"]:
                message = "no evidence found."
            else:
                message = "verification complete."
            yield PipelineEvent("claim_complete", f"Claim {index} / {total}: {message}", index, total, result=result)

    yield PipelineEvent("complete", f"Fact-check complete. Claims processed: {total}.", total, total)
