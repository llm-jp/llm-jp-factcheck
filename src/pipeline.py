"""Fact-checking workflow, with progress events independent of the UI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Iterator

from batching import completed_requests, request_batch
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
    num_evidences: int = 3
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
    claim_update: dict | None = None
    progress: float | None = None


def claim_event(stage: str, message: str, index: int, results: list[dict], progress: float) -> PipelineEvent:
    """Snapshot one claim so later updates cannot change an already emitted event."""
    snapshot = deepcopy(results[index])
    return PipelineEvent(
        stage,
        message,
        index + 1,
        len(results),
        result=snapshot if stage == "claim_complete" else None,
        claim_update=snapshot,
        progress=progress,
    )


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
    results = [
        {"claim": claim, "is_checkworthy": None, "evidences": [], "no_evidence": False, "status": "checkworthiness"}
        for claim in claims
    ]
    with request_batch(requests, config.max_concurrency) as futures:
        for completed, (index, future) in enumerate(completed_requests(futures), 1):
            result = results[index]
            result["is_checkworthy"] = future.result()
            result["status"] = "retrieval" if result["is_checkworthy"] else "complete"
            progress = 0.05 + 0.15 * completed / total
            yield claim_event(
                "checkworthiness", f"Check-worthiness assessed: {completed} / {total} claims.", index, results, progress
            )
            if not result["is_checkworthy"]:
                yield claim_event(
                    "claim_complete",
                    f"Claim {index + 1} / {total}: not check-worthy; retrieval and verification skipped.",
                    index,
                    results,
                    progress,
                )
    if any(result["is_checkworthy"] for result in results):
        yield PipelineEvent(
            "preparation",
            "Preparing retrieval. The first run may require a tokenizer download…",
            total_claims=total,
        )
        tokenizer = get_tokenizer(config.tokenizer_name)
        yield PipelineEvent("preparation", "Connecting to the evidence database…", total_claims=total)
        es = get_search_client(config.es_host)

    search_targets = [index for index, result in enumerate(results) if result["is_checkworthy"]]
    if search_targets:
        yield PipelineEvent(
            "retrieval", f"Searching for evidence: 0 / {len(search_targets)} claims…", total_claims=total
        )
        requests = []
        # Keep tokenizer operations on the calling thread; workers only perform searches.
        for index in search_targets:
            result = results[index]
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
            for completed, (position, future) in enumerate(completed_requests(futures), 1):
                index = search_targets[position]
                result = results[index]
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
                result["status"] = "verification" if evidences else "complete"
                progress = 0.25 + 0.25 * completed / len(search_targets)
                yield claim_event(
                    "retrieval",
                    f"Evidence searches completed: {completed} / {len(search_targets)} claims.",
                    index,
                    results,
                    progress,
                )
                if not evidences:
                    yield claim_event(
                        "claim_complete", f"Claim {index + 1} / {total}: no evidence found.", index, results, progress
                    )

    pairs = [(index, evidence) for index, result in enumerate(results) for evidence in result["evidences"]]
    requests = [
        partial(
            verify_claim,
            results[index]["claim"],
            evidence["passage"],
            model=config.engine,
            prompt_path=config.verification_prompt,
        )
        for index, evidence in pairs
    ]
    if requests:
        yield PipelineEvent("verification", f"Verifying claim–evidence pairs: 0 / {len(requests)}…", total_claims=total)
    with request_batch(requests, config.max_concurrency) as futures:
        for completed, (position, future) in enumerate(completed_requests(futures), 1):
            index, evidence = pairs[position]
            evidence["verification"] = future.result()
            result = results[index]
            finished = all("verification" in item for item in result["evidences"])
            if finished:
                result["status"] = "complete"
            progress = 0.5 + 0.48 * completed / len(pairs)
            yield claim_event(
                "verification", f"Verified claim–evidence pairs: {completed} / {len(pairs)}.", index, results, progress
            )
            if finished:
                yield claim_event(
                    "claim_complete", f"Claim {index + 1} / {total}: verification complete.", index, results, progress
                )

    yield PipelineEvent("complete", f"Fact-check complete. Claims processed: {total}.", total, total)
