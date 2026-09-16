"""Deterministic fictional fixtures for trying the interface entirely offline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from time import sleep
from typing import Iterator

from batching import request_batch
from pipeline import PipelineConfig, PipelineEvent

CHAT_INITIAL_DELAY = 1.0
CHAT_CHUNK_DELAY = 0.08
DECOMPOSITION_DELAY = 1.0
CHECKWORTHINESS_DELAY = 0.6
PREPARATION_DELAY = 0.4
RETRIEVAL_STAGE_DELAY = 0.4
VERIFICATION_DELAY = 0.6


@dataclass(frozen=True)
class _Fixture:
    claim: str
    label: str
    passages: tuple[str, ...]
    rationale: str
    is_checkworthy: bool = True


_FIXTURES = (
    _Fixture(
        "Northstar Museum opened in 2012.",
        "Supported",
        (
            "Fictional opening record: Northstar Museum first opened to visitors in 2012.",
            "Fictional anniversary note: Northstar Museum celebrated ten years since its 2012 opening in 2022.",
        ),
        "The sample record explicitly confirms that the museum opened in 2012.",
    ),
    _Fixture(
        "Northstar Museum opens at 9 a.m. and closes at 6 p.m.",
        "Partially supported",
        (
            "Fictional visitor guide: Northstar Museum opens at 9 a.m. The guide does not list a closing time.",
            "Fictional timetable: Northstar Museum opening time is 09:00. Closing time is not recorded.",
        ),
        "The sample supports the 9 a.m. opening time but leaves the claimed 6 p.m. closing time unresolved.",
    ),
    _Fixture(
        "Northstar Museum has a cafe and a gift shop.",
        "Partially refuted",
        (
            "Fictional facilities list: Northstar Museum has a cafe. It does not have a gift shop.",
            "Fictional floor plan: Northstar Museum includes a cafe; the museum has no gift shop.",
        ),
        "The sample confirms the cafe but contradicts the gift shop, refuting one part of the claim.",
    ),
    _Fixture(
        "Northstar Museum is closed on Mondays.",
        "Refuted",
        (
            "Fictional weekly schedule: Northstar Museum is open every Monday.",
            "Fictional visitor notice: Northstar Museum welcomes visitors on Mondays and is closed on Tuesdays.",
        ),
        "The sample directly contradicts the claim by stating that the museum is open on Mondays.",
    ),
    _Fixture(
        "The director of Northstar Museum is Morgan Vale.",
        "Not enough information",
        (
            "Fictional access guide: Northstar Museum has bicycle parking beside its entrance.",
            "Fictional visitor guide: Northstar Museum provides lockers for visitors' bags.",
        ),
        "The sample describes visitor facilities and provides no information about the museum's director.",
    ),
    _Fixture("Northstar Museum is wonderful.", "", (), "", is_checkworthy=False),
)

MOCK_CLAIMS = tuple(fixture.claim for fixture in _FIXTURES)


def stream_mock_chat_response(messages: list[dict], model: str) -> Iterator[str]:
    """Stream a fixed sample, with the current user-turn count for orientation."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("Chat history must contain a user message.")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            raise ValueError("Chat messages must have a user or assistant role.")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("Chat messages must contain nonempty text.")
    if messages[-1]["role"] != "user":
        raise ValueError("The last chat message must be from the user.")
    turn = sum(message["role"] == "user" for message in messages)
    response = (
        f"Fictional sample response (turn {turn})\n\n"
        "These fixed claims describe the invented Northstar Museum. "
        "Select Fact-check response to view sample evidence and verdicts.\n\n"
        + "\n".join(f"- {claim}" for claim in MOCK_CLAIMS)
    )
    sleep(CHAT_INITIAL_DELAY)
    for start in range(0, len(response), 48):
        if start:
            sleep(CHAT_CHUNK_DELAY)
        yield response[start : start + 48]


def _check_fixture(fixture: _Fixture) -> bool:
    sleep(CHECKWORTHINESS_DELAY)
    return fixture.is_checkworthy


def _verify_fixture(fixture: _Fixture) -> dict[str, str]:
    sleep(VERIFICATION_DELAY)
    return {"label": fixture.label, "rationale": fixture.rationale}


def _retrieve_fixture(fixture: _Fixture, num_evidences: int) -> list[dict]:
    sleep(RETRIEVAL_STAGE_DELAY)
    return [{"passage": passage, "dataset": "Mock evidence"} for passage in fixture.passages[:num_evidences]]


def run_mock_factcheck(document: str, context: str | None, config: PipelineConfig) -> Iterator[PipelineEvent]:
    """Check only recognized fixture claims using bundled synthetic evidence."""
    yield PipelineEvent("decomposition", "Finding sample claims in this response…")
    sleep(DECOMPOSITION_DELAY)
    fixtures = sorted(
        (fixture for fixture in _FIXTURES if fixture.claim in document),
        key=lambda fixture: document.index(fixture.claim),
    )
    total = len(fixtures)
    yield PipelineEvent(
        "decomposition", f"Sample claims found: {total}.", total_claims=total, claims=[item.claim for item in fixtures]
    )
    if not fixtures:
        yield PipelineEvent("complete", "No recognized sample claims in this response.", claims=[])
        return
    yield PipelineEvent("checkworthiness", "Identifying check-worthy sample claims…", total_claims=total)
    labels = []
    with request_batch([partial(_check_fixture, fixture) for fixture in fixtures], config.max_concurrency) as futures:
        for index, future in enumerate(futures, 1):
            labels.append(future.result())
            yield PipelineEvent(
                "checkworthiness", f"Sample check-worthiness assessed: {index} / {total} claims.", index, total
            )
    if any(labels):
        yield PipelineEvent("preparation", "Preparing bundled mock evidence…", total_claims=total)
        sleep(PREPARATION_DELAY)
    results = [
        {
            "claim": fixture.claim,
            "is_checkworthy": is_checkworthy,
            "evidences": [],
            "no_evidence": False,
            "mock": True,
        }
        for fixture, is_checkworthy in zip(fixtures, labels, strict=True)
    ]
    search_targets = [index for index, label in enumerate(labels) if label]
    if search_targets:
        yield PipelineEvent(
            "retrieval", f"Selecting mock evidence: 0 / {len(search_targets)} claims…", total_claims=total
        )
        requests = [partial(_retrieve_fixture, fixtures[index], config.num_evidences) for index in search_targets]
        with request_batch(requests, config.max_concurrency) as futures:
            for completed, (index, future) in enumerate(zip(search_targets, futures, strict=True), 1):
                results[index]["evidences"] = future.result()
                yield PipelineEvent(
                    "retrieval",
                    f"Mock evidence selected: {completed} / {len(search_targets)} claims.",
                    index + 1,
                    total,
                )
    requests = [
        partial(_verify_fixture, fixture)
        for fixture, result in zip(fixtures, results, strict=True)
        for _ in result["evidences"]
    ]
    if requests:
        yield PipelineEvent("verification", f"Applying sample verdicts: 0 / {len(requests)}…", total_claims=total)
    with request_batch(requests, config.max_concurrency) as futures:
        pending = iter(futures)
        completed_pairs = 0
        for index, result in enumerate(results, 1):
            for evidence in result["evidences"]:
                evidence["verification"] = next(pending).result()
                completed_pairs += 1
                yield PipelineEvent(
                    "verification", f"Sample verdicts applied: {completed_pairs} / {len(requests)}.", index, total
                )
            message = (
                "sample verification complete."
                if result["is_checkworthy"]
                else "not check-worthy; retrieval and verification skipped."
            )
            yield PipelineEvent(
                "claim_complete", f"Sample claim {index} / {total}: {message}", index, total, result=result
            )
    yield PipelineEvent("complete", f"Sample fact-check complete. Claims processed: {total}.", total, total)
