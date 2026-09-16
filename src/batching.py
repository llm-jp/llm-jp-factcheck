"""Run independent synchronous requests concurrently with bounded worker counts."""

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")


@contextmanager
def request_batch(requests: Iterable[Callable[[], T]], max_concurrency: int) -> Iterator[list[Future[T]]]:
    """Submit requests together and keep futures in input order.

    Workers only execute requests; callers collect results and update the UI.
    On exit, cancel queued work and wait for already-running requests to finish.
    """
    executor = ThreadPoolExecutor(max_workers=max_concurrency)
    try:
        yield [executor.submit(request) for request in requests]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
