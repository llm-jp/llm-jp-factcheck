"""Run independent synchronous requests concurrently with bounded worker counts."""

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Condition
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")


class ExecutionCancelled(Exception):
    """The owning fact-check was discarded."""


class RunControl:
    """Pause new work without discarding responses already in flight."""

    def __init__(self):
        self._condition = Condition()
        self._paused = False
        self._cancelled = False

    def pause(self):
        with self._condition:
            self._paused = True

    def resume(self):
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def cancel(self):
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()

    def checkpoint(self):
        with self._condition:
            self._condition.wait_for(lambda: not self._paused or self._cancelled)
            if self._cancelled:
                raise ExecutionCancelled()

    def call(self, request: Callable[[], T]) -> T:
        self.checkpoint()
        return request()


_control: ContextVar[RunControl | None] = ContextVar("factcheck_control", default=None)


@contextmanager
def controlled_execution(control: RunControl):
    token = _control.set(control)
    try:
        control.checkpoint()
        yield
    finally:
        _control.reset(token)


def completed_requests(futures: list[Future[T]]) -> Iterator[tuple[int, Future[T]]]:
    """Yield completed requests immediately, retaining their original positions."""
    positions = {future: index for index, future in enumerate(futures)}
    for future in as_completed(futures):
        yield positions[future], future


@contextmanager
def request_batch(requests: Iterable[Callable[[], T]], max_concurrency: int) -> Iterator[list[Future[T]]]:
    """Submit requests together and keep futures in input order.

    Workers only execute requests; callers collect results and update the UI.
    On exit, cancel queued work and wait for already-running requests to finish.
    """
    executor = ThreadPoolExecutor(max_workers=max_concurrency)
    control = _control.get()
    try:
        yield [executor.submit(control.call, request) if control else executor.submit(request) for request in requests]
    except BaseException:
        if control:
            control.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
