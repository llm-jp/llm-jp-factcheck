"""Keep an in-progress pipeline alive across Streamlit reruns and pauses."""

from concurrent.futures import Future, ThreadPoolExecutor
from weakref import finalize

from batching import RunControl, controlled_execution


def _next_event(events, control):
    try:
        with controlled_execution(control):
            return next(events)
    except BaseException:
        control.cancel()
        close = getattr(events, "close", None)
        if close is not None:
            close()
        raise


def _dispose_job(control, executor, events):
    control.cancel()
    # Close suspended generators on their own worker, after any current call.
    close = getattr(events, "close", None)
    if close is not None:
        executor.submit(close)
    executor.shutdown(wait=False)


class FactcheckJob:
    """Advance one generator on a dedicated worker; only the UI consumes events."""

    def __init__(self, events):
        self.control = RunControl()
        self._events = events
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="factcheck")
        self.pending: Future | None = None
        self._cleanup = finalize(self, _dispose_job, self.control, self._executor, events)

    def advance(self) -> Future:
        if self.pending is None:
            self.pending = self._executor.submit(_next_event, self._events, self.control)
        return self.pending

    def consume(self):
        event = self.pending.result()
        self.pending = None
        return event

    def close(self):
        self._cleanup()
