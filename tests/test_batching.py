"""Concurrency limits, ordering, and cleanup without network requests."""

import unittest
from functools import partial
from threading import Event, Lock, Thread

from batching import request_batch


class RequestBatchTests(unittest.TestCase):
    def test_limits_running_requests_and_keeps_input_order(self):
        release = Event()
        two_started = Event()
        lock = Lock()
        active = 0
        peak = 0

        def request(index):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 2:
                    two_started.set()
            try:
                if not release.wait(3):
                    raise TimeoutError("Requests were not released")
                return index
            finally:
                with lock:
                    active -= 1

        with request_batch([partial(request, index) for index in range(6)], 2) as futures:
            try:
                self.assertTrue(two_started.wait(2), "Requests did not overlap")
                self.assertFalse(any(future.running() for future in futures[2:]))
            finally:
                release.set()
            self.assertEqual([future.result() for future in futures], list(range(6)))
        self.assertEqual(peak, 2)
        self.assertEqual(active, 0)

    def test_exit_cancels_queued_requests_and_waits_for_running_request(self):
        started = Event()
        release = Event()
        cancelled = Event()

        def request():
            started.set()
            if not release.wait(3):
                raise TimeoutError("Request was not released")
            return "done"

        batch = request_batch([request] * 4, 1)
        futures = batch.__enter__()
        futures[-1].add_done_callback(lambda future: cancelled.set() if future.cancelled() else None)
        closing = Thread(target=lambda: batch.__exit__(None, None, None))
        try:
            self.assertTrue(started.wait(2))
            closing.start()
            self.assertTrue(cancelled.wait(2), "Queued work was not cancelled")
            self.assertTrue(all(future.cancelled() for future in futures[1:]))
            self.assertTrue(closing.is_alive(), "Shutdown did not wait for the running request")
        finally:
            release.set()
            if closing.ident is None:
                batch.__exit__(None, None, None)
            else:
                closing.join(timeout=3)
        self.assertFalse(closing.is_alive())
        self.assertEqual(futures[0].result(), "done")


if __name__ == "__main__":
    unittest.main()
