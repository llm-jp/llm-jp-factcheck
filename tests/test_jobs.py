"""Pausing preserves the generator and completed requests across UI reruns."""

import unittest
from threading import Event

from batching import ExecutionCancelled, completed_requests, request_batch
from jobs import FactcheckJob


class FactcheckJobTests(unittest.TestCase):
    def next_event(self, job):
        job.advance().result(timeout=3)
        return job.consume()

    def test_pause_retains_inflight_result_and_blocks_queued_requests_until_resume(self):
        first_started = Event()
        release_first = Event()
        second_started = Event()
        calls = []

        def first():
            calls.append("first")
            first_started.set()
            if not release_first.wait(3):
                raise TimeoutError("First request was not released")
            return "first result"

        def second():
            calls.append("second")
            second_started.set()
            return "second result"

        def pipeline():
            yield "claims"
            with request_batch([first, second], 1) as futures:
                for _, future in completed_requests(futures):
                    yield future.result()
            yield "complete"

        job = FactcheckJob(pipeline())
        try:
            self.assertEqual(self.next_event(job), "claims")
            pending = job.advance()
            self.assertTrue(first_started.wait(2))
            job.control.pause()
            release_first.set()
            self.assertEqual(pending.result(timeout=2), "first result")
            self.assertFalse(second_started.wait(0.05))
            self.assertIs(job.advance(), pending)
            self.assertEqual(job.consume(), "first result")
            job.control.resume()
            self.assertEqual(self.next_event(job), "second result")
            self.assertEqual(self.next_event(job), "complete")
            self.assertEqual(calls, ["first", "second"])
        finally:
            release_first.set()
            job.close()

    def test_closing_paused_job_releases_worker_without_starting_new_work(self):
        started = Event()

        def pipeline():
            started.set()
            yield "unexpected"

        job = FactcheckJob(pipeline())
        job.control.pause()
        future = job.advance()
        job.close()
        job.close()
        with self.assertRaises(ExecutionCancelled):
            future.result(timeout=2)
        self.assertFalse(started.is_set())

    def test_pausing_one_job_does_not_pause_another_job(self):
        first = FactcheckJob(iter(["first"]))
        second = FactcheckJob(iter(["second"]))
        try:
            first.control.pause()
            blocked = first.advance()
            self.assertEqual(self.next_event(second), "second")
            self.assertFalse(blocked.done())
            first.control.resume()
            self.assertEqual(self.next_event(first), "first")
        finally:
            first.close()
            second.close()

    def test_failure_wakes_paused_workers_and_propagates_without_deadlock(self):
        release_failure = Event()
        started = Event()

        def failing_request():
            started.set()
            if not release_failure.wait(3):
                raise TimeoutError("Failure was not released")
            raise ValueError("Request failed")

        def pipeline():
            with request_batch([failing_request, lambda: "queued"], 1) as futures:
                for _, future in completed_requests(futures):
                    yield future.result()

        job = FactcheckJob(pipeline())
        try:
            future = job.advance()
            self.assertTrue(started.wait(2))
            job.control.pause()
            release_failure.set()
            with self.assertRaisesRegex(ValueError, "Request failed"):
                future.result(timeout=2)
        finally:
            release_failure.set()
            job.close()
