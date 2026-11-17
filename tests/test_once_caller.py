"""Focused tests for the shared-task semantics of ``/infer/once``.

These drive ``InferenceService`` internals directly so they can prove, without
depending on how uvicorn handles a dropped connection, that a caller's
timeout/cancellation cannot cancel the one worker computation, cannot decrement
its in-flight count early, and cannot disturb a second caller waiting on the
same record.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi import HTTPException

from app.idempotency import OnceRecord, OnceStore, features_fingerprint
from app.main import InferenceService


class _CountingRegistry:
    def __init__(self) -> None:
        self.completions = 0

    def complete_work(self, version: str) -> None:
        self.completions += 1


def _service() -> InferenceService:
    service = InferenceService.__new__(InferenceService)  # skip process startup
    service.once = OnceStore(retention_seconds=600.0)
    service.registry = _CountingRegistry()
    return service


def _record(store: OnceStore, key=("d", "w")) -> OnceRecord:
    record = store.create(
        device_id=key[0],
        workpiece_id=key[1],
        fingerprint=features_fingerprint([1.0, 2.0]),
        version="v1",
        inference_id="inf-1",
    )
    return record


async def _finish(record: OnceRecord, delay: float, label: str = "ok") -> None:
    await asyncio.sleep(delay)
    record.label = label
    record.score = 0.5
    record.compute_ms = delay * 1000.0
    record.completed_at = time.time()


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_cancel_shared_computation():
    service = _service()
    record = _record(service.once)
    record.task = asyncio.create_task(_finish(record, 0.4))

    caller = asyncio.create_task(
        service._await_once(record, 10_000, time.perf_counter(), True, "joined")
    )
    await asyncio.sleep(0.05)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    # The underlying computation and the record are untouched.
    assert not record.task.done()
    assert not record.finished
    assert record.error_status is None

    await record.task
    assert record.finished
    assert record.label == "ok"


@pytest.mark.asyncio
async def test_one_caller_timeout_does_not_disturb_another():
    service = _service()
    record = _record(service.once)
    record.task = asyncio.create_task(_finish(record, 0.3))

    with pytest.raises(HTTPException) as exc:
        await service._await_once(record, 50, time.perf_counter(), True, "joined")
    assert exc.value.status_code == 504
    assert exc.value.detail["inference_id"] == "inf-1"
    assert exc.value.detail["record_state"] == "in_progress"

    # A later caller joins the same task and gets the result.
    result = await service._await_once(record, 5_000, time.perf_counter(), True, "joined")
    assert result.inference_id == "inf-1"
    assert result.label == "ok"
    assert result.waited is True
    assert result.record_state == "joined"


@pytest.mark.asyncio
async def test_timeout_after_settlement_replays_instead_of_504():
    """A wait expiring in the same tick the work finished replays the result."""
    service = _service()
    record = _record(service.once)
    record.task = asyncio.create_task(_finish(record, 0.0))
    await asyncio.sleep(0.02)  # let the task settle before the wait starts

    result = await service._await_once(record, 1, time.perf_counter(), False, "computed")
    assert result.label == "ok"
    assert result.record_state == "computed"


@pytest.mark.asyncio
async def test_failed_record_replays_error_without_dispatch():
    service = _service()
    record = _record(service.once)
    record.error_status = 502
    record.error_detail = {
        "error": "worker_error",
        "version": "v1",
        "inference_id": "inf-1",
        "message": "boom",
    }

    async def fail() -> None:
        record.completed_at = time.time()

    record.task = asyncio.create_task(fail())
    for _ in range(2):
        with pytest.raises(HTTPException) as exc:
            await service._await_once(record, 5_000, time.perf_counter(), True, "joined")
        assert exc.value.status_code == 502
        assert exc.value.detail == record.error_detail
    # No dispatch ever happened through _run_once, so nothing completed.
    assert service.registry.completions == 0


@pytest.mark.asyncio
async def test_shutdown_cancellation_stores_error_and_does_not_hang():
    """When shutdown cancels the shared task, the record still settles."""
    service = _service()
    record = _record(service.once)

    async def fail() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            record.error_status = 503
            record.error_detail = {
                "error": "service_shutting_down",
                "version": "v1",
                "inference_id": "inf-1",
                "message": "stopped",
            }
            raise
        finally:
            record.completed_at = time.time()
        record.label = "never"

    record.task = asyncio.create_task(fail())
    await asyncio.sleep(0.02)
    wait = asyncio.create_task(
        service._await_once(record, 5_000, time.perf_counter(), True, "joined")
    )
    await asyncio.sleep(0.02)
    record.task.cancel()  # what InferenceService.shutdown() does
    # The waiter did not cancel itself, so it reports the stored shutdown
    # error (a clean 503) instead of a bare CancelledError.
    with pytest.raises(HTTPException) as first:
        await wait
    assert first.value.status_code == 503
    assert record.finished
    assert record.error_status == 503
    # A retry before the process actually exits replays the stored error.
    with pytest.raises(HTTPException) as retry:
        await service._await_once(record, 5_000, time.perf_counter(), True, "completed")
    assert retry.value.status_code == 503
    assert retry.value.detail["error"] == "service_shutting_down"


@pytest.mark.asyncio
async def test_in_flight_released_once_when_worker_settles(monkeypatch):
    """``_run_once`` clears the registry count exactly at worker completion."""

    class _Resp:
        status_code = 200

        def json(self):
            return {"label": "pass", "score": 1.0}

    class _Client:
        def __init__(self) -> None:
            self.calls = 0

        async def post(self, url, json=None):  # noqa: A002 - httpx signature
            self.calls += 1
            await asyncio.sleep(0.2)
            return _Resp()

    class _Handle:
        def inference_url(self) -> str:
            return "http://worker/infer"

    service = _service()
    service.client = _Client()
    record = _record(service.once)
    await service._run_once(record, _Handle(), {"features": [1.0, 2.0]})

    assert service.client.calls == 1
    assert record.label == "pass"
    assert record.finished
    assert service.registry.completions == 1
