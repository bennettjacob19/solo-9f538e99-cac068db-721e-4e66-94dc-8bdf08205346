"""Unit tests for the /infer/once idempotency bookkeeping."""

from __future__ import annotations

import time

import pytest

from app.idempotency import (
    OnceConflict,
    OnceStore,
    features_fingerprint,
)


def _create(store: OnceStore, device="d1", piece="w1", features=(0.5, 1.5)):
    fp = features_fingerprint(list(features))
    return store.create(
        device_id=device,
        workpiece_id=piece,
        fingerprint=fp,
        version="v1",
        inference_id="inf-1",
    )


# -- fingerprint ------------------------------------------------------------


def test_fingerprint_is_stable_and_normalises_negative_zero():
    assert features_fingerprint([0.1, -0.0, 2]) == features_fingerprint([0.1, 0.0, 2.0])


def test_fingerprint_changes_with_features_and_length():
    base = features_fingerprint([1.0, 2.0])
    assert base != features_fingerprint([1.0, 2.0, 0.0])
    assert base != features_fingerprint([1.0, 2.0000001])
    assert base != features_fingerprint([2.0, 1.0])


# -- lookup / conflict ------------------------------------------------------


def test_lookup_returns_same_record_for_same_key_and_features():
    store = OnceStore()
    record = _create(store)
    assert store.lookup("d1", "w1", record.features_fingerprint) is record


def test_unknown_key_returns_none():
    store = OnceStore()
    assert store.lookup("d1", "w1", features_fingerprint([1.0])) is None


def test_duplicate_key_with_different_features_conflicts():
    store = OnceStore()
    record = _create(store, features=(1.0, 2.0))
    with pytest.raises(OnceConflict) as exc:
        store.lookup("d1", "w1", features_fingerprint([9.0, 9.0]))
    assert exc.value.record is record
    # Even after completion the key remains claimed until retention expires.
    record.completed_at = time.time()
    with pytest.raises(OnceConflict):
        store.lookup("d1", "w1", features_fingerprint([9.0, 9.0]))


def test_same_workpiece_on_a_different_device_is_independent():
    store = OnceStore()
    _create(store, device="line-a")
    store.lookup("line-b", "w1", features_fingerprint([0.5, 1.5])) is None
    assert len(store) == 1


def test_create_refuses_to_replace_an_existing_record():
    store = OnceStore()
    _create(store)
    with pytest.raises(RuntimeError):
        _create(store)


def test_discard_frees_an_unwired_record():
    store = OnceStore()
    _create(store)
    store.discard("d1", "w1")
    assert len(store) == 0
    # The key can be claimed again immediately.
    again = _create(store, features=(9.0, 9.0))
    assert again.inference_id == "inf-1"
    assert store.lookup("d1", "w1", again.features_fingerprint) is again


# -- retention --------------------------------------------------------------


def test_completed_records_expire_after_retention():
    store = OnceStore(retention_seconds=600, reap_interval=0)
    record = _create(store)
    done_at = time.time()
    record.completed_at = done_at
    assert store.reap(now=done_at + 599, force=True) == 0
    assert len(store) == 1
    assert store.reap(now=done_at + 600, force=True) == 1
    assert len(store) == 0


def test_in_progress_records_never_expire():
    store = OnceStore(retention_seconds=600, reap_interval=0)
    _create(store)  # completed_at stays None
    assert store.reap(now=time.time() + 10_000, force=True) == 0
    assert len(store) == 1


def test_reap_is_throttled_unless_forced():
    store = OnceStore(retention_seconds=0, reap_interval=60)
    record = _create(store)
    record.completed_at = time.time()
    now = time.time()
    assert store.reap(now=now) == 1  # first call establishes the window
    record2 = _create(store, piece="w2")
    record2.completed_at = time.time()
    assert store.reap(now=now + 1) == 0  # throttled
    assert len(store) == 1
    assert store.reap(now=now + 1, force=True) == 1


# -- error / summary --------------------------------------------------------


def test_failed_record_replays_its_error():
    store = OnceStore()
    record = _create(store)
    record.error_status = 502
    record.error_detail = {"error": "worker_error", "message": "boom"}
    record.completed_at = time.time()
    assert record.finished and record.failed
    assert record.error_detail["message"] == "boom"


def test_summary_counts_states():
    store = OnceStore()
    pending = _create(store, piece="p")
    done = _create(store, piece="d")
    done.completed_at = time.time()
    done.label, done.score = "ok", 1.0
    failed = _create(store, piece="f")
    failed.completed_at = time.time()
    failed.error_status = 502
    failed.error_detail = {"error": "x"}
    assert store.summary() == {
        "retention_seconds": 600.0,
        "records": 3,
        "in_progress": 1,
        "completed": 1,
        "failed": 1,
    }
    assert store.pending_count() == 1
    assert pending.finished is False


def test_record_key_and_result_view():
    record = _create(store=OnceStore())
    assert record.key == ("d1", "w1")
    with pytest.raises(RuntimeError):
        record.result_view()
    record.label, record.score, record.compute_ms = "dent", 0.25, 3.5
    assert record.result_view() == {"label": "dent", "score": 0.25, "compute_ms": 3.5}
