"""Integration tests for ``POST /infer/once`` against real worker processes.

These cover the behaviour the production line needs when its network flaps:

* one computation per (device, workpiece), even for concurrent first attempts;
* retries after a 504 (or a real client disconnect) retrieve that computation;
* a version swap between acceptance and retry does not change the judgement;
* a successful or failed record is replayed, never re-dispatched;
* a finished record is retained for the configured window, an in-progress one
  is retained indefinitely.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent

W_A = [[1.0, 0.0], [0.0, 1.0]]
W_B = [[0.0, 1.0], [1.0, 0.0]]
LABELS = ["defect-A", "defect-B"]
FEATURES = [0.9, 0.1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ApiServer:
    def __init__(self, data_dir: Path, **env_overrides: str) -> None:
        self.data_dir = data_dir
        self.env_overrides = env_overrides
        self.port = _free_port()
        self.proc: subprocess.Popen | None = None
        self.log = open(data_dir / "api.log", "w", encoding="utf-8")

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "ApiServer":
        env = {
            **os.environ,
            "DATA_DIR": str(self.data_dir),
            "LOG_LEVEL": "WARNING",
            **self.env_overrides,
        }
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=str(ROOT),
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("API exited during startup:\n" + self._tail())
            try:
                if httpx.get(f"{self.base}/healthz", timeout=1.0).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        raise TimeoutError("API did not start:\n" + self._tail())

    def __exit__(self, *exc) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), 15)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), 9)
                self.proc.wait(10)
        subprocess.run(
            ["pkill", "-f", f"{self.data_dir}/workers"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.log.close()

    def _tail(self) -> str:
        self.log.flush()
        return self.log.read_text()[-4000:] if self.log.name else ""


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("qcdata-once")
    # A short drain wait keeps the capacity assertion in test_05 snappy.
    with ApiServer(data_dir, SWAP_WAIT_SECONDS="1") as server:
        yield server
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def short_retention_api(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("qcdata-once-short")
    with ApiServer(data_dir, ONCE_RETENTION_SECONDS="1") as server:
        yield server
    shutil.rmtree(data_dir, ignore_errors=True)


def upload(base: str, version: str, weights, **extra) -> httpx.Response:
    return httpx.post(
        f"{base}/admin/models",
        json={
            "version": version,
            "weights": weights,
            "bias": [0.0, 0.0],
            "labels": LABELS,
            **extra,
        },
        timeout=60.0,
    )


def once(
    base: str,
    device: str,
    piece: str,
    features=FEATURES,
    timeout_ms: int | None = 10_000,
    slow_ms: int = 0,
    client_timeout: float = 60.0,
) -> httpx.Response:
    body = {
        "device_id": device,
        "workpiece_id": piece,
        "features": features,
        "slow_ms": slow_ms,
    }
    if timeout_ms is not None:
        body["timeout_ms"] = timeout_ms
    return httpx.post(f"{base}/infer/once", json=body, timeout=client_timeout)


def status(base: str) -> dict:
    return httpx.get(f"{base}/admin/status", timeout=5.0).json()


def wait_for(predicate, timeout=15.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met in time")


def accepted(base: str) -> int:
    return int(status(base)["total_accepted"])


def unique(tag: str) -> str:
    return f"{tag}-{uuid.uuid4().hex[:10]}"


def test_00_bootstrap_model(api):
    resp = upload(api.base, "once-v1", W_A)
    assert resp.status_code == 201, resp.text

def test_01_first_call_computes_and_retry_replays(api):
    key = unique("piece")
    before = accepted(api.base)

    first = once(api.base, "line-1", key)
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["version"] == "once-v1"
    assert body["label"] == "defect-A"
    assert body["score"] == pytest.approx(0.9)
    assert body["waited"] is False
    assert body["record_state"] == "computed"
    assert body["inference_id"]
    assert body["retained_until"] > time.time() + 590

    after_first = accepted(api.base)
    assert after_first == before + 1

    # Same features, later retry: same result/id, no second dispatch.
    retry = once(api.base, "line-1", key)
    assert retry.status_code == 200, retry.text
    replay = retry.json()
    assert replay["inference_id"] == body["inference_id"]
    assert replay["label"] == body["label"]
    assert replay["score"] == pytest.approx(body["score"])
    assert replay["waited"] is True
    assert replay["record_state"] == "completed"
    assert accepted(api.base) == after_first

    st = status(api.base)
    assert st["once_records"]["retention_seconds"] == 600.0
    assert st["once_records"]["completed"] >= 1


def test_02_same_workpiece_different_device_is_separate(api):
    piece = unique("shared-piece")
    a = once(api.base, "line-a", piece)
    b = once(api.base, "line-b", piece)
    assert a.status_code == 200 and b.status_code == 200
    assert a.json()["inference_id"] != b.json()["inference_id"]


def test_03_different_features_conflict_without_dispatch(api):
    key = unique("piece")
    first = once(api.base, "line-1", key, features=[0.9, 0.1])
    assert first.status_code == 200
    before = accepted(api.base)

    conflict = once(api.base, "line-1", key, features=[0.1, 0.9])
    assert conflict.status_code == 409, conflict.text
    detail = conflict.json()["detail"]
    assert detail["error"] == "features_conflict"
    assert detail["inference_id"] == first.json()["inference_id"]
    assert detail["record_state"] == "completed"
    assert accepted(api.base) == before  # nothing dispatched

    # The original features still retrieve the original judgement.
    again = once(api.base, "line-1", key, features=[0.9, 0.1])
    assert again.status_code == 200
    assert again.json()["inference_id"] == first.json()["inference_id"]


def test_04_concurrent_first_calls_share_one_computation(api):
    key = unique("piece")
    before = accepted(api.base)

    def call() -> dict:
        resp = once(api.base, "line-1", key, timeout_ms=15_000, slow_ms=1200)
        return {"code": resp.status_code, "body": resp.json()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(call) for _ in range(6)]
        results = [f.result(timeout=30) for f in futures]

    assert all(r["code"] == 200 for r in results), results
    ids = {r["body"]["inference_id"] for r in results}
    assert len(ids) == 1, results
    # Exactly one request won admission; the rest joined it.
    assert sum(1 for r in results if r["body"]["waited"] is False) == 1
    assert accepted(api.base) == before + 1


def test_05_per_caller_timeout_then_swap_then_retry_original_version(api):
    key = unique("piece")
    before = accepted(api.base)

    # This caller gives up well before the computation ends.
    timed_out = once(api.base, "line-2", key, timeout_ms=150, slow_ms=2500)
    assert timed_out.status_code == 504, timed_out.text
    detail = timed_out.json()["detail"]
    assert detail["error"] == "inference_timeout"
    assert detail["version"] == "once-v1"
    assert detail["record_state"] == "in_progress"
    inference_id = detail["inference_id"]
    assert inference_id

    st = status(api.base)
    assert st["total_in_flight"] == 1  # abandoned caller does not cancel work
    assert st["once_records"]["in_progress"] == 1
    assert accepted(api.base) == before + 1

    # Swap while the original computation is still running. The old version
    # goes to draining but keeps its in-flight call and slot.
    swapped = upload(api.base, "once-v2", W_B)
    assert swapped.status_code == 201, swapped.text
    st = status(api.base)
    assert st["active_version"] == "once-v2"
    states = {s["version"]: s["process_state"] for s in st["slots"] if s["version"]}
    assert states.get("once-v1") == "draining"

    # Retrying the same workpiece returns the *original* version's judgement
    # and the same inference id, not the new version's.
    retry = once(api.base, "line-2", key, timeout_ms=20_000)
    assert retry.status_code == 200, retry.text
    body = retry.json()
    assert body["inference_id"] == inference_id
    assert body["version"] == "once-v1"
    assert body["label"] == "defect-A"  # v1 weights, not v2's defect-B
    assert body["score"] == pytest.approx(0.9)
    assert body["waited"] is True
    assert accepted(api.base) == before + 1

    # New traffic uses the new version.
    plain = httpx.post(f"{api.base}/infer", json={"features": FEATURES}, timeout=10)
    assert plain.status_code == 200
    assert plain.json()["version"] == "once-v2"
    assert plain.json()["label"] == "defect-B"

    # The old slot is released once the shared computation has settled.
    wait_for(lambda: status(api.base)["slots_in_use"] == 1, timeout=15)


def test_06_client_disconnect_does_not_cancel_or_redispatch(api):
    # This test may run in isolation; make sure a model is active.
    if status(api.base)["active_version"] is None:
        assert upload(api.base, "disconnect-v1", W_A).status_code == 201

    key = unique("piece")
    before = accepted(api.base)

    # Open a raw socket, send the request, then close it while the computation
    # is still running.  This guarantees the API observes a real client
    # disconnect (an httpx timeout races the response on a fast loopback).
    body = json.dumps(
        {
            "device_id": "line-3",
            "workpiece_id": key,
            "features": FEATURES,
            "slow_ms": 1500,
            "timeout_ms": 10_000,
        }
    ).encode()
    sock = socket.create_connection(("127.0.0.1", api.port), timeout=2)
    sock.sendall(
        b"POST /infer/once HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    time.sleep(0.2)  # let the server accept and start the computation
    sock.close()  # disconnect without reading the response

    wait_for(lambda: accepted(api.base) == before + 1, timeout=5)
    st = status(api.base)
    assert st["total_in_flight"] == 1
    assert st["once_records"]["in_progress"] == 1

    # The record kept computing to completion despite the disconnect.
    retry = wait_for(
        lambda: (
            r
            if (r := once(api.base, "line-3", key, timeout_ms=20_000)).status_code == 200
            else None
        ),
        timeout=15,
    )
    assert retry is not None
    body = retry.json()
    assert body["waited"] is True
    # It either joined the still-running computation or read it after it settled
    # while the retry was queued; either way it is the same underlying record.
    assert body["record_state"] in {"joined", "completed"}
    assert body["compute_ms"] >= 1000  # the 1500ms work really ran once
    assert accepted(api.base) == before + 1
    st = status(api.base)
    assert st["total_in_flight"] == 0
    assert st["once_records"]["in_progress"] == 0


def test_07_worker_failure_is_replayed_not_redispatched(api):
    # This test may run in isolation; make sure a model is active.
    if status(api.base)["active_version"] is None:
        assert upload(api.base, "failure-v1", W_A).status_code == 201

    key = unique("piece")
    before = accepted(api.base)
    before_completed = status(api.base)["total_completed"]

    # Start a slow computation, then kill the worker underneath it.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            once, api.base, "line-4", key, timeout_ms=10_000, slow_ms=15_000
        )
        wait_for(
            lambda: status(api.base)["total_in_flight"] == 1
            or status(api.base)["total_completed"] > before_completed,
            timeout=5,
        )
        pid = next(
            s["pid"]
            for s in status(api.base)["slots"]
            if s["process_state"] == "active" and s["pid"]
        )
        os.kill(pid, signal.SIGKILL)
        result = future.result(timeout=30)

    assert result.status_code == 502, result.text
    first_detail = result.json()["detail"]
    assert first_detail["error"] in {"worker_unreachable", "worker_error"}
    inference_id = first_detail["inference_id"]
    assert inference_id
    assert accepted(api.base) == before + 1

    # Retry replays the same error and id; no second dispatch.
    retry = once(api.base, "line-4", key, timeout_ms=15_000)
    assert retry.status_code == 502, retry.text
    assert retry.json()["detail"] == first_detail
    assert accepted(api.base) == before + 1
    st = status(api.base)
    assert st["active_version"] is None  # the killed version is gone, not replaced
    assert st["once_records"]["failed"] >= 1


def test_08_validation(api):
    missing = httpx.post(
        f"{api.base}/infer/once",
        json={"device_id": "d", "features": [1.0, 2.0]},
        timeout=5,
    )
    assert missing.status_code == 422

    blank = httpx.post(
        f"{api.base}/infer/once",
        json={"device_id": "  ", "workpiece_id": "w", "features": [1.0, 2.0]},
        timeout=5,
    )
    assert blank.status_code == 422

    empty = httpx.post(
        f"{api.base}/infer/once",
        json={"device_id": "d", "workpiece_id": "w", "features": []},
        timeout=5,
    )
    assert empty.status_code == 422

    # JSON parsers accept NaN; the API must reject it cleanly (422, not a 500
    # from trying to serialise NaN back in a validation body).
    non_finite = httpx.post(
        f"{api.base}/infer/once",
        content=b'{"device_id":"d","workpiece_id":"w","features":[NaN]}',
        headers={"content-type": "application/json"},
        timeout=5,
    )
    assert non_finite.status_code == 422
    assert non_finite.json()["detail"]["error"] == "invalid_features"


# -- retention (separate API with a 1s window) ------------------------------


def test_09_completed_record_expires_but_in_progress_does_not(short_retention_api):
    base = short_retention_api.base
    assert upload(base, "short-v1", W_A).status_code == 201

    done_key = unique("piece")
    first = once(base, "line-s", done_key)
    assert first.status_code == 200
    first_id = first.json()["inference_id"]

    # In progress across the retention boundary: still joined, same id.
    slow_key = unique("piece")
    timed_out = once(base, "line-s", slow_key, timeout_ms=100, slow_ms=2200)
    assert timed_out.status_code == 504
    slow_id = timed_out.json()["detail"]["inference_id"]
    time.sleep(1.4)  # > retention, but the record is still in progress
    st = status(base)
    assert st["once_records"]["in_progress"] >= 1

    joined = once(base, "line-s", slow_key, timeout_ms=15_000)
    assert joined.status_code == 200, joined.text
    assert joined.json()["inference_id"] == slow_id
    assert joined.json()["waited"] is True

    # The completed record has now outlived the 1s window, so the key is free:
    # a retry is a fresh computation with a new inference id.
    time.sleep(1.4)
    fresh = once(base, "line-s", done_key)
    assert fresh.status_code == 200
    assert fresh.json()["inference_id"] != first_id
    assert fresh.json()["waited"] is False
    assert fresh.json()["record_state"] == "computed"
