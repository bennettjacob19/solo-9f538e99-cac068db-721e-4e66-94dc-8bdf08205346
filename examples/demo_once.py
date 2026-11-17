#!/usr/bin/env python3
"""End-to-end demo of ``POST /infer/once`` against a live API.

Usage::

    python examples/demo_once.py [BASE_URL]
    # inside Docker:  python examples/demo_once.py http://127.0.0.1:8000

Demonstrates the device-retry story:

0. load a model (unless one is already active);
1. a slow workpiece call whose caller gives up -> 504 with an inference id;
2. the identical retry (same device/workpiece/features) -> same inference id,
   label and score, with a longer per-caller timeout;
3. the same workpiece with different features -> 409, never dispatched;
4. a second workpiece is accepted, then a model swap happens while it is still
   computing: the retry still returns the *original* version's judgement;
5. completed records replay, and new traffic uses the new version.
"""

from __future__ import annotations

import json
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
LABELS = ["pass", "scratch", "dent"]
W1 = [[1.2, -0.4, 0.0, 0.3], [-0.2, 0.9, 0.5, -0.1], [0.1, -0.8, 1.1, 0.2]]
W2 = [[0.3, 1.1, -0.2, 0.1], [1.3, -0.3, 0.1, -0.2], [-0.4, 0.2, 0.9, 0.7]]
FEATURES = [0.7, 0.2, 0.1, 0.5]

client = httpx.Client(base_url=BASE, timeout=60.0)


def show(label: str, resp: httpx.Response) -> httpx.Response:
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 - demo formatting
        body = resp.text
    print(f"  {label:<46} {resp.status_code}  {json.dumps(body)[:200]}")
    return resp


def status() -> dict:
    st = client.get("/admin/status").json()
    layout = [
        (s["slot_id"], s["version"], s["process_state"], s["in_flight"])
        for s in st["slots"]
    ]
    print(
        f"  status: active={st['active_version']} slots={layout} "
        f"once={st['once_records']}"
    )
    return st


def upload(version: str, weights) -> httpx.Response:
    return client.post(
        "/admin/models",
        json={
            "version": version,
            "weights": weights,
            "bias": [0.0, 0.0, 0.0],
            "labels": LABELS,
        },
    )


def once(device: str, piece: str, features, timeout_ms: int, slow_ms: int = 0):
    return client.post(
        "/infer/once",
        json={
            "device_id": device,
            "workpiece_id": piece,
            "features": features,
            "timeout_ms": timeout_ms,
            "slow_ms": slow_ms,
        },
    )


def wait_idle_slot(timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if status()["slots_in_use"] == 1:
            return
        time.sleep(0.3)


def main() -> int:
    stamp = str(int(time.time()))
    device = "line-3-station-7"
    piece = f"WP-{stamp}"

    print("\n== 0. ensure a model is active ==")
    v1 = f"once-demo-v1-{stamp}"
    resp = upload(v1, W1)
    if resp.status_code == 409:  # rerun against the same data directory
        show("upload v1 (already present, using active)", resp)
        v1 = status()["active_version"]
    else:
        show("upload v1", resp)
    status()

    print("\n== 1. slow workpiece, caller times out (504), computation continues ==")
    first = show(
        "infer/once slow_ms=3000 timeout_ms=150",
        once(device, piece, FEATURES, timeout_ms=150, slow_ms=3000),
    )
    first_id = first.json()["detail"]["inference_id"]
    status()  # in_flight=1, once_records.in_progress=1

    print("\n== 2. identical retry retrieves the same computation ==")
    retry = show(
        "infer/once same key, timeout_ms=15000",
        once(device, piece, FEATURES, timeout_ms=15_000),
    )
    body = retry.json()
    print(
        f"  same inference_id={body['inference_id'] == first_id}  "
        f"label={body['label']}  waited={body['waited']}  "
        f"state={body['record_state']}"
    )
    status()

    print("\n== 3. same workpiece, different features -> 409, not dispatched ==")
    show("infer/once altered features", once(device, piece, [0.1, 0.2, 0.3, 0.4], 5000))
    status()

    print("\n== 4. swap between acceptance and retry keeps the original version ==")
    piece2 = f"WP-{stamp}-B"
    accepted = show(
        "infer/once piece B, timeout_ms=150",
        once(device, piece2, FEATURES, timeout_ms=150, slow_ms=3000),
    )
    piece2_id = accepted.json()["detail"]["inference_id"]
    v2 = f"once-demo-v2-{stamp}"
    show("upload v2 (hot swap while B computes)", upload(v2, W2))
    status()
    show(
        "plain /infer already uses the new version",
        client.post("/infer", json={"features": FEATURES}),
    )
    piece2_retry = show(
        "infer/once piece B retry", once(device, piece2, FEATURES, timeout_ms=15_000)
    )
    b = piece2_retry.json()
    print(
        f"  original inference_id preserved={b['inference_id'] == piece2_id}  "
        f"version={b['version']}  label={b['label']}"
    )

    print("\n== 5. completed records replay; wait for the old slot to drain ==")
    show("piece A replay", once(device, piece, FEATURES, timeout_ms=5000))
    wait_idle_slot()
    status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
