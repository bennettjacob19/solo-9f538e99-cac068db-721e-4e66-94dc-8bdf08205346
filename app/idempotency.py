"""Per-workpiece idempotent inference bookkeeping.

Device networks retry: the same workpiece may be submitted several times over
the same TCP connection, from several concurrent connections, or minutes after
the first attempt timed out.  ``/infer/once`` must therefore treat
``(device_id, workpiece_id)`` as an idempotency key:

* the first accepted call pins the active model version and an inference id;
* identical features (same key) join that computation instead of starting a
  second one -- concurrently or after it finished;
* different features for the same key are a conflict;
* a completed result or error is replayed verbatim for as long as the record is
  retained (10 minutes by default);
* in-progress records never expire, completed ones do.

Everything here runs on the API's single asyncio event loop: the service is
the only process that owns the registry (enforced by the data-directory lock),
and each record's worker call is a single asyncio task.  The "look up existing
then create" sequence contains no ``await``, so concurrent requests cannot both
become the owner.  The store has its own re-entrant lock only because the sync
``/admin/status`` route reads its summary from FastAPI's threadpool.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

Key = Tuple[str, str]


class OnceConflict(RuntimeError):
    """The same device/workpiece was already accepted with different features."""

    def __init__(self, record: "OnceRecord", received_fingerprint: str) -> None:
        super().__init__(
            f"device {record.device_id!r} workpiece {record.workpiece_id!r} was "
            "already accepted with different features"
        )
        self.record = record
        self.received_fingerprint = received_fingerprint


def features_fingerprint(features: List[float]) -> str:
    """Canonical fingerprint of a numeric feature vector.

    JSON's shortest-round-trip float representation is used so ``0.1`` sent
    twice gives the same digest, and ``-0.0`` is normalised to ``0.0`` (they
    compare equal numerically).
    """
    normalised = [0.0 if value == 0 else float(value) for value in features]
    encoded = json.dumps(normalised, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class OnceRecord:
    """One accepted computation, shared by every retry of the same workpiece."""

    device_id: str
    workpiece_id: str
    features_fingerprint: str
    version: str
    inference_id: str
    created_at: float
    # Set when the underlying worker call settles, whatever the outcome.  A
    # record with ``completed_at is None`` is in progress and must not expire.
    completed_at: Optional[float] = None
    label: Optional[str] = None
    score: Optional[float] = None
    compute_ms: Optional[float] = None
    # Failure replayed verbatim on retry; no second dispatch is attempted.
    error_status: Optional[int] = None
    error_detail: Optional[Dict[str, Any]] = None
    # ``asyncio.Task`` that owns the single worker call.  Kept untyped so this
    # module stays independent of the event loop.
    task: Any = None

    @property
    def key(self) -> Key:
        return (self.device_id, self.workpiece_id)

    @property
    def finished(self) -> bool:
        return self.completed_at is not None

    @property
    def failed(self) -> bool:
        return self.error_status is not None

    def result_view(self) -> Dict[str, Any]:
        if self.label is None or self.score is None:
            raise RuntimeError("result requested before the computation completed")
        return {
            "label": self.label,
            "score": self.score,
            "compute_ms": float(self.compute_ms or 0.0),
        }


class OnceStore:
    """In-memory, retention-bounded map of workpiece computations.

    The mutating methods are called from the API's single event-loop thread,
    but ``summary``/``reap`` may be reached from FastAPI's threadpool (sync
    status route), so the store itself is guarded by a plain re-entrant lock.
    That lock never spans an ``await``.
    """

    def __init__(self, retention_seconds: float = 600.0, reap_interval: float = 1.0) -> None:
        self.retention_seconds = max(0.0, float(retention_seconds))
        self.reap_interval = max(0.0, float(reap_interval))
        self._records: Dict[Key, OnceRecord] = {}
        self._last_reap = 0.0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def get(self, device_id: str, workpiece_id: str) -> Optional[OnceRecord]:
        with self._lock:
            return self._records.get((device_id, workpiece_id))

    def lookup(
        self,
        device_id: str,
        workpiece_id: str,
        fingerprint: str,
    ) -> Optional[OnceRecord]:
        """Return the existing record for this key, or ``None``.

        Raises :class:`OnceConflict` if the key is already in use with
        different features -- even if its computation already finished.
        """
        with self._lock:
            record = self._records.get((device_id, workpiece_id))
            if record is None:
                return None
            if record.features_fingerprint != fingerprint:
                raise OnceConflict(record, fingerprint)
            return record

    def create(
        self,
        *,
        device_id: str,
        workpiece_id: str,
        fingerprint: str,
        version: str,
        inference_id: str,
    ) -> OnceRecord:
        """Register an accepted computation.

        The caller must have already checked :meth:`lookup`; the check and this
        call happen without yielding to the event loop, so no concurrent request
        can slip in between.
        """
        key = (device_id, workpiece_id)
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                raise RuntimeError(f"record for {key!r} already exists")
            record = OnceRecord(
                device_id=device_id,
                workpiece_id=workpiece_id,
                features_fingerprint=fingerprint,
                version=version,
                inference_id=inference_id,
                created_at=time.time(),
            )
            self._records[key] = record
            return record

    def discard(self, device_id: str, workpiece_id: str) -> None:
        """Forget a record that never got wired to a computation.

        Used only when setup fails between ``create`` and starting the task, so
        the key does not stay claimed by an in-progress record that can never
        finish (and therefore would never expire).
        """
        with self._lock:
            self._records.pop((device_id, workpiece_id), None)

    def reap(self, now: Optional[float] = None, force: bool = False) -> int:
        """Drop completed records older than the retention window.

        In-progress records are never touched, however old they are.  The
        throttle keeps per-request cleanup cheap; the background loop calls
        with ``force=True``.
        """
        current = time.time() if now is None else float(now)
        with self._lock:
            if not force and current - self._last_reap < self.reap_interval:
                return 0
            self._last_reap = current
            expired = [
                key
                for key, record in self._records.items()
                if record.completed_at is not None
                and current - record.completed_at >= self.retention_seconds
                and not (record.task is not None and not record.task.done())
            ]
            for key in expired:
                del self._records[key]
            return len(expired)

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for record in self._records.values() if not record.finished)

    def summary(self) -> Dict[str, Any]:
        """Small observability snapshot (never includes feature values)."""
        with self._lock:
            pending = completed = failed = 0
            for record in self._records.values():
                if not record.finished:
                    pending += 1
                else:
                    completed += 1
                    if record.failed:
                        failed += 1
            return {
                "retention_seconds": self.retention_seconds,
                "records": len(self._records),
                "in_progress": pending,
                "completed": completed - failed,
                "failed": failed,
            }
