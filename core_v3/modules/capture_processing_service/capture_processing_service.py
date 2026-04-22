import threading

from .async_capture_worker import AsyncCaptureWorker

"""Capture processing service responsibilities.

This service owns the lifecycle of `AsyncCaptureWorker` instances:
1) create one worker per pipeline on demand,
2) return existing worker when already created,
3) stop/remove a single worker, and
4) stop/remove all workers during shutdown.
"""


class CaptureProcessingService:
    """Thread-safe registry and lifecycle manager for capture workers."""

    def __init__(self):
        self._workers = {}
        self._lock = threading.Lock()

    def get_or_create_worker(
        self,
        pipeline_id: str,
        worker_id: str,
        worker_source_id: str,
        frame_drawer,
    ) -> AsyncCaptureWorker:
        """Return existing worker for a pipeline or create a new one."""
        with self._lock:
            worker = self._workers.get(pipeline_id)
            if worker is None:
                worker = AsyncCaptureWorker(
                    pipeline_id=pipeline_id,
                    worker_id=worker_id,
                    worker_source_id=worker_source_id,
                    frame_drawer=frame_drawer,
                )
                self._workers[pipeline_id] = worker
            return worker

    def stop_worker(self, pipeline_id: str):
        """Stop and remove one worker by pipeline id if present."""
        with self._lock:
            worker = self._workers.pop(pipeline_id, None)

        if worker is not None:
            worker.stop()

    def stop_all(self):
        """Stop and remove all workers safely."""
        with self._lock:
            workers = list(self._workers.items())
            self._workers.clear()

        for _, worker in workers:
            worker.stop()
