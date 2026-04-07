import threading

from ..deepstream_pipeline.async_capture_worker import AsyncCaptureWorker


class CaptureProcessingService:
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
        with self._lock:
            worker = self._workers.pop(pipeline_id, None)

        if worker is not None:
            worker.stop()

    def stop_all(self):
        with self._lock:
            workers = list(self._workers.items())
            self._workers.clear()

        for _, worker in workers:
            worker.stop()
