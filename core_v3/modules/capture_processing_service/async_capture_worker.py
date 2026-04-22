import copy
import logging
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2

from ...database.DatabaseManager import DatabaseManager
from ...repositories.PPEDetectionRepository import PPEDetectionRepository
from ..drawing.DrawingUtils import DrawingUtils

"""Async capture worker responsibilities.

This worker decouples heavy capture processing from the DeepStream thread by:
1) buffering recent RGBA frames keyed by PTS,
2) accepting capture events through an internal queue,
3) annotating/saving full + cropped images, and
4) persisting detection metadata to the repository.
"""


class AsyncCaptureWorker:
    """Background worker for capture event processing.

    Producer side:
    - `store_frame(...)` is called frequently from appsink callbacks.
    - `enqueue_capture(...)` is called when capture rules are triggered.

    Consumer side:
    - A dedicated daemon thread (`_run`) reads `_capture_queue` and runs
      `_process_capture(...)` for each event.
    """

    def __init__(
        self,
        pipeline_id: str,
        worker_id: str,
        worker_source_id: str,
        frame_drawer,
        max_cached_frames: int = 120,
    ):
        self.pipeline_id = pipeline_id
        self.worker_id = worker_id
        self.worker_source_id = worker_source_id
        self.frame_drawer = frame_drawer
        self.max_cached_frames = max_cached_frames

        self._frame_lock = threading.Lock()
        self._cached_frames = OrderedDict()
        self._capture_queue = queue.Queue(maxsize=256)
        self._running = True
        self._ppe_repository = PPEDetectionRepository()

        self._worker_thread = threading.Thread(
            target=self._run,
            name=f"capture-worker-{pipeline_id}",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self):
        """Stop worker thread gracefully by sending a sentinel queue item."""
        self._running = False
        try:
            self._capture_queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker_thread.join(timeout=2.0)

    def store_frame(self, pts: int, frame_rgba):
        """Cache latest frames by PTS for later capture event lookup."""
        if frame_rgba is None:
            return

        with self._frame_lock:
            self._cached_frames[pts] = frame_rgba
            self._cached_frames.move_to_end(pts)
            while len(self._cached_frames) > self.max_cached_frames:
                self._cached_frames.popitem(last=False)

    def enqueue_capture(self, pts: int, tracked_object: dict):
        """Enqueue capture work item; drop when queue is saturated."""
        payload = {
            "pts": pts,
            "tracked_object": copy.deepcopy(tracked_object),
            "timestamp_ms": int(time.time() * 1000),
        }
        try:
            self._capture_queue.put_nowait(payload)
        except queue.Full:
            logging.warning("Capture queue full for pipeline %s, dropping capture event", self.pipeline_id)

    def _run(self):
        """Consume queue items and process captures on the worker thread."""
        while self._running:
            item = self._capture_queue.get()
            if item is None:
                break

            try:
                self._process_capture(item)
            except Exception:
                logging.exception("Failed to process capture event for pipeline %s", self.pipeline_id)

    def _process_capture(self, payload: dict):
        """Render/save capture artifacts and persist detection metadata."""
        frame_rgba = self._get_frame(payload["pts"])
        if frame_rgba is None:
            logging.warning("No cached frame found for capture event on pipeline %s", self.pipeline_id)
            return

        tracked_object = payload["tracked_object"]
        frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)

        annotated_frame = self.frame_drawer.draw_frame(frame_bgr.copy(), [copy.deepcopy(tracked_object)])

        cropped_frame, cropped_object = DrawingUtils.crop_with_bounding_box(
            frame_bgr.copy(),
            copy.deepcopy(tracked_object),
        )
        annotated_crop = self.frame_drawer.draw_frame(cropped_frame, [cropped_object])

        output_dir = Path(DatabaseManager.STORAGE_PATHS["files"]) / "ppe_detections"
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{payload['timestamp_ms']}_{self.pipeline_id}_{tracked_object['track_id']}"
        full_path = output_dir / f"{stem}.jpg"
        crop_path = output_dir / f"{stem}_crop.jpg"

        cv2.imwrite(str(full_path), annotated_frame)
        cv2.imwrite(str(crop_path), annotated_crop)

        self._ppe_repository.save_ppe_detection(
            worker_id=self.worker_id,
            worker_source_id=self.worker_source_id,
            person_id=str(tracked_object["person_id"]),
            image_path=str(full_path),
            image_tile_path=str(crop_path),
            detection_count=int(tracked_object.get("detections", 0)),
            person_bbox=tracked_object["bbox"],
            attributes=tracked_object.get("attributes", []),
        )

    def _get_frame(self, pts: int):
        """Get matching frame by PTS, fallback to the most recent cached frame."""
        with self._frame_lock:
            frame = self._cached_frames.get(pts)
            if frame is not None:
                return frame.copy()

            if not self._cached_frames:
                return None

            latest_pts = next(reversed(self._cached_frames))
            return self._cached_frames[latest_pts].copy()
