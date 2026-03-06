import threading
import traceback
from typing import List, Optional
from ..pipeline_sync_notifier.pipeline_sync_notifier_interface import PipelineSyncNotifierInterface
from ...repositories.WorkerSourcePipelineRepository import WorkerSourcePipelineRepository
from ...models.worker_source_pipeline import WorkerSourcePipelineEntity

class PipelineSyncService:
    def __init__(self, pipeline_repository: WorkerSourcePipelineRepository):
        self._is_updating = False
        self._interval_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._update_subscribers: List[PipelineSyncNotifierInterface] = []
        self._update_lock = threading.Lock()

        self._pipeline_repository = pipeline_repository

        self._stop_event.clear()

        self._pipelines = []

        self._start_checking_interval_thread()

    def update_pipeline_status(self, pipeline_id: str, new_status: str):
        with self._update_lock:
            self._is_updating = True

            try:
                for i, _ in enumerate(self._pipelines):
                    if self._pipelines[i]["id"] == pipeline_id:
                        print(f"Updating local pipeline storage for id {pipeline_id} with {new_status}")
                        self._pipelines[i]["pipeline_status_code"] = new_status

                print(f"Updating pipeline db for id {pipeline_id} with {new_status}")
                self._pipeline_repository.update_pipeline_status(pipeline_id, new_status)
            finally:
                self._is_updating = False

    def subscribe_update(self, pipeline_subscriber: PipelineSyncNotifierInterface):
        self._update_subscribers.append(pipeline_subscriber)
    
    def stop(self):
        self._stop_event.set()
        if self._interval_thread is not None:
            self._interval_thread.join(timeout=10)

    def _start_checking_interval_thread(self):
        self._interval_thread = threading.Thread(target=self._interval_checking)
        self._interval_thread.start()
    
    def _interval_checking(self):
        while not self._stop_event.wait(timeout=5):
            if self._is_updating:
                continue

            try:
                # Fetch latest pipelines from DB
                latest_pipelines = self._get_normalized_pipelines()

                changes = self._check_difference(self._pipelines, latest_pipelines)

                if not changes:
                    continue

                # Update local snapshot FIRST to avoid duplicate notifications
                self._pipelines = latest_pipelines

                for change in changes:
                    pipeline_id = change["id"]
                    change_type = change["type"]

                    for subscriber in self._update_subscribers:
                        if self._stop_event.is_set():
                            continue

                        if change_type == "updated":
                            subscriber.notify_pipeline_update(pipeline_id)
                        elif change_type == "new":
                            subscriber.notify_new_pipeline(pipeline_id)
                        elif change_type == "deleted":
                            subscriber.notify_deleted_pipeline(pipeline_id)

            except Exception as e:
                traceback.print_exc() 
                print(f"Error during pipeline interval checking: {e}")

    def _check_difference(self, before, after):
        """
        Returns list of changes:
        [
            {"id": "xxx", "type": "updated"},
            {"id": "yyy", "type": "new"},
            {"id": "zzz", "type": "deleted"},
        ]
        """

        if before is None:
            before = []

        if after is None:
            after = []

        changes = []

        # Convert to dict for O(1) lookup
        before_dict = {
            p["id"]: p
            for p in before
        }

        after_dict = {
            p["id"]: p
            for p in after
        }

        before_ids = set(before_dict.keys())
        after_ids = set(after_dict.keys())

        # 🔹 Detect new pipelines
        for new_id in after_ids - before_ids:
            changes.append({"id": new_id, "type": "new"})

        # 🔹 Detect deleted pipelines
        for deleted_id in before_ids - after_ids:
            changes.append({"id": deleted_id, "type": "deleted"})

        # 🔹 Detect updated pipelines
        for common_id in before_ids & after_ids:
            if before_dict[common_id] != after_dict[common_id]:
                changes.append({"id": common_id, "type": "updated"})

        return changes

    def _normalize_pipeline(self, pipeline: WorkerSourcePipelineEntity):
        return {
            "id": pipeline.id,
            "name": pipeline.name,
            "worker_source_id": pipeline.worker_source_id,
            "worker_id": pipeline.worker_id,
            "ai_model_id": pipeline.ai_model_id,
            "pipeline_status_code": pipeline.pipeline_status_code,
            "location_name": pipeline.location_name,
            "last_preview_request_at": pipeline.last_preview_request_at,
        }
    
    def _get_normalized_pipelines(self):
        pipelines_db = self._pipeline_repository.get_all_pipelines()
        return [self._normalize_pipeline(p) for p in pipelines_db]