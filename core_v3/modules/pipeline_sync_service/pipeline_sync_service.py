import threading
import traceback
from typing import Dict, List, Optional

from ..pipeline_sync_notifier.pipeline_sync_notifier_interface import PipelineSyncNotifierInterface
from ...models.worker_source_pipeline import WorkerSourcePipelineEntity
from ...repositories.AIModelRepository import AIModelRepository
from ...repositories.WorkerSourcePipelineRepository import WorkerSourcePipelineRepository
from ...repositories.WorkerSourceRepository import WorkerSourceRepository


class PipelineSyncService:
    def __init__(
        self,
        pipeline_repository: WorkerSourcePipelineRepository,
        source_repository: WorkerSourceRepository | None = None,
        ai_model_repository: AIModelRepository | None = None,
    ):
        self._is_updating = False
        self._interval_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._update_subscribers: List[PipelineSyncNotifierInterface] = []
        self._update_lock = threading.Lock()

        self._pipeline_repository = pipeline_repository
        self._source_repository = source_repository or WorkerSourceRepository()
        self._ai_model_repository = ai_model_repository or AIModelRepository()

        self._stop_event.clear()

        self._pipelines: List[Dict] = []
        self._pipelines_by_id: Dict[str, Dict] = {}
        self._last_changes_by_id: Dict[str, Dict] = {}

        self._start_checking_interval_thread()

    def update_pipeline_status(self, pipeline_id: str, new_status: str):
        with self._update_lock:
            self._is_updating = True

            try:
                snapshot = self._pipelines_by_id.get(pipeline_id)
                if snapshot:
                    print(f"Updating local pipeline storage for id {pipeline_id} with {new_status}")
                    snapshot["pipeline"]["pipeline_status_code"] = new_status

                print(f"Updating pipeline db for id {pipeline_id} with {new_status}")
                self._pipeline_repository.update_pipeline_status(pipeline_id, new_status)
            finally:
                self._is_updating = False

    def subscribe_update(self, pipeline_subscriber: PipelineSyncNotifierInterface):
        self._update_subscribers.append(pipeline_subscriber)

    def get_pipeline_snapshot(self, pipeline_id: str) -> Optional[Dict]:
        return self._pipelines_by_id.get(pipeline_id)

    def get_last_change(self, pipeline_id: str) -> Optional[Dict]:
        return self._last_changes_by_id.get(pipeline_id)

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
                latest_pipelines = self._get_enriched_pipelines()
                changes = self._check_difference(self._pipelines, latest_pipelines)

                if not changes:
                    continue

                self._pipelines = latest_pipelines
                self._pipelines_by_id = {pipeline["pipeline_id"]: pipeline for pipeline in latest_pipelines}
                self._last_changes_by_id = {
                    change["id"]: change
                    for change in changes
                    if change["type"] != "deleted"
                }

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

    def _check_difference(self, before: List[Dict] | None, after: List[Dict] | None) -> List[Dict]:
        if before is None:
            before = []

        if after is None:
            after = []

        changes = []

        before_dict = {p["pipeline_id"]: p for p in before}
        after_dict = {p["pipeline_id"]: p for p in after}

        before_ids = set(before_dict.keys())
        after_ids = set(after_dict.keys())

        for new_id in after_ids - before_ids:
            changes.append(self._build_new_change(new_id, after_dict[new_id]))

        for deleted_id in before_ids - after_ids:
            changes.append({"id": deleted_id, "type": "deleted"})

        for common_id in before_ids & after_ids:
            if before_dict[common_id] != after_dict[common_id]:
                changes.append(
                    self._build_update_change(
                        pipeline_id=common_id,
                        before_snapshot=before_dict[common_id],
                        after_snapshot=after_dict[common_id],
                    )
                )

        return changes

    def _build_new_change(self, pipeline_id: str, snapshot: Dict) -> Dict:
        source_status = ((snapshot.get("source") or {}).get("status_code") or "").strip().lower()
        return {
            "id": pipeline_id,
            "type": "new",
            "reasons": ["pipeline_created"],
            "requires_restart": False,
            "source_stopped": source_status == "stopped",
            "status_only": False,
        }

    def _build_update_change(self, pipeline_id: str, before_snapshot: Dict, after_snapshot: Dict) -> Dict:
        reasons: List[str] = []
        requires_restart = False
        status_only = False

        before_pipeline = before_snapshot.get("pipeline") or {}
        after_pipeline = after_snapshot.get("pipeline") or {}
        before_source = before_snapshot.get("source") or {}
        after_source = after_snapshot.get("source") or {}
        before_model = before_snapshot.get("ai_model") or {}
        after_model = after_snapshot.get("ai_model") or {}
        before_configs = before_snapshot.get("pipeline_configs") or {}
        after_configs = after_snapshot.get("pipeline_configs") or {}

        for key in sorted(set(before_pipeline.keys()) | set(after_pipeline.keys())):
            if before_pipeline.get(key) == after_pipeline.get(key):
                continue
            reasons.append(f"pipeline.{key}")
            if key != "pipeline_status_code":
                requires_restart = True

        for key in sorted(set(before_source.keys()) | set(after_source.keys())):
            if before_source.get(key) == after_source.get(key):
                continue
            reasons.append(f"source.{key}")
            if key != "status_code":
                requires_restart = True

        for key in sorted(set(before_model.keys()) | set(after_model.keys())):
            if before_model.get(key) == after_model.get(key):
                continue
            reasons.append(f"ai_model.{key}")
            requires_restart = True

        if before_configs != after_configs:
            reasons.append("pipeline_configs")
            requires_restart = True

        status_only = bool(reasons) and all(reason == "pipeline.pipeline_status_code" for reason in reasons)
        source_status = (after_source.get("status_code") or "").strip().lower()

        return {
            "id": pipeline_id,
            "type": "updated",
            "reasons": reasons,
            "requires_restart": requires_restart,
            "source_stopped": source_status == "stopped",
            "status_only": status_only,
        }

    def _normalize_pipeline(self, pipeline: WorkerSourcePipelineEntity):
        return {
            "id": pipeline.id,
            "name": pipeline.name,
            "worker_source_id": pipeline.worker_source_id,
            "worker_id": pipeline.worker_id,
            "ai_model_id": pipeline.ai_model_id,
            "pipeline_status_code": pipeline.pipeline_status_code,
            "location_name": pipeline.location_name,
        }

    def _normalize_source(self, source) -> Optional[Dict]:
        if source is None:
            return None

        return {
            "id": source.id,
            "name": source.name,
            "worker_id": source.worker_id,
            "type_code": source.type_code,
            "file_path": source.file_path,
            "url": source.url,
            "resolution": source.resolution,
            "status_code": source.status_code,
            "frame_rate": source.frame_rate,
            "source_location_code": source.source_location_code,
            "latitude": source.latitude,
            "longitude": source.longitude,
        }

    def _normalize_ai_model(self, ai_model) -> Optional[Dict]:
        if ai_model is None:
            return None

        return {
            "id": ai_model.id,
            "file": ai_model.file,
            "type": ai_model.type,
            "name": ai_model.name,
            "version": ai_model.version,
            "classes": ai_model.get_classes(),
            "ppe_class_groups": ai_model.get_ppe_class_groups(),
            "main_class": ai_model.get_main_class(),
        }

    def _get_enriched_pipelines(self) -> List[Dict]:
        pipelines_db = self._pipeline_repository.get_all_pipelines()
        pipeline_configs_by_id = self._pipeline_repository.get_all_pipeline_configs_grouped()
        sources_by_id = {
            source.id: source
            for source in self._source_repository.get_worker_sources()
        }
        ai_models_by_id = {
            model.id: model
            for model in self._ai_model_repository.get_ai_models()
        }

        enriched = []
        for pipeline in pipelines_db:
            normalized_pipeline = self._normalize_pipeline(pipeline)
            source = sources_by_id.get(pipeline.worker_source_id)
            ai_model = ai_models_by_id.get(pipeline.ai_model_id) if pipeline.ai_model_id else None

            enriched.append({
                "pipeline_id": pipeline.id,
                "pipeline": normalized_pipeline,
                "source": self._normalize_source(source),
                "ai_model": self._normalize_ai_model(ai_model),
                "pipeline_configs": pipeline_configs_by_id.get(pipeline.id, {}),
            })

        enriched.sort(key=lambda item: item["pipeline_id"])
        return enriched
