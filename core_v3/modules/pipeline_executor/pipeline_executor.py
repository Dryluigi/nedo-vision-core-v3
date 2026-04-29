import queue
import threading
import re
from .pipeline_executor_interface import PipelineExecutorInterface
from ..pipeline_sync_notifier.pipeline_sync_notifier_interface import PipelineSyncNotifierInterface
from ..pipeline_sync_service.pipeline_sync_service import PipelineSyncService
from ..source_sync_notifier.source_sync_notifier_interface import SourceSyncNotifierInterface
from ..deepstream_pipeline.file_deepstream_pipeline import FileDeepstreamPipeline
from ..deepstream_pipeline.live_deepstream_pipeline import LiveRtspDeepstreamPipeline
from ..deepstream_pipeline.constant import PIPELINE_STATUS_RUNNING, PIPELINE_STATUS_STARTING, PIPELINE_STATUS_STOPPED, PIPELINE_STATUS_STOPPING
from ..triton_model_manager.triton_model_manager import TritonModelManager
from ..triton_model_converter.rfdetr_artifact_layout import get_rfdetr_infer_config_path
from ..deepstream_pipeline.person_attribute_aggregator import PersonAttributeAggregator
from ..deepstream_pipeline.capture_decision_engine import CaptureDecisionEngine
from ..capture_processing_service.capture_processing_service import CaptureProcessingService
from ..drawing.FrameDrawer import FrameDrawer
from ...repositories.WorkerSourcePipelineRepository import WorkerSourcePipelineRepository
from ...repositories.WorkerSourceRepository import WorkerSourceRepository
from ...utils.RTMPUrl import RTMPUrl


PIPELINE_STATUS_RESTARTING = "restarting"


class PipelineExecutor(PipelineExecutorInterface, PipelineSyncNotifierInterface, SourceSyncNotifierInterface):
    DEFAULT_OUTPUT_WIDTH = 1280
    DEFAULT_OUTPUT_HEIGHT = 720
    DEFAULT_OUTPUT_FPS = 30

    def __init__(
            self,
            deepstream_pipelines_update_queue: queue.Queue,
            pipeline_sync_service: PipelineSyncService,
            pipeline_repository: WorkerSourcePipelineRepository,
            source_repository: WorkerSourceRepository,
            triton_model_manager: TritonModelManager,
            capture_processing_service: CaptureProcessingService,
        ):
        self._pipelines = {}
        self._deepstream_pipelines_update_queue = deepstream_pipelines_update_queue
        self._pipeline_sync_service = pipeline_sync_service
        self._pipeline_repository = pipeline_repository
        self._source_repository = source_repository
        self._triton_model_manager = triton_model_manager
        self._capture_processing_service = capture_processing_service
        self._update_status_listener_thread = None

        self._start_status_update_listener()

    def start(self, pipeline_id: str):
        print(f"Starting pipeline {pipeline_id}")
        # Obtain pipeline and source data from db
        pipeline = self._pipeline_repository.get_pipeline_by_id(pipeline_id)
        if not pipeline:
            return

        source = self._source_repository.get_worker_source(pipeline.worker_source_id)
        if not source:
            print(f"Cannot start pipeline {pipeline_id}: source not found")
            return
        if (source.status_code or "").strip().lower() == "stopped":
            print(f"Skipping start for pipeline {pipeline_id}: source is stopped")
            return

        output_width, output_height = self._parse_resolution(
            source.resolution
        )
        output_fps = self._parse_frame_rate(source.frame_rate)

        # Create deepstream pipeline
        frame_drawer = FrameDrawer()
        location_name = pipeline.location_name or source.name or "LOCATION"
        frame_drawer.location_name = location_name
        frame_drawer.update_config(
            icons={
                "helmet": "assets/icons/helmet-green.png",
                "no_helmet": "assets/icons/helmet-red.png",
                "vest": "assets/icons/vest-green.png",
                "no_vest": "assets/icons/vest-red.png",
            },
            violation_labels=["no_helmet", "no_vest"],
            compliance_labels=["helmet", "vest"],
        )
        class_id_to_label = {
            0: "background",
            1: "helmet",
            2: "no_helmet",
            3: "no_vest",
            4: "person",
            5: "vest",
        }
        capture_decision_engine = CaptureDecisionEngine(
            capture_threshold=5,
            track_timeout_seconds=5
        )
        person_attribute_aggregator = PersonAttributeAggregator(
            person_class_id=4,
            attribute_class_ids=[1, 2, 3, 5],
            coverage_threshold=0.3,
            person_conf_threshold=0.5,
            attribute_conf_threshold=0.5,
        )
        infer_config_path = get_rfdetr_infer_config_path(pipeline.ai_model_id)

        if source.type_code == "live":
            pipeline = LiveRtspDeepstreamPipeline(
                pipeline.id,
                f"{pipeline.name}",
                self._deepstream_pipelines_update_queue,
                source.url,
                infer_config_path,
                self._triton_model_manager,
                pipeline.ai_model_id,
                pipeline.worker_id,
                source.id,
                location_name,
                capture_decision_engine,
                person_attribute_aggregator,
                frame_drawer,
                self._capture_processing_service,
                class_id_to_label,
                RTMPUrl.get_publish_url(f"pipeline-{pipeline.id}"),
                output_width=output_width,
                output_height=output_height,
                target_fps=output_fps,
            )

            self._pipelines[pipeline_id] = pipeline
        elif source.type_code == "file":
            file_url = source.file_path
            if not file_url.startswith("file://"):
                file_url = "file://" + file_url

            pipeline = FileDeepstreamPipeline(
                pipeline.id,
                f"{pipeline.name}",
                self._deepstream_pipelines_update_queue,
                file_url,
                infer_config_path,
                RTMPUrl.get_publish_url(f"pipeline-{pipeline.id}"),
                self._triton_model_manager,
                pipeline.ai_model_id,
                pipeline.worker_id,
                source.id,
                location_name,
                capture_decision_engine,
                person_attribute_aggregator,
                frame_drawer,
                self._capture_processing_service,
                class_id_to_label,
                output_width=output_width,
                output_height=output_height,
                target_fps=output_fps,
            )

            self._pipelines[pipeline_id] = pipeline

        # Start the pipeline
        self._pipelines[pipeline_id].play()

    def stop(self, pipeline_id: str):
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            return
        pipeline.stop()
        del self._pipelines[pipeline_id]

    def restart(self, pipeline_id: str):
        if pipeline_id in self._pipelines:
            self.stop(pipeline_id)
        self.start(pipeline_id)

    def notify_pipeline_update(self, updated_pipeline_id: str):
        pipeline = self._pipeline_repository.get_pipeline_by_id(updated_pipeline_id)
        if not pipeline:
            return

        snapshot = None
        change = None
        if self._pipeline_sync_service is not None:
            snapshot = self._pipeline_sync_service.get_pipeline_snapshot(updated_pipeline_id)
            change = self._pipeline_sync_service.get_last_change(updated_pipeline_id)

        source_stopped = bool(change and change.get("source_stopped"))

        # If pipeline never been started, start directly.
        if pipeline.id not in self._pipelines:
            if source_stopped:
                self._pipeline_sync_service.update_pipeline_status(updated_pipeline_id, PIPELINE_STATUS_STOPPED)
                return
            if pipeline.pipeline_status_code in [PIPELINE_STATUS_STARTING, PIPELINE_STATUS_RUNNING]:
                self.start(updated_pipeline_id)

            return

        pipeline_metadata = self._pipelines[pipeline.id].get_metadata()
        active_pipeline_status = pipeline_metadata["pipeline_status"]

        if source_stopped:
            print(f"Stopping pipeline {pipeline.id}: source is stopped")
            self.stop(pipeline.id)
            self._pipeline_sync_service.update_pipeline_status(updated_pipeline_id, PIPELINE_STATUS_STOPPED)
            return

        if active_pipeline_status == PIPELINE_STATUS_RESTARTING:
            print(f"Ignoring update for pipeline {pipeline.id}: pipeline is restarting")
            return

        if change and change.get("requires_restart") and pipeline.pipeline_status_code in [PIPELINE_STATUS_STARTING, PIPELINE_STATUS_RUNNING]:
            print(f"Restarting pipeline {pipeline.id} because {change.get('reasons', [])}")
            self.restart(updated_pipeline_id)
            return

        # If pipeline already running here and metadata is changed, then restart it.
        if (
            pipeline_metadata["pipeline_id"] != pipeline.id
            or pipeline_metadata["pipeline_name"] != pipeline.name
            or pipeline_metadata.get("location_name") != pipeline.location_name
        ):
            print("Ada perubahan metadata coy")
            if active_pipeline_status in [PIPELINE_STATUS_STARTING, PIPELINE_STATUS_RUNNING]:
                print("Restarting")
                self.stop(pipeline.id)
                self.start(updated_pipeline_id)
                return

        # If only status is changed then it's mean that worker service is doing action.
        if active_pipeline_status != pipeline.pipeline_status_code:
            if pipeline.pipeline_status_code in [PIPELINE_STATUS_STOPPING, PIPELINE_STATUS_STOPPED]:
                if active_pipeline_status in [PIPELINE_STATUS_STOPPING, PIPELINE_STATUS_STOPPED]:
                    self._pipeline_sync_service.update_pipeline_status(updated_pipeline_id, active_pipeline_status)
                    return
                print("Stopping")
                self.stop(pipeline.id)
            
            if pipeline.pipeline_status_code in [PIPELINE_STATUS_STARTING, PIPELINE_STATUS_RUNNING]:
                if active_pipeline_status in [PIPELINE_STATUS_STARTING, PIPELINE_STATUS_RUNNING]:
                    self._pipeline_sync_service.update_pipeline_status(updated_pipeline_id, active_pipeline_status)
                    return
                self.start(pipeline.id)

            return

    def notify_new_pipeline(self, new_pipeline_id: str):
        pipeline = self._pipeline_repository.get_pipeline_by_id(new_pipeline_id)
        if not pipeline:
            return

        snapshot = None
        if self._pipeline_sync_service is not None:
            snapshot = self._pipeline_sync_service.get_pipeline_snapshot(new_pipeline_id)
        source_status = (((snapshot or {}).get("source") or {}).get("status_code") or "").strip().lower()

        if source_status == "stopped":
            self._pipeline_sync_service.update_pipeline_status(new_pipeline_id, PIPELINE_STATUS_STOPPED)
            return

        # It means run action being triggered.
        if pipeline.pipeline_status_code in [PIPELINE_STATUS_STARTING, PIPELINE_STATUS_RUNNING]:
            self.start(new_pipeline_id)
        else:
            self._pipeline_sync_service.update_pipeline_status(new_pipeline_id, PIPELINE_STATUS_STOPPED)

    def notify_deleted_pipeline(self, deleted_pipeline_id: str):
        self.stop(deleted_pipeline_id)

    def notify_source_status_update(self, source_id: str, before_status: str, after_status: str):
        pass

    def notify_source_url_update(self, source_id: str, new_url: str):
        pass
    
    def notify_source_deleted(self, source_id: str):
        pass

    def _start_status_update_listener(self):
        self._update_status_listener_thread = threading.Thread(target=self._listen_pipeline_status_update_queue)
        self._update_status_listener_thread.start()

    @classmethod
    def _parse_resolution(cls, resolution: str | None) -> tuple[int, int]:
        if not resolution:
            return cls.DEFAULT_OUTPUT_WIDTH, cls.DEFAULT_OUTPUT_HEIGHT

        cleaned = resolution.strip().lower()
        match = re.search(r"(\d+)\s*[x:*,/ -]\s*(\d+)", cleaned)
        if not match:
            return cls.DEFAULT_OUTPUT_WIDTH, cls.DEFAULT_OUTPUT_HEIGHT

        width = int(match.group(1))
        height = int(match.group(2))
        if width <= 0 or height <= 0:
            return cls.DEFAULT_OUTPUT_WIDTH, cls.DEFAULT_OUTPUT_HEIGHT

        return width, height

    @classmethod
    def _parse_frame_rate(cls, frame_rate: float | None) -> int:
        if frame_rate is None:
            return cls.DEFAULT_OUTPUT_FPS

        try:
            parsed = int(round(float(frame_rate)))
        except (TypeError, ValueError):
            return cls.DEFAULT_OUTPUT_FPS

        if parsed <= 0:
            return cls.DEFAULT_OUTPUT_FPS

        return parsed

    def  _listen_pipeline_status_update_queue(self):
        while True:
            data = self._deepstream_pipelines_update_queue.get()

            print(f"Obtain status data from pipeline {data}")
            if data["status"] == PIPELINE_STATUS_RUNNING or data["status"] == PIPELINE_STATUS_STOPPED:
                self._pipeline_sync_service.update_pipeline_status(data["pipeline_id"], data["status"])
