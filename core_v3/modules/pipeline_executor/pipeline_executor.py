import queue
import threading
from .pipeline_executor_interface import PipelineExecutorInterface
from ..pipeline_sync_notifier.pipeline_sync_notifier_interface import PipelineSyncNotifierInterface
from ..pipeline_sync_service.pipeline_sync_service import PipelineSyncService
from ..source_sync_notifier.source_sync_notifier_interface import SourceSyncNotifierInterface
from ..deepstream_pipeline.file_deepstream_pipeline import FileDeepstreamPipeline
from ..deepstream_pipeline.live_deepstream_pipeline import LiveRtspDeepstreamPipeline
from ..deepstream_pipeline.constant import PIPELINE_STATUS_RUNNING, PIPELINE_STATUS_STARTING, PIPELINE_STATUS_STOPPED, PIPELINE_STATUS_STOPPING
from ..triton_model_manager.triton_model_manager import TritonModelManager
from ...repositories.WorkerSourcePipelineRepository import WorkerSourcePipelineRepository
from ...repositories.WorkerSourceRepository import WorkerSourceRepository
from ...utils.RTMPUrl import RTMPUrl

class PipelineExecutor(PipelineExecutorInterface, PipelineSyncNotifierInterface, SourceSyncNotifierInterface):
    def __init__(
            self,
            deepstream_pipelines_update_queue: queue.Queue,
            pipeline_sync_service: PipelineSyncService,
            pipeline_repository: WorkerSourcePipelineRepository,
            source_repository: WorkerSourceRepository,
            triton_model_manager: TritonModelManager,
        ):
        self._pipelines = {}
        self._deepstream_pipelines_update_queue = deepstream_pipelines_update_queue
        self._pipeline_sync_service = pipeline_sync_service
        self._pipeline_repository = pipeline_repository
        self._source_repository = source_repository
        self._triton_model_manager = triton_model_manager
        self._update_status_listener_thread = None

        self._start_status_update_listener()

    def start(self, pipeline_id: str):
        print(f"Starting pipeline {pipeline_id}")
        # Obtain pipeline and source data from db
        pipeline = self._pipeline_repository.get_pipeline_by_id(pipeline_id)
        if not pipeline:
            return
        
        source = self._source_repository.get_worker_source(pipeline.worker_source_id)
        if not pipeline:
            return

        # Create deepstream pipeline
        if source.type_code == "live":
            pipeline = LiveRtspDeepstreamPipeline(
                pipeline.id,
                f"{pipeline.name}",
                self._deepstream_pipelines_update_queue,
                source.url,
                self._triton_model_manager,
                RTMPUrl.get_publish_url(f"pipeline-{pipeline.id}")
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
                "/app/config/deepstream-inferserver-yolo.txt",
                RTMPUrl.get_publish_url(f"pipeline-{pipeline.id}"),
                self._triton_model_manager
            )

            self._pipelines[pipeline_id] = pipeline

        # Start the pipeline
        self._pipelines[pipeline_id].play()

    def stop(self, pipeline_id: str):
        self._pipelines[pipeline_id].stop()
        del self._pipelines[pipeline_id]

    def restart(self, pipeline_id: str):
        pass

    def notify_pipeline_update(self, updated_pipeline_id: str):
        pipeline = self._pipeline_repository.get_pipeline_by_id(updated_pipeline_id)
        if not pipeline:
            return
    
        # If pipeline never been started, start directly.
        if pipeline.id not in self._pipelines:
            if pipeline.pipeline_status_code in [PIPELINE_STATUS_STARTING, PIPELINE_STATUS_RUNNING]:
                self.start(updated_pipeline_id)

            return

        pipeline_metadata = self._pipelines[pipeline.id].get_metadata()
        active_pipeline_status = pipeline_metadata["pipeline_status"]

        # If pipeline already running here and metadata is changed, then restart it.
        if pipeline_metadata["pipeline_id"] != pipeline.id or pipeline_metadata["pipeline_name"] != pipeline.name:
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
        
        # It means run action being triggered.
        if pipeline.pipeline_status_code == PIPELINE_STATUS_STARTING:
            self.start(new_pipeline_id)
        else:
            self._pipeline_sync_service.update_pipeline_status(new_pipeline_id, "stop")

    def notify_deleted_pipeline(self, deleted_pipeline_id: str):
        pass

    def notify_source_status_update(self, source_id: str, before_status: str, after_status: str):
        pass

    def notify_source_url_update(self, source_id: str, new_url: str):
        pass
    
    def notify_source_deleted(self, source_id: str):
        pass

    def _start_status_update_listener(self):
        self._update_status_listener_thread = threading.Thread(target=self._listen_pipeline_status_update_queue)
        self._update_status_listener_thread.start()

    def  _listen_pipeline_status_update_queue(self):
        while True:
            data = self._deepstream_pipelines_update_queue.get()
            
            print(f"Obtain status data from pipeline {data}")
            if data["status"] == PIPELINE_STATUS_RUNNING or data["status"] == PIPELINE_STATUS_STOPPED:
                self._pipeline_sync_service.update_pipeline_status(data["pipeline_id"], data["status"])
