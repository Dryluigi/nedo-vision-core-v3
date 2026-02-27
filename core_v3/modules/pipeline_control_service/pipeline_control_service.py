from ..pipeline_sync_notifier.pipeline_sync_notifier_interface import PipelineSyncNotifierInterface
from ..pipeline_executor.pipeline_executor_interface import PipelineExecutorInterface

class PipelineControlService(PipelineSyncNotifierInterface):
    def __init__(self, pipeline_executor: PipelineExecutorInterface):
        self._pipeline_executor = pipeline_executor

    def notify_pipeline_update(self, updated_pipeline_id: str):
        self._pipeline_executor.restart(updated_pipeline_id)

    def notify_new_pipeline(self, new_pipeline_id: str):
        self._pipeline_executor.start(new_pipeline_id)

    def notify_deleted_pipeline(self, deleted_pipeline_id: str):
        self._pipeline_executor.stop(deleted_pipeline_id)