from abc import ABC, abstractmethod
from ...models.worker_source_pipeline import WorkerSourcePipelineEntity

class PipelineSyncNotifierInterface(ABC):
    @abstractmethod
    def notify_pipeline_update(self, updated_pipeline_id: str):
        pass

    @abstractmethod
    def notify_new_pipeline(self, new_pipeline_id: str):
        pass

    @abstractmethod
    def notify_deleted_pipeline(self, deleted_pipeline_id: str):
        pass