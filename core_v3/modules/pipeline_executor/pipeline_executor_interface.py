from abc import ABC, abstractmethod

class PipelineExecutorInterface(ABC):
    @abstractmethod
    def start(self, pipeline_id: str):
        pass

    @abstractmethod
    def stop(self, pipeline_id: str):
        pass

    @abstractmethod
    def restart(self, pipeline_id: str):
        pass