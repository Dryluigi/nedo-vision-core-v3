from abc import ABC, abstractmethod

class SourceSyncNotifierInterface(ABC):
    @abstractmethod
    def notify_source_status_update(self, source_id: str, before_status: str, after_status: str):
        pass

    @abstractmethod
    def notify_source_url_update(self, source_id: str, new_url: str):
        pass
    
    @abstractmethod
    def notify_source_deleted(self, source_id: str):
        pass