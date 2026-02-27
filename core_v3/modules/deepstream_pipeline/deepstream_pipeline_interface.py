from abc import ABC, abstractmethod
from typing import Dict

class DeepstreamPipelineInterface(ABC):
    @abstractmethod
    def play(self):
        pass

    @abstractmethod
    def stop(self):
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict:
        pass