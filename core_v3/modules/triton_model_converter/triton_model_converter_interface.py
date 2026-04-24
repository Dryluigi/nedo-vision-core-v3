from abc import ABC, abstractmethod
from ...models.ai_model import AIModelEntity


class TritonModelConverterInterface(ABC):
    @abstractmethod
    def is_ready(self, ai_model: AIModelEntity) -> bool:
        pass

    @abstractmethod
    def prepare(self, ai_model: AIModelEntity):
        pass
