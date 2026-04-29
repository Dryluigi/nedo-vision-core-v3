from .BaseRepository import BaseRepository
from ..models.ai_model import AIModelEntity


class AIModelRepository(BaseRepository):
    def __init__(self):
        super().__init__(db_name="default")

    def get_ai_models(self) -> list[AIModelEntity]:
        with self._get_session() as session:
            session.expire_all()
            models = session.query(AIModelEntity).all()
            for model in models:
                session.expunge(model)
            return models

    def get_ai_model_by_id(self, ai_model_id: str) -> AIModelEntity | None:
        with self._get_session() as session:
            session.expire_all()
            model = session.query(AIModelEntity).filter(AIModelEntity.id == ai_model_id).first()
            if model:
                session.expunge(model)
            return model
