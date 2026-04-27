from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import relationship
from ..database.DatabaseManager import Base

class WorkerSourcePipelineEntity(Base):
    __tablename__ = "worker_source_pipeline"
    __bind_key__ = "config"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    worker_source_id = Column(String, nullable=False)
    worker_id = Column(String, nullable=False)
    ai_model_id = Column(String, nullable=True)
    pipeline_status_code = Column(String, nullable=False)
    location_name = Column(String, nullable=True)
    last_preview_request_at = Column(DateTime, nullable=True)

    worker_source_pipeline_configs = relationship(
        "WorkerSourcePipelineConfigEntity",
        back_populates="pipeline",
        cascade="all, delete-orphan"
    )
