from sqlalchemy import Column, String, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from ..database.DatabaseManager import Base


class WorkerSourcePipelineConfigEntity(Base):
    __tablename__ = "worker_source_pipeline_config"
    __bind_key__ = "config"

    id = Column(String, primary_key=True)
    worker_source_pipeline_id = Column(
        String, ForeignKey("worker_source_pipeline.id", ondelete="CASCADE"), nullable=False
    )
    pipeline_config_id = Column(String, nullable=False)
    is_enabled = Column(Boolean, nullable=False)
    value = Column(String, nullable=True)
    pipeline_config_name = Column(String, nullable=False)
    pipeline_config_code = Column(String, nullable=False)

    pipeline = relationship(
        "WorkerSourcePipelineEntity",
        back_populates="worker_source_pipeline_configs",
        passive_deletes=True 
    )
