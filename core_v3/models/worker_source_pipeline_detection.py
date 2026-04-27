from datetime import datetime
import uuid
from sqlalchemy import Column, DateTime, String
from ..database.DatabaseManager import Base

class WorkerSourcePipelineDetectionEntity(Base):
    __tablename__ = "worker_source_pipeline_detection"
    __bind_key__ = "default"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    worker_source_pipeline_id = Column(String, nullable=False)
    image_path = Column(String, nullable=True)
    data = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
