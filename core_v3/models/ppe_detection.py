import uuid
import datetime
from sqlalchemy import Column, String, ForeignKey, DateTime, Float, Integer
from sqlalchemy.orm import relationship
from ..database.DatabaseManager import Base

class PPEDetectionEntity(Base):
    __tablename__ = "ppe_detections"
    __bind_key__ = "default"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    worker_id = Column(String, nullable=False)
    worker_source_id = Column(String, nullable=False)
    person_id = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    image_tile_path = Column(String, nullable=False)
    detection_count = Column(Integer, nullable=False, default=0)  # Tracks total detections before saving
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    b_box_x1 = Column(Float, nullable=False)
    b_box_y1 = Column(Float, nullable=False)
    b_box_x2 = Column(Float, nullable=False)
    b_box_y2 = Column(Float, nullable=False)

    ppe_labels = relationship("PPEDetectionLabelEntity", back_populates="detection")

    def __repr__(self):
        return (
            f"<PPEDetectionEntity(id={self.id}, worker_id={self.worker_id}, "
            f"worker_source_id={self.worker_source_id}, person_id={self.person_id}, "
            f"image_path={self.image_path}, detection_count={self.detection_count}, "
            f"created_at={self.created_at})>"
        )

    def __str__(self):
        return (
            f"PPEDetectionEntity(id={self.id}, worker_id={self.worker_id}, "
            f"worker_source_id={self.worker_source_id}, person_id={self.person_id}, "
            f"detection_count={self.detection_count}, created_at={self.created_at})"
        )