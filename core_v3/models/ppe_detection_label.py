import uuid
from sqlalchemy import Column, String, ForeignKey, DateTime, Float, Integer
from sqlalchemy.orm import relationship
from ..database.DatabaseManager import Base

class PPEDetectionLabelEntity(Base):
    __tablename__ = "ppe_detection_labels"
    __bind_key__ = "default"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    detection_id = Column(String, ForeignKey("ppe_detections.id"), nullable=False)
    code = Column(String, nullable=False)  # helmet, vest, etc.
    confidence_score = Column(Float, nullable=False)
    detection_count = Column(Integer, nullable=False, default=0)
    b_box_x1 = Column(Float, nullable=False)
    b_box_y1 = Column(Float, nullable=False)
    b_box_x2 = Column(Float, nullable=False)
    b_box_y2 = Column(Float, nullable=False)

    detection = relationship("PPEDetectionEntity", back_populates="ppe_labels")
