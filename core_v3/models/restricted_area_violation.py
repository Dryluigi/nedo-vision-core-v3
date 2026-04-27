import uuid
import datetime
from sqlalchemy import Column, Float, String, DateTime
from ..database.DatabaseManager import Base

class RestrictedAreaViolationEntity(Base):
    __tablename__ = "restricted_area_violation"
    __bind_key__ = "default"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    worker_source_id = Column(String, nullable=False)
    person_id = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    image_tile_path = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    b_box_x1 = Column(Float, nullable=False)
    b_box_y1 = Column(Float, nullable=False)
    b_box_x2 = Column(Float, nullable=False)
    b_box_y2 = Column(Float, nullable=False)
