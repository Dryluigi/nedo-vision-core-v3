from sqlalchemy import Column, String, Float
from ..database.DatabaseManager import Base

class WorkerSourceEntity(Base):
    __tablename__ = "worker_source"
    __bind_key__ = "config"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    worker_id = Column(String, nullable=False)
    type_code = Column(String, nullable=False)
    file_path = Column(String, nullable=True)
    url = Column(String, nullable=False)
    resolution = Column(String, nullable=True)
    status_code = Column(String, nullable=True)
    frame_rate = Column(Float, nullable=True)
    source_location_code = Column(String, nullable=True)  # Optional field
    latitude = Column(Float, nullable=True)  # Optional field
    longitude = Column(Float, nullable=True)  # Optional field
