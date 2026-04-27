from sqlalchemy import Column, String, Integer
from ..database.DatabaseManager import Base

class LogEntity(Base):
    __tablename__ = "logs"
    __bind_key__ = "logging"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message = Column(String, nullable=False)