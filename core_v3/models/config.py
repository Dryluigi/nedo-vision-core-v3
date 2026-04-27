from sqlalchemy import Column, String
from ..database.DatabaseManager import Base

class ConfigEntity(Base):
    __tablename__ = "server_config"
    __bind_key__ = "config"

    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)
