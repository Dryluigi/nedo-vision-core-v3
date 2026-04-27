from sqlalchemy import Column, String
from ..database.DatabaseManager import Base

class UserEntity(Base):
    __tablename__ = "user"
    __bind_key__ = "auth"

    id = Column(String, primary_key=True)
    username = Column(String, nullable=False)
    password = Column(String, nullable=False)