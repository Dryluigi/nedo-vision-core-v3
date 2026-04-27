from sqlalchemy import Column, String, Integer
from ..database.DatabaseManager import Base

class AuthEntity(Base):
    __tablename__ = "auth"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    password = Column(String, nullable=False)

    def __repr__(self):
        return f"<AuthEntity(id={self.id}, username={self.username})>"

    def to_dict(self):
        return {"id": self.id, "username": self.username}
