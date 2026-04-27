import uuid
import json
from sqlalchemy import Column, String, DateTime
from ..database.DatabaseManager import Base

class AIModelEntity(Base):
    __tablename__ = "ai_model"
    __bind_key__ = "default"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    file = Column(String, nullable=False)
    type = Column(String, nullable=False)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    download_status = Column(String, nullable=True, default="completed")  # pending, downloading, completed, failed
    last_download_attempt = Column(DateTime, nullable=True)
    download_error = Column(String, nullable=True)
    classes = Column(String, nullable=True) 
    ppe_class_groups = Column(String, nullable=True) 
    main_class = Column(String, nullable=True)

    def __repr__(self):
        return (
            f"<AIModelEntity(id={self.id}, name={self.name}, type={self.type}, "
            f"file={self.file}, version={self.version})>"
        )

    def __str__(self):
        return (
            f"AIModelEntity(id={self.id}, name={self.name}, type={self.type}, "
            f"file={self.file}, version={self.version}, status={self.download_status})"
        )

    def is_ready_for_use(self) -> bool:
        """Check if the model is ready for use (downloaded and available)."""
        return self.download_status == "completed"

    def is_downloading(self) -> bool:
        """Check if the model is currently being downloaded."""
        return self.download_status in ["pending", "downloading"]

    def has_download_failed(self) -> bool:
        """Check if the model download has failed."""
        return self.download_status == "failed"

    def set_classes(self, classes_list):
        self.classes = json.dumps(classes_list)

    def get_classes(self):
        return json.loads(self.classes) if self.classes else []

    def set_ppe_class_groups(self, groups_list):
        self.ppe_class_groups = json.dumps(groups_list)

    def get_ppe_class_groups(self):
        return json.loads(self.ppe_class_groups) if self.ppe_class_groups else []

    def set_main_class(self, main_class):
        self.main_class = main_class

    def get_main_class(self):
        return self.main_class or None