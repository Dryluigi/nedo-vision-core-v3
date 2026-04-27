from sqlalchemy import Column, String, Integer
from ..database.DatabaseManager import Base

class DatasetSourceEntity(Base):
    __tablename__ = "dataset_sources"
    __bind_key__ = "default"

    id = Column(String, primary_key=True)
    dataset_id = Column(String, nullable=False)
    worker_source_id = Column(String, nullable=False)
    sampling_interval = Column(Integer, nullable=False)
    dataset_name = Column(String, nullable=False)
    worker_source_name = Column(String, nullable=False)
    worker_source_url = Column(String, nullable=False)

    def __repr__(self):
        return (
            f"<DatasetSourceEntity(id={self.id}, dataset_id={self.dataset_id}, "
            f"worker_source_id={self.worker_source_id}, sampling_interval={self.sampling_interval}, "
            f"dataset_name={self.dataset_name}, worker_source_name={self.worker_source_name}, "
            f"worker_source_url={self.worker_source_url})>"
        )

    def __str__(self):
        return (
            f"DatasetSourceEntity(id={self.id}, dataset_id={self.dataset_id}, "
            f"worker_source_id={self.worker_source_id}, sampling_interval={self.sampling_interval}, "
            f"dataset_name={self.dataset_name}, worker_source_name={self.worker_source_name}, "
            f"worker_source_url={self.worker_source_url})"
        ) 