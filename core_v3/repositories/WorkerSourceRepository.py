from .BaseRepository import BaseRepository
from ..models.worker_source import WorkerSourceEntity
from ..modules.deepstream_pipeline.constant import SOURCE_STATUS_CONNECTED

class WorkerSourceRepository(BaseRepository):
    def __init__(self):
        super().__init__(db_name="config")

    def get_worker_sources(self):
        """
        Fetch all worker sources from the local database in a single query.

        Returns:
            list: A list of WorkerSourceEntity records.
        """
        with self._get_session() as session:
            session.expire_all()
            sources = session.query(WorkerSourceEntity).all()
            for source in sources:
                session.expunge(source)
            return sources
    
    def get_worker_source(self, source_id: str):
        """
        Fetch a single worker source by ID.
        
        Args:
            source_id (str): The worker source ID
            
        Returns:
            WorkerSourceEntity: The worker source entity or None if not found
        """
        with self._get_session() as session:
            session.expire_all()
            source = session.query(WorkerSourceEntity).filter(
                WorkerSourceEntity.id == source_id
            ).first()
            if source:
                session.expunge(source)
            return source
    
    def is_source_connected(self, source_id: str) -> bool:
        """
        Check if a worker source is connected.
        
        Args:
            source_id (str): The worker source ID
            
        Returns:
            bool: True if source is connected, False otherwise
        """
        source = self.get_worker_source(source_id)
        if not source:
            return False
        
        return source.status_code == SOURCE_STATUS_CONNECTED if source.status_code else False
