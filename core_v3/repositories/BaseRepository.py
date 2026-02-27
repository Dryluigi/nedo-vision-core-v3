from contextlib import contextmanager
from ..database.DatabaseManager import DatabaseManager


class BaseRepository:
    """
    Base repository class that provides thread-safe database session management.
    
    All repositories should inherit from this class to ensure proper connection pooling
    and to prevent connection leaks in multi-threaded environments.
    """
    
    def __init__(self, db_name: str = "default"):
        """
        Initialize the base repository.
        
        Args:
            db_name: Name of the database to connect to ('default', 'config', or 'logging')
        """
        self.db_manager = DatabaseManager()
        self.db_name = db_name
    
    @contextmanager
    def _get_session(self):
        """
        Context manager for database sessions.
        
        Ensures sessions are properly opened and closed, preventing connection leaks.
        Each operation gets a fresh session that is automatically closed when done.
        
        Usage:
            with self._get_session() as session:
                results = session.query(Model).all()
                return results
        """
        session = self.db_manager.get_session(self.db_name)
        try:
            yield session
            session.commit()  # Commit any pending changes
        except Exception:
            session.rollback()  # Rollback on error
            raise
        finally:
            session.close()  # Always close the session
