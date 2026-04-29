import json
from typing import List
from sqlalchemy.exc import SQLAlchemyError
from .BaseRepository import BaseRepository
from ..models.worker_source_pipeline import WorkerSourcePipelineEntity
from ..models.worker_source_pipeline_config import WorkerSourcePipelineConfigEntity


class WorkerSourcePipelineRepository(BaseRepository):
    def __init__(self):
        super().__init__(db_name="config")

    def get_all_pipelines(self) -> List[WorkerSourcePipelineEntity]:
        """
        Fetch all worker source pipelines from the local database in a single query.

        Returns:
            list: A list of WorkerSourcePipelineEntity records.
        """
        with self._get_session() as session:
            session.expire_all()
            # Query and detach from session before returning
            pipelines = session.query(WorkerSourcePipelineEntity).all()
            # Expunge objects so they can be used outside session
            for pipeline in pipelines:
                session.expunge(pipeline)
            return pipelines
        
    def get_pipeline_by_id(self, pipeline_id: str) -> WorkerSourcePipelineEntity | None:
        """
        Retrieve a single worker source pipeline by its ID.

        Args:
            pipeline_id (str): Pipeline ID

        Returns:
            WorkerSourcePipelineEntity | None
        """
        try:
            with self._get_session() as session:
                session.expire_all()

                pipeline = (
                    session
                    .query(WorkerSourcePipelineEntity)
                    .filter(WorkerSourcePipelineEntity.id == pipeline_id)
                    .first()
                )

                if pipeline:
                    session.expunge(pipeline)

                return pipeline

        except SQLAlchemyError as e:
            print(f"Database error while retrieving pipeline by id: {e}")
            return None

    def get_pipeline_configs_by_pipeline_id(self, pipeline_id):
        """
        Retrieves all pipeline configurations for a given pipeline ID and returns them as a dictionary.

        The dictionary format:
        {
            "config_code_1": { "id": "xxx", "is_enabled": true, "value": "some_value", "name": "Config Name" },
            "config_code_2": { "id": "yyy", "is_enabled": false, "value": "another_value", "name": "Another Config Name" }
        }

        Args:
            pipeline_id (str): The unique identifier of the pipeline.

        Returns:
            dict: A dictionary mapping pipeline_config_code to its configuration details.
        """
        try:
            with self._get_session() as session:
                pipeline_configs = (
                    session.query(WorkerSourcePipelineConfigEntity)
                    .filter(WorkerSourcePipelineConfigEntity.worker_source_pipeline_id == pipeline_id)
                    .all()
                )

                def parse_value(value):
                    """Attempts to parse the value as JSON if applicable."""
                    if not value:
                        return value  # Keep None or empty string as is
                    
                    value = value.strip()  # Remove leading/trailing spaces
                    if (value.startswith("{") and value.endswith("}")) or (value.startswith("[") and value.endswith("]")):
                        try:
                            return json.loads(value)  # Parse JSON object or list
                        except json.JSONDecodeError:
                            pass  # Keep as string if parsing fails
                    return value  # Return original value if not JSON

                # Convert result into a dictionary with pipeline_config_code as key
                config_dict = {
                    config.pipeline_config_code: {
                        "id": config.id,
                        "is_enabled": config.is_enabled,  # Keep original boolean value
                        "value": parse_value(config.value),  # Parse JSON if applicable
                        "name": config.pipeline_config_name
                    }
                    for config in pipeline_configs
                }

                return config_dict

        except SQLAlchemyError as e:
            print(f"Database error while retrieving pipeline configs: {e}")
            return {}

    def get_all_pipeline_configs_grouped(self):
        try:
            with self._get_session() as session:
                pipeline_configs = session.query(WorkerSourcePipelineConfigEntity).all()

                def parse_value(value):
                    if not value:
                        return value

                    value = value.strip()
                    if (value.startswith("{") and value.endswith("}")) or (value.startswith("[") and value.endswith("]")):
                        try:
                            return json.loads(value)
                        except json.JSONDecodeError:
                            pass
                    return value

                grouped_configs = {}
                for config in pipeline_configs:
                    grouped_configs.setdefault(config.worker_source_pipeline_id, {})[config.pipeline_config_code] = {
                        "id": config.id,
                        "is_enabled": config.is_enabled,
                        "value": parse_value(config.value),
                        "name": config.pipeline_config_name,
                    }

                return grouped_configs

        except SQLAlchemyError as e:
            print(f"Database error while retrieving all pipeline configs: {e}")
            return {}
        
    def get_worker_source_pipeline(self, pipeline_id):
        with self._get_session() as session:
            session.expire_all()
            pipeline = session.query(WorkerSourcePipelineEntity).filter_by(id=pipeline_id).first()
            if pipeline:
                session.expunge(pipeline)  # Detach from session
            return pipeline
    
    def update_pipeline_status(self, pipeline_id: str, status_code: str) -> bool:
        """
        Update the status of a pipeline.
        
        Args:
            pipeline_id: The ID of the pipeline to update
            status_code: The new status code ('run', 'stop', 'restart')
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            with self._get_session() as session:
                pipeline = session.query(WorkerSourcePipelineEntity).filter_by(id=pipeline_id).first()
                if pipeline:
                    pipeline.pipeline_status_code = status_code
                    # Commit happens automatically via context manager
                    return True
                return False
        except SQLAlchemyError as e:
            print(f"Database error while updating pipeline status: {e}")
            return False
