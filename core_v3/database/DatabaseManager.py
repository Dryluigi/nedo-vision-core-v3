import os
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.operations.ops import AlterColumnOp
from alembic.autogenerate import produce_migrations
from sqlalchemy import Column, Connection, Table, create_engine, inspect, MetaData, select, text
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from pathlib import Path
import logging

# Add dotenv for environment loading
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Base class for ORM models
Base = declarative_base()

# Dictionary to hold engines and session factories
ENGINES = {}
SESSION_FACTORIES = {}
_initialized = False  # To track whether databases have been initialized

def alter_drop_not_null_sqlite(conn: Connection, table_name: str, column_name: str):
    # Load existing metadata
    metadata = MetaData()
    metadata.reflect(conn)

    table = metadata.tables[table_name]

    # Create new table with updated schema (column without NOT NULL)
    new_columns = []
    for col in table.columns:
        new_col = Column(
            col.name,
            col.type,
            nullable=(True if col.name == column_name else col.nullable),
            primary_key=col.primary_key,
            default=col.default,
            server_default=col.server_default
        )
        new_columns.append(new_col)
    
    new_table = Table(f"{table_name}_new", MetaData(), *new_columns)
    new_table.create(conn)

    # Copy data
    data = conn.execute(select(table)).mappings().all()
    if data:
        conn.execute(new_table.insert(), data)

    # Drop old table, rename new table
    table.drop(conn)
    conn.execute(text(f"ALTER TABLE {table_name}_new RENAME TO {table_name}"))

class DatabaseManager:
    """
    Manages multiple database connections, engines, and session factories.
    """

    STORAGE_PATH = None
    STORAGE_PATHS = None

    @staticmethod
    def _load_env_file():
        """Load .env file from multiple possible locations (like worker_service.py)."""
        if load_dotenv is None:
            return
        from os.path import join, dirname
        import os
        possible_paths = [
            ".env",
            join(dirname(__file__), '..', '..', '.env'),
            join(dirname(__file__), '..', '.env'),
            join(os.path.expanduser("~"), '.nedo_vision_worker.env'),
        ]
        for env_path in possible_paths:
            if os.path.exists(env_path):
                try:
                    load_dotenv(env_path)
                    logging.info(f"✅ [DB] Loaded environment from: {env_path}")
                    break
                except Exception as e:
                    logging.warning(f"⚠️ [DB] Failed to load .env from {env_path}: {e}")

    @staticmethod
    def init_databases(storage_path: str = None):
        """
        Initialize databases and create tables for all models based on their `__bind_key__`.
        
        Args:
            storage_path: Storage path for databases and files. If None, will try to load from environment.
        """
        global _initialized
        if _initialized:
            logging.info("✅ Databases already initialized.")
            return

        # Set storage paths - prioritize parameter over environment variables
        if storage_path:
            DatabaseManager.STORAGE_PATH = Path(storage_path).resolve()
        else:
            # Fallback to environment variables for backward compatibility
            DatabaseManager._load_env_file()
            DatabaseManager.STORAGE_PATH = Path(os.environ.get("STORAGE_PATH", "data")).resolve()
            
        DatabaseManager.STORAGE_PATHS = {
            "db": DatabaseManager.STORAGE_PATH / "sqlite",
            "files": DatabaseManager.STORAGE_PATH / "files",
            "models": DatabaseManager.STORAGE_PATH / "model",
        }

        # Define database paths
        DB_PATHS = {
            "default": DatabaseManager.STORAGE_PATHS["db"] / "default.db",
            "config": DatabaseManager.STORAGE_PATHS["db"] / "config.db",
            "logging": DatabaseManager.STORAGE_PATHS["db"] / "logging.db",
        }

        # Initialize engines and session factories for each database
        for name, path in DB_PATHS.items():
            path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            
            # Configure connection pool for multi-threaded usage
            # pool_size: Max connections to keep open
            # max_overflow: Additional connections that can be created temporarily
            # pool_pre_ping: Test connections before using (prevents stale connections)
            # pool_recycle: Recycle connections after N seconds (prevents long-lived stale connections)
            engine = create_engine(
                f"sqlite:///{path.as_posix()}",
                pool_size=20,          # Base pool size for persistent connections
                max_overflow=30,       # Allow up to 30 additional temporary connections
                pool_pre_ping=True,    # Verify connection health before use
                pool_recycle=3600,     # Recycle connections after 1 hour
                connect_args={
                    "check_same_thread": False,  # Required for SQLite with multiple threads
                    "timeout": 30.0              # Connection timeout
                }
            )
            ENGINES[name] = engine
            SESSION_FACTORIES[name] = scoped_session(sessionmaker(bind=engine))  # Use scoped sessions
            DatabaseManager.synchronize(name)

        _initialized = True
        logging.info("✅ [APP] Databases initialized successfully.")

    @staticmethod
    def synchronize(db_name):
        engine = ENGINES[db_name]

        models = []
        for mapper_object in Base.registry.mappers:
            model_class = mapper_object.class_
            if hasattr(model_class, "__bind_key__") and model_class.__bind_key__ == db_name:
                models.append(model_class)

        inspector = inspect(engine)
        dialect = inspector.dialect.name
        existing_tables = inspector.get_table_names()

        if not existing_tables:
            if not models:
                return

            for model in models:
                model.metadata.create_all(bind=engine)

            logging.info(f"✅ [APP] Created initial tables for '{db_name}' database")
            return
        
        connection: Connection = engine.connect()
        try:
           with connection.begin():
                context = MigrationContext.configure(connection)

                target_metadata = MetaData()
                for model in models:
                    for table in model.metadata.tables.values():
                        if table.name not in target_metadata.tables:
                            table.tometadata(target_metadata)

                ops_list = produce_migrations(context, target_metadata).upgrade_ops_list

                if ops_list[0].ops:
                    logging.info(f"⚡ [APP] Detected schema changes in '{db_name}' database. Applying migrations...")
                
                    op = Operations(context)
                    for upgrade_ops in ops_list:
                        for schema_element in upgrade_ops.ops:
                            if hasattr(schema_element, 'ops'):
                                for table_op in schema_element.ops:
                                    if dialect == 'sqlite' and isinstance(table_op, AlterColumnOp):
                                        alter_drop_not_null_sqlite(connection, table_op.table_name, table_op.column_name)
                                    else:
                                        op.invoke(table_op)
                            else:
                                if dialect == 'sqlite' and isinstance(schema_element, AlterColumnOp):
                                    alter_drop_not_null_sqlite(connection, schema_element.table_name, schema_element.column_name)
                                else:
                                    op.invoke(schema_element)
                            
                        for diff in upgrade_ops.as_diffs():
                            if diff[0] == 'add_table':
                                table = diff[1]
                                logging.info(f"🆕 [APP] Creating new table: {table.name}")
                            
                            elif diff[0] == 'remove_table':
                                table_name = diff[1].name
                                logging.info(f"🗑️ [APP] Removing table: {table_name}")
                            
                            elif diff[0] == 'add_column':
                                table_name, column = diff[2], diff[3]
                                logging.info(f"➕ [APP] Adding column: {column.name} to table {table_name}")
                            
                            elif diff[0] == 'remove_column':
                                table_name, column = diff[2], diff[3]
                                logging.info(f"➖ [APP] Removing column: {column.name} from table {table_name}")

                    logging.info(f"✅ [APP] Schema migrations for '{db_name}' database completed successfully")
                else:
                    logging.info(f"✅ [APP] No schema changes detected for '{db_name}' database")

        finally:
            connection.close()

    @staticmethod
    def get_session(db_name: str = "default"):
        """
        Provide a session object for the specified database.

        Args:
            db_name (str): The name of the database connection.

        Returns:
            sqlalchemy.orm.Session: A session for the specified database.

        Raises:
            ValueError: If the database name is invalid.
        """
        if db_name not in SESSION_FACTORIES:
            raise ValueError(f"❌ Invalid database name: {db_name}")
        return SESSION_FACTORIES[db_name]

    @staticmethod
    def shutdown():
        """
        Cleanly dispose of all database connections.
        """
        for name, engine in ENGINES.items():
            engine.dispose()
        logging.info("🛑 All database connections have been closed.")
