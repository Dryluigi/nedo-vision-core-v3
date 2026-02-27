# init DeepStream first
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
Gst.init(None)

from core_v3.database.DatabaseManager import DatabaseManager
from core_v3.modules.pipeline_executor.pipeline_executor import PipelineExecutor

def main():
    run()

def run():
    # DB
    DatabaseManager.init_databases(storage_path="/mnt/linux-data/nedo-vision/data")

    # Import executor but DO NOT create sync service
    import queue

    update_queue = queue.Queue()

    # Pass None for sync service
    executor = PipelineExecutor(
        update_queue,
        None,
        None,
        None
    )

    loop = GLib.MainLoop()

    def start():
        executor.start("testtt")
        return False

    GLib.idle_add(start)
    loop.run()

if __name__ == "__main__":
    main()
