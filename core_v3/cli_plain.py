# import time
import queue
# import argparse
# import signal
# import sys
# import traceback
# import logging

import gi
from gi.repository import GLib, Gst
gi.require_version("Gst", "1.0")
Gst.init(None)

# import faulthandler
# from .database.DatabaseManager import DatabaseManager
# from .repositories.WorkerSourcePipelineRepository import WorkerSourcePipelineRepository
# from .repositories.WorkerSourceRepository import WorkerSourceRepository
# from .modules.pipeline_sync_service.pipeline_sync_service import PipelineSyncService
from .modules.pipeline_executor.pipeline_executor import PipelineExecutor

# Enable fault handler to get a traceback
# faulthandler.enable()

# def signal_handler(signum, frame):
#     """Handle system signals for graceful shutdown"""
#     logging.info(f"Received signal {signum}, shutting down...")
#     sys.exit(0)


def main():
    """Main CLI entry point."""
#     parser = argparse.ArgumentParser(
#         description="Nedo Vision Core Library CLI",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Start core service with default settings (includes auto video sharing daemon)
#   nedo-core run

#   # Start with custom drawing assets path
#   nedo-core run --drawing-assets /path/to/assets

#   # Start with debug logging
#   nedo-core run --log-level DEBUG

#   # Start with custom storage path and RTMP server
#   nedo-core run --storage-path /path/to/storage --rtmp-server rtmp://server.com:1935/live

#   # Start without automatic video sharing daemon
#   nedo-core run --disable-video-sharing-daemon

#   # Start with all custom parameters
#   nedo-core run --drawing-assets /path/to/assets --log-level DEBUG --storage-path /data --rtmp-server rtmp://server.com:1935/live

#   # Run system diagnostics
#   nedo-core doctor

# Video Sharing Daemon:
#   By default, the core service automatically starts video sharing daemons for devices
#   as needed. This enables multiple processes to access the same video device simultaneously
#   without "device busy" errors. Use --disable-video-sharing-daemon to turn this off.

# Detection Callbacks:
#   The core service supports detection callbacks for extensible event handling.
#   See example_callbacks.py for usage examples.
#         """
#     )
    
    # # Add subcommands
    # subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # # Run command
    # run_parser = subparsers.add_parser(
    #     'run',
    #     help='Start the core service',
    #     description='Start the Nedo Vision Core Service'
    # )
    
    # # Doctor command
    # doctor_parser = subparsers.add_parser(
    #     'doctor',
    #     help='Run system diagnostics',
    #     description='Run diagnostic checks for system requirements and dependencies'
    # )
    
    # run_parser.add_argument(
    #     "--drawing-assets",
    #     default=None,
    #     help="Path to drawing assets directory (optional, uses bundled assets by default)"
    # )
    
    # run_parser.add_argument(
    #     "--log-level",
    #     choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    #     default="INFO",
    #     help="Logging level (default: INFO)"
    # )
    
    # run_parser.add_argument(
    #     "--storage-path",
    #     default="data",
    #     help="Storage path for databases and files (default: data)"
    # )
    
    # run_parser.add_argument(
    #     "--rtmp-server",
    #     default="rtmp://live.vision.sindika.co.id:1935/live",
    #     help="RTMP server URL for video streaming (default: rtmp://live.vision.sindika.co.id:1935/live)"
    # )
    
    # run_parser.add_argument(
    #     "--disable_video_sharing_daemon",
    #     action="store_true",
    #     default=False,
    #     help="Disable automatic video sharing daemon management (default: False)"
    # )

    # run_parser.add_argument(
    #     "--rtmp-publish-query-strings",
    #     default="",
    #     help="RTMP additional query strings. Usually for authentication. (format: user=admin&pass=admin123). Don't forget to escape the '&' if via CLI."
    # )
    
    # parser.add_argument(
    #     "--version",
    #     action="version"
    #     # version=f"nedo-vision-core {__version__}"
    # )
    
    # args = parser.parse_args()
    
    run_core_service(None)
    # # Handle subcommands
    # if args.command == 'run':
    # else:
    #     parser.print_help()
    #     sys.exit(1)


def run_core_service(args):
    """Run the core service with the provided arguments."""
   # Determine video sharing daemon setting
    # enable_daemon = True  # Default
    # if hasattr(args, 'disable_video_sharing_daemon') and args.disable_video_sharing_daemon:
    #     enable_daemon = False

    # Initialize Database with storage path
    # DatabaseManager.init_databases(storage_path=args.storage_path)

    # pipeline_repo = WorkerSourcePipelineRepository()
    # source_repo = WorkerSourceRepository()
    # pipeline_sync_service = PipelineSyncService(pipeline_repo)
    update_queue = queue.Queue()
    pipeline_executor = PipelineExecutor(
        update_queue,
        None,
        None,
        None
    )

    # pipeline_sync_service.subscribe_update(pipeline_executor)
    
    # logger.info("🚀 Starting Nedo Vision Core V3...")
    # if args.drawing_assets:
    #     logger.info(f"🎨 Drawing Assets: {args.drawing_assets}")
    # else:
    #     logger.info("🎨 Drawing Assets: Using bundled assets")
    # logger.info(f"📝 Log Level: {args.log_level}")
    # logger.info(f"💾 Storage Path: {args.storage_path}")
    # logger.info(f"📡 RTMP Server: {args.rtmp_server}")
    # logger.info(f"🔗 Video Sharing Daemon: {'Enabled' if enable_daemon else 'Disabled'}")
    # logger.info("Press Ctrl+C to stop the service")
    
    # # Start the service
    # success = service.run()
    
    # if success:
    #     logger.info("✅ Core service completed successfully")
    # else:
    #     logger.error("❌ Core service failed")
    #     sys.exit(1)

    loop = GLib.MainLoop()

    def start_pipelines():
        pipeline_executor.start("testtt")
        return False  # run once
    
    GLib.idle_add(start_pipelines)

    loop.run()


if __name__ == "__main__":
    main() 