import time
import queue
import argparse
import signal
import sys
import traceback
import logging
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

Gst.init(None)

import faulthandler
from .database.DatabaseManager import DatabaseManager
from .repositories.WorkerSourcePipelineRepository import WorkerSourcePipelineRepository
from .repositories.WorkerSourceRepository import WorkerSourceRepository
from .modules.capture_processing_service.capture_processing_service import CaptureProcessingService
from .modules.drawing.DrawingUtils import DrawingUtils
from .modules.pipeline_sync_service.pipeline_sync_service import PipelineSyncService
from .modules.pipeline_executor.pipeline_executor import PipelineExecutor
from .modules.triton_model_manager.triton_model_manager import TritonModelManager
from .utils.RTMPUrl import RTMPUrl

# Enable fault handler to get a traceback
faulthandler.enable()

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logging.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Nedo Vision Core Library CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start core service with default settings (includes auto video sharing daemon)
  nedo-core run

  # Start with custom drawing assets path
  nedo-core run --drawing-assets /path/to/assets

  # Start with debug logging
  nedo-core run --log-level DEBUG

  # Start with custom storage path and RTMP server
  nedo-core run --storage-path /path/to/storage --rtmp-server rtmp://server.com:1935/live

  # Start without automatic video sharing daemon
  nedo-core run --disable-video-sharing-daemon

  # Start with all custom parameters
  nedo-core run --drawing-assets /path/to/assets --log-level DEBUG --storage-path /data --rtmp-server rtmp://server.com:1935/live

  # Run system diagnostics
  nedo-core doctor

Video Sharing Daemon:
  By default, the core service automatically starts video sharing daemons for devices
  as needed. This enables multiple processes to access the same video device simultaneously
  without "device busy" errors. Use --disable-video-sharing-daemon to turn this off.

Detection Callbacks:
  The core service supports detection callbacks for extensible event handling.
  See example_callbacks.py for usage examples.
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser(
        'run',
        help='Start the core service',
        description='Start the Nedo Vision Core Service'
    )
    
    # Doctor command
    doctor_parser = subparsers.add_parser(
        'doctor',
        help='Run system diagnostics',
        description='Run diagnostic checks for system requirements and dependencies'
    )
    
    run_parser.add_argument(
        "--drawing-assets",
        default=None,
        help="Path to drawing assets directory (optional, uses bundled assets by default)"
    )
    
    run_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    run_parser.add_argument(
        "--storage-path",
        default="data",
        help="Storage path for databases and files (default: data)"
    )
    
    run_parser.add_argument(
        "--rtmp-server",
        default="rtmp://live.vision.sindika.co.id:1935/live",
        help="RTMP server URL for video streaming (default: rtmp://live.vision.sindika.co.id:1935/live)"
    )
    
    run_parser.add_argument(
        "--disable_video_sharing_daemon",
        action="store_true",
        default=False,
        help="Disable automatic video sharing daemon management (default: False)"
    )

    run_parser.add_argument(
        "--rtmp-publish-query-strings",
        default="",
        help="RTMP additional query strings. Usually for authentication. (format: user=admin&pass=admin123). Don't forget to escape the '&' if via CLI."
    )
    
    parser.add_argument(
        "--version",
        action="version"
        # version=f"nedo-vision-core {__version__}"
    )
    
    args = parser.parse_args()
    
    # Handle subcommands
    if args.command == 'run':
        run_core_service(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_core_service(args):
    """Run the core service with the provided arguments."""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        repo_root = Path(__file__).resolve().parent.parent
        drawing_assets_path = Path(args.drawing_assets).resolve() if args.drawing_assets else repo_root / "assets" / "drawing_assets"

        # Determine video sharing daemon setting
        enable_daemon = True  # Default
        if hasattr(args, 'disable_video_sharing_daemon') and args.disable_video_sharing_daemon:
            enable_daemon = False

        DrawingUtils.initialize(str(drawing_assets_path))

        # Initialize Database with storage path
        DatabaseManager.init_databases(storage_path=args.storage_path)

        RTMPUrl.configure(
            rtmp_server_url=args.rtmp_server,
            rtmp_publish_query_strings=args.rtmp_publish_query_strings
        )

        pipeline_repo = WorkerSourcePipelineRepository()
        source_repo = WorkerSourceRepository()
        pipeline_sync_service = PipelineSyncService(pipeline_repo)
        triton_model_manager = TritonModelManager()
        capture_processing_service = CaptureProcessingService()
        update_queue = queue.Queue()
        pipeline_executor = PipelineExecutor(
            update_queue,
            pipeline_sync_service,
            pipeline_repo,
            source_repo,
            triton_model_manager,
            capture_processing_service,
        )

        pipeline_sync_service.subscribe_update(pipeline_executor)
        
        logger.info("🚀 Starting Nedo Vision Core V3...")
        logger.info(f"🎨 Drawing Assets: {drawing_assets_path}")
        logger.info(f"📝 Log Level: {args.log_level}")
        logger.info(f"💾 Storage Path: {args.storage_path}")
        logger.info(f"📡 RTMP Server: {args.rtmp_server}")
        logger.info(f"🔗 Video Sharing Daemon: {'Enabled' if enable_daemon else 'Disabled'}")
        logger.info("Press Ctrl+C to stop the service")
        
        # # Start the service
        # success = service.run()
        
        # if success:
        #     logger.info("✅ Core service completed successfully")
        # else:
        #     logger.error("❌ Core service failed")
        #     sys.exit(1)

        loop = GLib.MainLoop()

        try:
            loop.run()
        except KeyboardInterrupt:
            logger.info("Stopping service...")
            loop.quit()
        finally:
            capture_processing_service.stop_all()
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 
