import sys
import time
from core_v3.modules.deepstream_pipeline.live_deepstream_pipeline import DeepstreamPipeline

if len(sys.argv) < 3:
    print("Usage:")
    print("python test_run_pipeline.py <RTSP_URL> <RTMP_URL>")
    sys.exit(1)

rtsp_url = sys.argv[1]
rtmp_url = sys.argv[2]

pipeline = DeepstreamPipeline(rtsp_url, rtmp_url)

try:
    pipeline.play()
    print("Pipeline running... Press CTRL+C to stop.")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    pipeline.stop()
    print("Pipeline stopped cleanly.")
