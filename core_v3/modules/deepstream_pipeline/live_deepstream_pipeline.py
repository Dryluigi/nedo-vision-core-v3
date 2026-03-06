import threading
import time
import queue
from typing import Dict

import pyds
from gi.repository import Gst

from .constant import PIPELINE_STATUS_RUNNING, PIPELINE_STATUS_STARTING, PIPELINE_STATUS_STOPPED, PIPELINE_STATUS_STOPPING
from .deepstream_pipeline_interface import DeepstreamPipelineInterface
from ..triton_model_manager.triton_model_manager import TritonModelManager

import time

class LiveRtspDeepstreamPipeline(DeepstreamPipelineInterface):

    def __init__(
        self,
        pipeline_id: str,
        pipeline_name: str,
        update_event_queue: queue.Queue,
        rtsp_url: str,
        triton_model_manager: TritonModelManager,
        rtmp_location: str = "rtmp://host.containers.internal:1935/live/cctv-5",
        output_width: int = 1280,
        output_height: int = 720,
    ):
        self._triton_model_manager = triton_model_manager

        self._play_thread = None
        self._is_playing = False
        self._status = PIPELINE_STATUS_STOPPED
        self._first_frame_sent = False

        self._update_event_queue = update_event_queue
        self._pipeline_id = pipeline_id
        self._pipeline_name = pipeline_name
        self._rtsp_url = rtsp_url
        self._rtmp_location = rtmp_location
        self._output_width = output_width
        self._output_height = output_height

        self._pipeline = Gst.Pipeline.new("deepstream-pipeline")
        self._bus = None
        self._source_bin = None

        self._build_pipeline()

    # -------------------------------------------------
    # PIPELINE BUILD
    # -------------------------------------------------

    def _build_pipeline(self):

        self.streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
        self.streammux.set_property("batch-size", 1)
        self.streammux.set_property("width", self._output_width)
        self.streammux.set_property("height", self._output_height)
        self.streammux.set_property("live-source", True)
        self.streammux.set_property("sync-inputs", False)
        self.streammux.set_property("batched-push-timeout", 40000)
        self.streammux.set_property("attach-sys-ts", True)

        self.pgie_queue = Gst.ElementFactory.make("queue", "pgie_queue")

        self.pgie = Gst.ElementFactory.make(
            "nvinferserver",
            "primary-inference"
        )
        self.pgie.set_property(
            "config-file-path",
            "/app/config/deepstream-inferserver-yolo.txt"
        )
        self.pgie.set_property("interval", 5)

        self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        self.tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
        self.tracker.set_property("ll-config-file", "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")

        self.conv = Gst.ElementFactory.make("nvvideoconvert", "conv")

        self.conv_caps = Gst.ElementFactory.make("capsfilter", "conv-caps")
        self.conv_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=RGBA"
        ))

        self.osd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        self.osd.set_property("process-mode", 0)

        self.enc_conv = Gst.ElementFactory.make("nvvideoconvert", "enc-conv")
        self.enc_caps = Gst.ElementFactory.make("capsfilter", "enc-caps")
        self.enc_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=I420"
        ))

        self.encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264-encoder")
        self.encoder.set_property("bitrate", 4000000)
        self.encoder.set_property("iframeinterval", 30)

        self.h264parse = Gst.ElementFactory.make("h264parse", "h264parse")
        h264_src_pad = self.h264parse.get_static_pad("src")
        h264_src_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._first_frame_probe,
            None
        )


        self.flvmux = Gst.ElementFactory.make("flvmux", "flvmux")
        self.flvmux.set_property("streamable", True)

        self.rtmpsink = Gst.ElementFactory.make("rtmpsink", "rtmpsink")
        self.rtmpsink.set_property("location", self._rtmp_location)
        self.rtmpsink.set_property("sync", True)

        # Add elements
        elements = [
            self.streammux,
            self.pgie_queue,
            self.pgie,
            self.tracker,
            self.conv,
            self.conv_caps,
            self.osd,
            self.enc_conv,
            self.enc_caps,
            self.encoder,
            self.h264parse,
            self.flvmux,
            self.rtmpsink,
        ]

        for elem in elements:
            self._pipeline.add(elem)

        # Link elements
        self.streammux.link(self.pgie_queue)
        self.pgie_queue.link(self.pgie)
        self.pgie.link(self.tracker)
        self.tracker.link(self.conv)
        self.conv.link(self.conv_caps)      # ← add
        self.conv_caps.link(self.osd)       # ← change from conv.link(self.osd)
        self.osd.link(self.enc_conv)
        self.enc_conv.link(self.enc_caps)
        self.enc_caps.link(self.encoder)
        self.encoder.link(self.h264parse)
        self.h264parse.link(self.flvmux)
        self.flvmux.link(self.rtmpsink)

        # Add OSD probe
        osd_sink_pad = self.osd.get_static_pad("sink")
        osd_sink_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._osd_probe,
            None,
        )

        # Bus
        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message", self._bus_call)

    # -------------------------------------------------
    # SOURCE BIN (NVMM ZERO COPY)
    # -------------------------------------------------

    def _create_rtsp_source_bin(self, uri, index):
        bin = Gst.Bin.new(f"rtsp-source-bin-{index}")

        rtspsrc = Gst.ElementFactory.make("rtspsrc", None)
        rtspsrc.set_property("location", uri)
        rtspsrc.set_property("latency", 200)  # important for live stream
        rtspsrc.set_property("drop-on-latency", True)
        rtspsrc.set_property("protocols", 4)  # TCP only (more stable)

        depay = Gst.ElementFactory.make("rtph264depay", None)
        h264parse = Gst.ElementFactory.make("h264parse", None)
        decoder = Gst.ElementFactory.make("nvv4l2decoder", None)
        conv = Gst.ElementFactory.make("nvvideoconvert", None)
        capsfilter = Gst.ElementFactory.make("capsfilter", None)

        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM)")
        )

        for elem in [rtspsrc, depay, h264parse, decoder, conv, capsfilter]:
            bin.add(elem)

        depay.link(h264parse)
        h264parse.link(decoder)
        decoder.link(conv)
        conv.link(capsfilter)

        def on_pad_added(src, pad):
            caps = pad.get_current_caps()
            name = caps.to_string()

            if "application/x-rtp" in name:
                sink_pad = depay.get_static_pad("sink")
                if not sink_pad.is_linked():
                    pad.link(sink_pad)

        rtspsrc.connect("pad-added", on_pad_added)

        ghost_pad = Gst.GhostPad.new(
            "src",
            capsfilter.get_static_pad("src")
        )
        bin.add_pad(ghost_pad)

        return bin

    # -------------------------------------------------
    # PLAY
    # -------------------------------------------------

    def play(self):

        if self._is_playing:
            return
        
        self._publish_status(
            PIPELINE_STATUS_STARTING,
            "Pipeline is starting"
        )

        # Build pipeline
        self._source_bin = self._create_rtsp_source_bin(
            self._rtsp_url,
            0
        )

        self._pipeline.add(self._source_bin)

        sink_pad = self.streammux.request_pad_simple("sink_0")
        src_pad = self._source_bin.get_static_pad("src")
        src_pad.link(sink_pad)

        self._play_thread = threading.Thread(target=self._play_background)
        self._play_thread.start()

    def _play_background(self):
        self._triton_model_manager.request_model_access(self._pipeline_id, "yolo")

        self._triton_model_manager.wait_model_till_ready("yolo")

        self._source_bin.sync_state_with_parent()
        self._pipeline.set_state(Gst.State.PLAYING)

        self._is_playing = True

        print(f"[INFO]  Pipeline started → {self._rtmp_location}")

    # -------------------------------------------------
    # STOP
    # -------------------------------------------------

    def stop(self):

        if not self._is_playing:
            return
        
        self._publish_status(
            PIPELINE_STATUS_STOPPING,
            "Pipeline is stopping"
        )

        self._pipeline.set_state(Gst.State.NULL)

        # Release streammux pad
        if self.streammux:
            sinkpad = self.streammux.get_static_pad("sink_0")
            if sinkpad:
                self.streammux.release_request_pad(sinkpad)

        # Remove source bin
        if self._source_bin:
            self._pipeline.remove(self._source_bin)
            self._source_bin = None

        self._is_playing = False

        self._publish_status(
            PIPELINE_STATUS_STOPPED,
            "Pipeline stopped"
        )
    
    def get_metadata(self) -> Dict:
        return {
            "pipeline_id": self._pipeline_id,
            "pipeline_name": self._pipeline_name,
            "pipeline_status": self._status,
            "source_type_code": "live",
            "source_file_path": "",
            "source_url": self._rtsp_url
        }

    # -------------------------------------------------
    # BUS
    # -------------------------------------------------

    def _bus_call(self, bus, message):

        t = message.type

        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print("ERROR:", err, dbg)

            self._publish_status(
                PIPELINE_STATUS_STOPPED,
                f"Pipeline error: {err}"
            )

            self.stop()

        elif t == Gst.MessageType.EOS:
            print("EOS received")

            # TODO: stop

        return True
    
    def _restart_pipeline(self):
        self._publish_status("restarting", "Restarting pipeline after EOS")

        self._pipeline.set_state(Gst.State.NULL)
        time.sleep(0.5)
        self._pipeline.set_state(Gst.State.PLAYING)

        self._first_frame_sent = False


    # -------------------------------------------------
    # OSD PROBE
    # -------------------------------------------------

    def _osd_probe(self, pad, info, user_data):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch_meta.frame_meta_list

        # Use known pipeline output size (safe fallback)
        frame_w = self._output_width
        frame_h = self._output_height

        # ---- Safe clamp helper ----
        def safe_offset(value: int, max_value: int):
            if value < 0:
                return 0
            if value > max_value:
                return max_value
            return value

        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame_meta.obj_meta_list

            while l_obj:
                try:
                    obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # 🎯 Apply only for PERSON
                if obj.class_id == 0:

                    # Remove default label safely
                    obj.text_params.display_text = ""
                    obj.text_params.set_bg_clr = 0
                    obj.text_params.font_params.font_size = 0

                    r = obj.rect_params
                    x = int(r.left)
                    y = int(r.top)
                    w = int(r.width)
                    h = int(r.height)

                    # ───────── Base Thin Neutral Rectangle ─────────
                    r.border_width = 2
                    r.border_color.set(0.7, 0.7, 0.7, 1)

                    # ───────── Corner Overlay ─────────
                    corner_len = min(int(w * 0.15), 25)
                    thickness = 4

                    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                    display_meta.num_lines = 8
                    display_meta.num_labels = 2

                    lines = display_meta.line_params

                    for i in range(8):
                        lines[i].line_width = thickness
                        lines[i].line_color.set(0.0, 0.7, 1.0, 1.0)

                    # Top-left
                    lines[0].x1, lines[0].y1 = x, y
                    lines[0].x2, lines[0].y2 = x + corner_len, y
                    lines[1].x1, lines[1].y1 = x, y
                    lines[1].x2, lines[1].y2 = x, y + corner_len

                    # Top-right
                    lines[2].x1, lines[2].y1 = x + w, y
                    lines[2].x2, lines[2].y2 = x + w - corner_len, y
                    lines[3].x1, lines[3].y1 = x + w, y
                    lines[3].x2, lines[3].y2 = x + w, y + corner_len

                    # Bottom-left
                    lines[4].x1, lines[4].y1 = x, y + h
                    lines[4].x2, lines[4].y2 = x + corner_len, y + h
                    lines[5].x1, lines[5].y1 = x, y + h
                    lines[5].x2, lines[5].y2 = x, y + h - corner_len

                    # Bottom-right
                    lines[6].x1, lines[6].y1 = x + w, y + h
                    lines[6].x2, lines[6].y2 = x + w - corner_len, y + h
                    lines[7].x1, lines[7].y1 = x + w, y + h
                    lines[7].x2, lines[7].y2 = x + w, y + h - corner_len

                    # ───────── GRADIENT BACKGROUND ─────────
                    gradient_height = 50
                    steps = 32  # smoother gradient

                    slice_height = max(1, gradient_height // steps)

                    base_r = 0.0
                    base_g = 0.7
                    base_b = 1.0

                    MAX_RECTS = 16  # DeepStream limit

                    remaining = steps
                    current_step = 0

                    while remaining > 0:
                        batch_count = min(remaining, MAX_RECTS)

                        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                        display_meta.num_rects = batch_count

                        for i in range(batch_count):
                            rect = display_meta.rect_params[i]

                            global_index = current_step + i

                            rect.left = x
                            rect.top = y + h - ((global_index + 1) * slice_height)
                            rect.width = w
                            rect.height = slice_height + 1

                            rect.border_width = 0
                            rect.has_bg_color = 1

                            progress = global_index / steps
                            alpha = 0.2 * (1 - progress**2)

                            rect.bg_color.set(base_r, base_g, base_b, alpha)

                        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

                        remaining -= batch_count
                        current_step += batch_count

                    # ───────── TEXT OVERLAY (FINAL CLEAN VERSION) ─────────
                    tracker_id = obj.object_id
                    location_text = getattr(self, "_location_name", "Unknown")
                    confidence_text = f"{obj.confidence:.2f}"

                    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                    display_meta.num_labels = 3
                    texts = display_meta.text_params

                    # Base bottom-left anchor
                    base_x = x + 8
                    base_y = y + h - 16

                    line_spacing = 12

                    # ─────── 1️⃣ TRACKER ID (ABOVE LOCATION NAME IN FOOTER) ───────
                    left_shift = 1      # move left by X pixels
                    up_shift = 4        # move up by Y pixels

                    texts[0].display_text = f"{tracker_id}"
                    texts[0].x_offset = safe_offset(base_x - left_shift, frame_w - 1)
                    texts[0].y_offset = safe_offset(base_y - line_spacing - up_shift, frame_h - 1)
                    texts[0].font_params.font_name = "Serif"
                    texts[0].font_params.font_size = 8
                    texts[0].font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                    texts[0].set_bg_clr = 0

                    # ─────── 2️⃣ LOCATION NAME ───────
                    texts[1].display_text = location_text
                    texts[1].x_offset = safe_offset(base_x, frame_w - 1)
                    texts[1].y_offset = safe_offset(base_y, frame_h - 1)
                    texts[1].font_params.font_name = "Serif"
                    texts[1].font_params.font_size = 6
                    texts[1].font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                    texts[1].set_bg_clr = 0

                    # ─────── 3️⃣ CONFIDENCE (BOTTOM RIGHT) ───────
                    char_width_estimate = 10
                    approx_text_width = len(confidence_text) * char_width_estimate

                    texts[2].display_text = confidence_text
                    texts[2].x_offset = safe_offset(x + w - approx_text_width + 8, frame_w - 1)
                    texts[2].y_offset = safe_offset(base_y, frame_h - 1)
                    texts[2].font_params.font_name = "Serif"
                    texts[2].font_params.font_size = 6
                    texts[2].font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                    texts[2].set_bg_clr = 0

                    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK
    
    def _first_frame_probe(self, pad, info, u_data):
        if not self._first_frame_sent:
            self._first_frame_sent = True

            self._publish_status(
                PIPELINE_STATUS_RUNNING,
                "Pipeline is running (frames reaching RTMP)"
            )

        return Gst.PadProbeReturn.OK
    
    def _publish_status(self, status: str, message: str):
        self._status = status

        if self._update_event_queue:
            self._update_event_queue.put({
                "pipeline_id": self._pipeline_id,
                "pipeline_name": self._pipeline_name,
                "status": status,
                "message": message,
                "timestamp": time.time(),
            })
