import threading
import queue
import time
from typing import Dict

from gi.repository import Gst
import pyds

from .constant import (
    PIPELINE_STATUS_RUNNING,
    PIPELINE_STATUS_STARTING,
    PIPELINE_STATUS_STOPPED,
    PIPELINE_STATUS_STOPPING,
)
from .deepstream_pipeline_interface import DeepstreamPipelineInterface
from ..triton_model_manager.triton_model_manager import TritonModelManager

# TODO: add icon on top of bounding box

class FileDeepstreamPipeline(DeepstreamPipelineInterface):

    def __init__(
        self,
        pipeline_id: str,
        pipeline_name: str,
        update_event_queue: queue.Queue,
        file_uri: str,
        infer_config: str,
        rtmp_uri: str,
        triton_model_manager: TritonModelManager,
        output_width: int = 1280,
        output_height: int = 720,
        target_fps: int = 30,
        encode_bitrate: int = 2000,
        gpu_id: int = 0,
    ):
        self.pipeline_id    = pipeline_id
        self.pipeline_name  = pipeline_name
        self.file_uri       = file_uri
        self.infer_config   = infer_config
        self.rtmp_uri       = rtmp_uri
        self.output_width   = output_width
        self.output_height  = output_height
        self.target_fps     = target_fps
        self.encode_bitrate = encode_bitrate
        self.gpu_id         = gpu_id
        self.triton_model_manager = triton_model_manager

        self._update_event_queue = update_event_queue
        self._status             = PIPELINE_STATUS_STOPPED
        self._first_frame_sent   = False
        
        self._osd_pad = None
        self._h264_src_pad = None
        self._first_frame_probe_id = None
        self._osd_probe_id = None

        self._play_thread = None

        self._pipeline = self._build_pipeline()

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call)

    # ── Interface ────────────────────────────────────────────────────────────

    def play(self):
        self._publish_status(PIPELINE_STATUS_STARTING, "Pipeline is starting")

        self._play_thread = threading.Thread(target=self._play_background)
        self._play_thread.start()

    def _play_background(self):
        self.triton_model_manager.request_model_access(self.pipeline_id, "yolo")

        self.triton_model_manager.wait_model_till_ready("yolo")

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._publish_status(PIPELINE_STATUS_STOPPED, "Pipeline failed to start")
            raise RuntimeError("Pipeline failed to transition to PLAYING state.")

        print(f"[INFO]  Pipeline started → {self.rtmp_uri}")

    def stop(self):
        self._publish_status(PIPELINE_STATUS_STOPPING, "Pipeline is stopping")
        
        if self._osd_probe_id:
            self._osd_pad.remove_probe(self._osd_probe_id)

        if self._first_frame_probe_id:
            self._h264_src_pad.remove_probe(self._first_frame_probe_id)

        self._pipeline.send_event(Gst.Event.new_eos())
        self._pipeline.set_state(Gst.State.NULL)
        self.triton_model_manager.release_model_access(self.pipeline_id)

        self._first_frame_sent = False
        self._publish_status(PIPELINE_STATUS_STOPPED, "Pipeline stopped")
        print("[INFO]  Pipeline stopped.")

    def get_metadata(self) -> Dict:
        return {
            "pipeline_id":      self.pipeline_id,
            "pipeline_name":    self.pipeline_name,
            "pipeline_status":  self._status,
            "source_type_code": "file",
            "source_file_path": self.file_uri,
            "source_url":       "",
        }

    # ── Build ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make(factory: str, name: str) -> Gst.Element:
        el = Gst.ElementFactory.make(factory, name)
        if not el:
            raise RuntimeError(f"Could not create element '{factory}' as '{name}'")
        return el

    def _build_pipeline(self) -> Gst.Pipeline:
        pipeline = Gst.Pipeline()

        # 1. Streammux
        streammux = self._make("nvstreammux", "stream-muxer")
        pipeline.add(streammux)
        streammux.set_property("batch-size",           1)
        streammux.set_property("width",                self.output_width)
        streammux.set_property("height",               self.output_height)
        streammux.set_property("gpu-id",               self.gpu_id)
        streammux.set_property("live-source",          False)
        streammux.set_property("batched-push-timeout", 400)
        streammux.set_property("nvbuf-memory-type",    0)
        streammux.set_property("attach-sys-ts",        True)

        # 2. Single source
        print(f"[INFO]  Adding source : {self.file_uri}")
        src_bin = self._create_source_bin(0, self.file_uri)
        pipeline.add(src_bin)

        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            sinkpad = streammux.request_pad_simple("sink_0")
        if not sinkpad:
            raise RuntimeError("Could not get streammux pad 'sink_0'")

        srcpad = src_bin.get_static_pad("src")
        if srcpad:
            if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link {self.file_uri} to streammux")
        else:
            def _link(element, pad, sp=sinkpad):
                if pad.get_name() != "src" or sp.is_linked():
                    return
                ok = pad.link(sp) == Gst.PadLinkReturn.OK
                print(f"[{'INFO' if ok else 'ERROR'}]  "
                      f"source-bin-0 → streammux sink_0 {'linked' if ok else 'FAILED'}")
            src_bin.connect("pad-added", _link)

        # 3. Mux → NV12
        mux_conv = self._make("nvvideoconvert", "mux-conv")
        mux_conv.set_property("gpu-id", self.gpu_id)
        mux_caps = self._make("capsfilter", "mux-caps")
        mux_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=NV12"))

        # 4. nvinferserver
        pgie = self._make("nvinferserver", "primary-inference")
        pgie.set_property("config-file-path", self.infer_config)
        pgie.set_property("interval", 5)

        # 5. nvtracker for tracker id
        tracker = self._make("nvtracker", "tracker")
        tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
        tracker.set_property("ll-config-file", "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")
        tracker.set_property("gpu-id", self.gpu_id)

        # 6. Pre-tiler
        pre_conv = self._make("nvvideoconvert", "pre-tiler-conv")
        pre_conv.set_property("gpu-id", self.gpu_id)
        pre_caps = self._make("capsfilter", "pre-tiler-caps")
        pre_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=NV12"))

        # 7. Tiler — 1×1 for single source
        tiler = self._make("nvmultistreamtiler", "tiler")
        tiler.set_property("rows",    1)
        tiler.set_property("columns", 1)
        tiler.set_property("width",   self.output_width)
        tiler.set_property("height",  self.output_height)
        tiler.set_property("gpu-id",  self.gpu_id)

        # 8. Post-tiler → RGBA for OSD
        post_conv = self._make("nvvideoconvert", "post-tiler-conv")
        post_conv.set_property("gpu-id", self.gpu_id)
        post_caps = self._make("capsfilter", "post-tiler-caps")
        post_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=RGBA"))

        # 9. OSD
        nvosd = self._make("nvdsosd", "osd")
        nvosd.set_property("process-mode", 0)
        nvosd.set_property("display-text", 1)

        # 10. Encode converter
        enc_conv = self._make("nvvideoconvert", "enc-conv")
        enc_conv.set_property("gpu-id", self.gpu_id)
        enc_caps = self._make("capsfilter", "enc-caps")
        enc_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=I420"))

        # 11. GPU H.264 encoder
        encoder = self._make("nvv4l2h264enc", "h264-encoder")
        encoder.set_property("bitrate",        self.encode_bitrate * 1000)
        encoder.set_property("iframeinterval", self.target_fps * 2)

        # 12. H.264 parser — first-frame probe attached here to detect
        #     when encoded frames are actually reaching the muxer/RTMP
        h264parse = self._make("h264parse", "h264-parse")
        h264parse.set_property("config-interval", -1)
        self._h264_src_pad = h264parse.get_static_pad("src")
        self._first_frame_probe_id = self._h264_src_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._first_frame_probe,
            None,
        )

        # 13. FLV muxer
        flvmux = self._make("flvmux", "flv-mux")
        flvmux.set_property("streamable", True)

        # 14. RTMP sink
        rtmpsink = self._make("rtmpsink", "rtmp-sink")
        rtmpsink.set_property("location", self.rtmp_uri)
        rtmpsink.set_property("sync",  True)
        rtmpsink.set_property("async", False)

        # Add & link
        for el in [mux_conv, mux_caps, pgie, tracker,
                   pre_conv, pre_caps, tiler,
                   post_conv, post_caps, nvosd,
                   enc_conv, enc_caps, encoder, h264parse, flvmux, rtmpsink]:
            pipeline.add(el)

        links = [
            (streammux, mux_conv), (mux_conv, mux_caps), (mux_caps, pgie),
            (pgie, tracker), (tracker, pre_conv), (pre_conv, pre_caps),  (pre_caps, tiler),
            (tiler, post_conv),    (post_conv, post_caps), (post_caps, nvosd),
            (nvosd, enc_conv),     (enc_conv, enc_caps),   (enc_caps, encoder),
            (encoder, h264parse),  (h264parse, flvmux),    (flvmux, rtmpsink),
        ]
        for src_el, dst_el in links:
            if not src_el.link(dst_el):
                raise RuntimeError(
                    f"Failed to link {src_el.get_name()} → {dst_el.get_name()}")

        # OSD probe — collect detection metadata
        self._osd_pad = nvosd.get_static_pad("sink")
        if self._osd_pad:
            self._osd_probe_id = self._osd_pad.add_probe(Gst.PadProbeType.BUFFER, self._osd_probe, None)

        return pipeline

    def _create_source_bin(self, index: int, uri: str) -> Gst.Bin:
        bin_name = f"source-bin-{index}"
        nbin = Gst.Bin.new(bin_name)

        uri_src = self._make("nvurisrcbin", f"uri-src-{index}")
        uri_src.set_property("uri",                 uri)
        uri_src.set_property("gpu-id",              self.gpu_id)
        uri_src.set_property("drop-frame-interval", 2)
        uri_src.set_property("file-loop",           True)
        nbin.add(uri_src)

        def _on_pad_added(src, pad):
            caps = pad.get_current_caps() or pad.query_caps()
            if not caps or not caps.get_structure(0).get_name().startswith("video"):
                return
            if nbin.get_static_pad("src"):
                return
            ghost = Gst.GhostPad.new("src", pad)
            ghost.set_active(True)
            nbin.add_pad(ghost)
            print(f"[INFO]  Ghost pad created on {bin_name}")

        def _child_added(child_proxy, obj, name, _):
            if name.startswith("decodebin"):
                obj.connect("child-added", _child_added, None)

        uri_src.connect("pad-added",   _on_pad_added)
        uri_src.connect("child-added", _child_added, None)
        return nbin

    # ── Probes ───────────────────────────────────────────────────────────────

    def _first_frame_probe(self, pad, info, user_data):
        """Fires once when the first encoded frame exits h264parse → RTMP."""
        if not self._first_frame_sent:
            self._first_frame_sent = True
            self._publish_status(
                PIPELINE_STATUS_RUNNING,
                "Pipeline is running (frames reaching RTMP)",
            )
        return Gst.PadProbeReturn.OK

    def _osd_probe(self, pad, info, user_data):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch_meta.frame_meta_list

        # Use known pipeline output size (safe fallback)
        frame_w = self.output_width
        frame_h = self.output_height

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

    # ── Bus ──────────────────────────────────────────────────────────────────

    def _bus_call(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("[INFO]  EOS received.")
        elif t == Gst.MessageType.WARNING:
            err, dbg = message.parse_warning()
            print(f"[WARN]  {err} | {dbg}")
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"[ERROR] {err} | {dbg}")
            self._publish_status(PIPELINE_STATUS_STOPPED, f"Pipeline error: {err}")
        return True

    # ── Status publish ───────────────────────────────────────────────────────

    def _publish_status(self, status: str, message: str):
        self._status = status
        if self._update_event_queue:
            self._update_event_queue.put({
                "pipeline_id":   self.pipeline_id,
                "pipeline_name": self.pipeline_name,
                "status":        status,
                "message":       message,
                "timestamp":     time.time(),
            })