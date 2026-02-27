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


class FileDeepstreamPipeline(DeepstreamPipelineInterface):

    def __init__(
        self,
        pipeline_id: str,
        pipeline_name: str,
        update_event_queue: queue.Queue,
        file_uri: str,
        infer_config: str,
        rtmp_uri: str,
        output_width: int = 1280,
        output_height: int = 720,
        target_fps: int = 30,
        encode_bitrate: int = 4000,
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

        self._update_event_queue = update_event_queue
        self._status             = PIPELINE_STATUS_STOPPED
        self._first_frame_sent   = False

        self._pipeline = self._build_pipeline()

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call)

    # ── Interface ────────────────────────────────────────────────────────────

    def play(self):
        self._publish_status(PIPELINE_STATUS_STARTING, "Pipeline is starting")

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._publish_status(PIPELINE_STATUS_STOPPED, "Pipeline failed to start")
            raise RuntimeError("Pipeline failed to transition to PLAYING state.")

        print(f"[INFO]  Pipeline started → {self.rtmp_uri}")

    def stop(self):
        self._publish_status(PIPELINE_STATUS_STOPPING, "Pipeline is stopping")

        self._pipeline.send_event(Gst.Event.new_eos())
        self._pipeline.set_state(Gst.State.NULL)

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
        streammux.set_property("batched-push-timeout", 40000)
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

        # 5. Pre-tiler
        pre_conv = self._make("nvvideoconvert", "pre-tiler-conv")
        pre_conv.set_property("gpu-id", self.gpu_id)
        pre_caps = self._make("capsfilter", "pre-tiler-caps")
        pre_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=NV12"))

        # 6. Tiler — 1×1 for single source
        tiler = self._make("nvmultistreamtiler", "tiler")
        tiler.set_property("rows",    1)
        tiler.set_property("columns", 1)
        tiler.set_property("width",   self.output_width)
        tiler.set_property("height",  self.output_height)
        tiler.set_property("gpu-id",  self.gpu_id)

        # 7. Post-tiler → RGBA for OSD
        post_conv = self._make("nvvideoconvert", "post-tiler-conv")
        post_conv.set_property("gpu-id", self.gpu_id)
        post_caps = self._make("capsfilter", "post-tiler-caps")
        post_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=RGBA"))

        # 8. OSD
        nvosd = self._make("nvdsosd", "osd")
        nvosd.set_property("process-mode", 0)
        nvosd.set_property("display-text", 1)

        # 9. Encode converter
        enc_conv = self._make("nvvideoconvert", "enc-conv")
        enc_conv.set_property("gpu-id", self.gpu_id)
        enc_caps = self._make("capsfilter", "enc-caps")
        enc_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=I420"))

        # 10. GPU H.264 encoder
        encoder = self._make("nvv4l2h264enc", "h264-encoder")
        encoder.set_property("bitrate",        self.encode_bitrate * 1000)
        encoder.set_property("iframeinterval", self.target_fps * 2)

        # 11. H.264 parser — first-frame probe attached here to detect
        #     when encoded frames are actually reaching the muxer/RTMP
        h264parse = self._make("h264parse", "h264-parse")
        h264parse.set_property("config-interval", -1)
        h264_src_pad = h264parse.get_static_pad("src")
        h264_src_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._first_frame_probe,
            None,
        )

        # 12. FLV muxer
        flvmux = self._make("flvmux", "flv-mux")
        flvmux.set_property("streamable", True)

        # 13. RTMP sink
        rtmpsink = self._make("rtmpsink", "rtmp-sink")
        rtmpsink.set_property("location", self.rtmp_uri)
        rtmpsink.set_property("sync",  True)
        rtmpsink.set_property("async", False)

        # Add & link
        for el in [mux_conv, mux_caps, pgie,
                   pre_conv, pre_caps, tiler,
                   post_conv, post_caps, nvosd,
                   enc_conv, enc_caps, encoder, h264parse, flvmux, rtmpsink]:
            pipeline.add(el)

        links = [
            (streammux, mux_conv), (mux_conv, mux_caps), (mux_caps, pgie),
            (pgie, pre_conv),      (pre_conv, pre_caps),  (pre_caps, tiler),
            (tiler, post_conv),    (post_conv, post_caps), (post_caps, nvosd),
            (nvosd, enc_conv),     (enc_conv, enc_caps),   (enc_caps, encoder),
            (encoder, h264parse),  (h264parse, flvmux),    (flvmux, rtmpsink),
        ]
        for src_el, dst_el in links:
            if not src_el.link(dst_el):
                raise RuntimeError(
                    f"Failed to link {src_el.get_name()} → {dst_el.get_name()}")

        # OSD probe — collect detection metadata
        osd_pad = nvosd.get_static_pad("sink")
        if osd_pad:
            osd_pad.add_probe(Gst.PadProbeType.BUFFER, self._osd_probe, None)

        return pipeline

    def _create_source_bin(self, index: int, uri: str) -> Gst.Bin:
        bin_name = f"source-bin-{index}"
        nbin = Gst.Bin.new(bin_name)

        uri_src = self._make("nvurisrcbin", f"uri-src-{index}")
        uri_src.set_property("uri",                 uri)
        uri_src.set_property("gpu-id",              self.gpu_id)
        uri_src.set_property("drop-frame-interval", 0)
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
        batch_detections = []
        l_frame = batch_meta.frame_meta_list

        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            src_id = frame_meta.source_id
            frame_dets = []
            l_obj = frame_meta.obj_meta_list

            while l_obj:
                try:
                    obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                r = obj.rect_params
                frame_dets.append({
                    "source_id":  src_id,
                    "frame_num":  frame_meta.frame_num,
                    "class_id":   obj.class_id,
                    "label":      obj.obj_label,
                    "confidence": round(obj.confidence, 4),
                    "bbox": {
                        "left":   round(r.left),
                        "top":    round(r.top),
                        "width":  round(r.width),
                        "height": round(r.height),
                    },
                })
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            batch_detections.extend(frame_dets)
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