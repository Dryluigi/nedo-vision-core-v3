import threading
import time
import queue
from typing import Dict

import pyds
from gi.repository import Gst, GLib

from .constant import PIPELINE_STATUS_RUNNING, PIPELINE_STATUS_STARTING, PIPELINE_STATUS_STOPPED, PIPELINE_STATUS_STOPPING
from .deepstream_pipeline_interface import DeepstreamPipelineInterface

import uuid
import time

class LiveRtspDeepstreamPipeline(DeepstreamPipelineInterface):

    def __init__(
        self,
        pipeline_id: str,
        pipeline_name: str,
        update_event_queue: queue.Queue,
        rtsp_url: str,
        rtmp_location: str = "rtmp://host.containers.internal:1935/live/cctv-5",
        width: int = 1280,
        height: int = 720,
    ):
        self._is_playing = False
        self._status = PIPELINE_STATUS_STOPPED
        self._first_frame_sent = False

        self._update_event_queue = update_event_queue
        self._pipeline_id = pipeline_id
        self._pipeline_name = pipeline_name
        self._rtsp_url = rtsp_url
        self._rtmp_location = rtmp_location
        self._width = width
        self._height = height

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
        self.streammux.set_property("width", self._width)
        self.streammux.set_property("height", self._height)
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
        self.pgie.set_property("interval", 0)

        self.conv = Gst.ElementFactory.make("nvvideoconvert", "conv")
        self.osd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

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
            self.conv,
            self.osd,
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
        self.pgie.link(self.conv)
        self.conv.link(self.osd)
        self.osd.link(self.encoder)
        self.encoder.link(self.h264parse)
        self.h264parse.link(self.flvmux)
        self.flvmux.link(self.rtmpsink)

        # Add OSD probe
        osd_sink_pad = self.osd.get_static_pad("sink")
        osd_sink_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._osd_sink_pad_buffer_probe,
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

        self._source_bin = self._create_rtsp_source_bin(
            self._rtsp_url,
            0
        )


        self._pipeline.add(self._source_bin)

        sink_pad = self.streammux.request_pad_simple("sink_0")
        src_pad = self._source_bin.get_static_pad("src")
        src_pad.link(sink_pad)

        self._source_bin.sync_state_with_parent()

        self._pipeline.set_state(Gst.State.PLAYING)

        self._is_playing = True

        print(f"Publishing at {self._rtmp_location}")

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

    def _osd_sink_pad_buffer_probe(self, pad, info, u_data):

        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

            obj_count = 0
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                obj_count += 1
                l_obj = l_obj.next

            l_frame = l_frame.next

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
