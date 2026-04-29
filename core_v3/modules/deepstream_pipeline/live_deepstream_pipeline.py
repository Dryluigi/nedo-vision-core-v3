import threading
import time
import queue
from typing import Dict
import numpy as np

import pyds
from gi.repository import Gst

from .capture_decision_engine import CaptureDecisionEngine
from .constant import PIPELINE_STATUS_RUNNING, PIPELINE_STATUS_STARTING, PIPELINE_STATUS_STOPPED, PIPELINE_STATUS_STOPPING
from .deepstream_pipeline_interface import DeepstreamPipelineInterface
from .person_attribute_aggregator import PersonAttributeAggregator
from .ppe_preview_renderer import PPEPreviewRenderer
from ..capture_processing_service.capture_processing_service import CaptureProcessingService
from ..drawing.FrameDrawer import FrameDrawer
from ..triton_model_manager.triton_model_manager import TritonModelManager

class LiveRtspDeepstreamPipeline(DeepstreamPipelineInterface):

    def __init__(
        self,
        pipeline_id: str,
        pipeline_name: str,
        update_event_queue: queue.Queue,
        rtsp_url: str,
        infer_config: str,
        triton_model_manager: TritonModelManager,
        ai_model_id: str | None = None,
        worker_id: str | None = None,
        worker_source_id: str | None = None,
        location_name: str | None = None,
        capture_decision_engine: CaptureDecisionEngine | None = None,
        person_attribute_aggregator: PersonAttributeAggregator | None = None,
        frame_drawer: FrameDrawer | None = None,
        capture_processing_service: CaptureProcessingService | None = None,
        class_id_to_label: Dict[int, str] | None = None,
        rtmp_location: str = "rtmp://host.containers.internal:1935/live/cctv-5",
        output_width: int = 1280,
        output_height: int = 720,
        target_fps: int = 30,
    ):
        self._triton_model_manager = triton_model_manager
        self._ai_model_id = ai_model_id

        self._play_thread = None
        self._is_playing = False
        self._stop_requested = False
        self._status = PIPELINE_STATUS_STOPPED
        self._first_frame_sent = False

        self._update_event_queue = update_event_queue
        self._pipeline_id = pipeline_id
        self._pipeline_name = pipeline_name
        self._rtsp_url = rtsp_url
        self._infer_config = infer_config
        self._rtmp_location = rtmp_location
        self._output_width = output_width
        self._output_height = output_height
        self._target_fps = target_fps
        self._worker_id = worker_id
        self._worker_source_id = worker_source_id
        self._location_name = location_name or pipeline_name
        self._capture_decision_engine = capture_decision_engine
        self._person_attribute_aggregator = person_attribute_aggregator
        self._capture_processing_service = capture_processing_service
        self._class_id_to_label = class_id_to_label or {
            0: "background",
            1: "helmet",
            2: "no_helmet",
            3: "no_vest",
            4: "person",
            5: "vest",
        }
        self._frame_drawer = frame_drawer or FrameDrawer()
        self._frame_drawer.location_name = self._location_name
        self._preview_renderer = PPEPreviewRenderer(
            frame_drawer=self._frame_drawer,
            class_id_to_label=self._class_id_to_label,
            location_name=self._location_name,
            person_class_id=4,
            attribute_class_ids={1, 2, 3, 5},
            preview_style_hold_seconds=3.0,
        )
        self._capture_appsink = None
        self._capture_bin = None
        self._capture_bin_sink_pad = None
        self._capture_branch_pad = None
        self._post_tee = None
        self._osd_pad = None
        self._osd_probe_id = None
        self._rtmp_sink_pad = None
        self._first_frame_probe_id = None
        self._streammux_sink_pad = None
        self._capture_worker = None
        if self._capture_processing_service is not None:
            self._capture_worker = self._capture_processing_service.get_or_create_worker(
                pipeline_id=self._pipeline_id,
                worker_id=self._worker_id or self._pipeline_id,
                worker_source_id=self._worker_source_id or "",
                frame_drawer=self._frame_drawer,
            )

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
        self.pgie.set_property("config-file-path", self._infer_config)
        self.pgie.set_property("interval", 5)

        self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        self.tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
        self.tracker.set_property("ll-config-file", "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")

        self.conv = Gst.ElementFactory.make("nvvideoconvert", "conv")

        self.conv_caps = Gst.ElementFactory.make("capsfilter", "conv-caps")
        self.conv_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=RGBA"
        ))
        self.post_tee = Gst.ElementFactory.make("tee", "post-tiler-tee")
        self.osd_queue = Gst.ElementFactory.make("queue", "osd-queue")
        self.capture_queue = Gst.ElementFactory.make("queue", "capture-queue")
        self.capture_conv = Gst.ElementFactory.make("nvvideoconvert", "capture-convert")
        self.capture_caps = Gst.ElementFactory.make("capsfilter", "capture-caps")
        self.capture_caps.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))
        self.capture_appsink = Gst.ElementFactory.make("appsink", "capture-appsink")
        self.capture_appsink.set_property("emit-signals", True)
        self.capture_appsink.set_property("sync", False)
        self.capture_appsink.set_property("max-buffers", 30)
        self.capture_appsink.set_property("drop", True)
        self.capture_appsink.connect("new-sample", self._on_new_sample)
        self._capture_appsink = self.capture_appsink

        self.osd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        self.osd.set_property("process-mode", 0)

        self.enc_conv = Gst.ElementFactory.make("nvvideoconvert", "enc-conv")
        self.enc_caps = Gst.ElementFactory.make("capsfilter", "enc-caps")
        self.enc_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=I420"
        ))

        self.encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264-encoder")
        self.encoder.set_property("bitrate", 4000000)
        self.encoder.set_property("iframeinterval", max(1, self._target_fps * 2))

        self.h264parse = Gst.ElementFactory.make("h264parse", "h264parse")

        self.flvmux = Gst.ElementFactory.make("flvmux", "flvmux")
        self.flvmux.set_property("streamable", True)

        self.rtmpsink = Gst.ElementFactory.make("rtmpsink", "rtmpsink")
        self.rtmpsink.set_property("location", self._rtmp_location)
        self.rtmpsink.set_property("sync", False)
        self._rtmp_sink_pad = self.rtmpsink.get_static_pad("sink")
        self._first_frame_probe_id = self._rtmp_sink_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._first_frame_probe,
            None
        )

        # Add elements
        elements = [
            self.streammux,
            self.pgie_queue,
            self.pgie,
            self.tracker,
            self.conv,
            self.conv_caps,
            self.post_tee,
            self.osd_queue,
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
        self.conv.link(self.conv_caps)
        self.conv_caps.link(self.post_tee)
        self._post_tee = self.post_tee
        tee_osd_pad = self.post_tee.get_request_pad("src_%u")
        osd_queue_pad = self.osd_queue.get_static_pad("sink")
        if not tee_osd_pad or not osd_queue_pad or tee_osd_pad.link(osd_queue_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link live tee to OSD branch")
        self._capture_bin = self._build_capture_bin()
        self._pipeline.add(self._capture_bin)
        tee_capture_pad = self.post_tee.get_request_pad("src_%u")
        capture_bin_sink_pad = self._capture_bin.get_static_pad("sink")
        if not tee_capture_pad or not capture_bin_sink_pad or tee_capture_pad.link(capture_bin_sink_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link live tee to capture branch")
        self._capture_branch_pad = tee_capture_pad
        self._capture_bin_sink_pad = capture_bin_sink_pad
        self.osd_queue.link(self.osd)
        self.osd.link(self.enc_conv)
        self.enc_conv.link(self.enc_caps)
        self.enc_caps.link(self.encoder)
        self.encoder.link(self.h264parse)
        self.h264parse.link(self.flvmux)
        self.flvmux.link(self.rtmpsink)

        # Add OSD probe
        osd_sink_pad = self.osd.get_static_pad("sink")
        self._osd_pad = osd_sink_pad
        self._osd_probe_id = osd_sink_pad.add_probe(
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

        self._stop_requested = False
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
        self._streammux_sink_pad = sink_pad
        src_pad = self._source_bin.get_static_pad("src")
        src_pad.link(sink_pad)

        self._play_thread = threading.Thread(target=self._play_background)
        self._play_thread.start()

    def _play_background(self):
        try:
            self._triton_model_manager.request_model_access(
                self._pipeline_id, "rfdetr", self._ai_model_id
            )
            self._triton_model_manager.wait_model_till_ready("rfdetr")

            if self._stop_requested:
                try:
                    self._triton_model_manager.release_model_access(self._pipeline_id)
                except Exception as release_exc:
                    print(f"[WARN]  Failed to release model access after stop request: {release_exc}")
                return

            self._source_bin.sync_state_with_parent()
            ret = self._pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Pipeline failed to transition to PLAYING state.")
            self._is_playing = True

            print(f"[INFO]  Pipeline started → {self._rtmp_location}")
        except Exception as exc:
            self._publish_status(PIPELINE_STATUS_STOPPED, f"Pipeline error: {exc}")
            self._pipeline.set_state(Gst.State.NULL)
            try:
                self._triton_model_manager.release_model_access(self._pipeline_id)
            except Exception as release_exc:
                print(f"[WARN]  Failed to release model access after startup error: {release_exc}")
            return

    # -------------------------------------------------
    # STOP
    # -------------------------------------------------

    def stop(self):
        self._stop_requested = True
        self._publish_status(
            PIPELINE_STATUS_STOPPING,
            "Pipeline is stopping"
        )
        pipeline = self._pipeline
        if pipeline is None:
            self._publish_status(PIPELINE_STATUS_STOPPED, "Pipeline stopped")
            return

        try:
            self._stop_capture_branch()
            self._remove_probe(self._osd_pad, "_osd_probe_id")
            self._remove_probe(self._rtmp_sink_pad, "_first_frame_probe_id")
            self._safe_set_pipeline_null(pipeline)

            try:
                self._triton_model_manager.release_model_access(self._pipeline_id)
            except Exception as release_exc:
                print(f"[WARN]  Failed to release model access while stopping: {release_exc}")

            if self._capture_processing_service is not None:
                try:
                    self._capture_processing_service.stop_worker(self._pipeline_id)
                except Exception:
                    pass

            self._release_streammux_pad()

            if self._source_bin is not None:
                try:
                    pipeline.remove(self._source_bin)
                except Exception:
                    pass
                self._source_bin = None

            self._is_playing = False
            self._clear_runtime_references()

            if self._play_thread and self._play_thread.is_alive() and threading.current_thread() is not self._play_thread:
                self._play_thread.join(timeout=1.0)

        finally:
            self._publish_status(
                PIPELINE_STATUS_STOPPED,
                "Pipeline stopped"
            )
            print("[INFO]  Pipeline stopped.", flush=True)

    def _stop_capture_branch(self):
        if self._capture_appsink is not None:
            self._capture_appsink.set_property("emit-signals", False)

        self._set_element_state_with_timeout(
            element=self._capture_bin,
            target_state=Gst.State.NULL,
            timeout_seconds=5.0,
            thread_name=f"live-capture-bin-null-{self._pipeline_id}",
            label="capture bin",
        )

        self._detach_capture_branch()

    def _safe_set_pipeline_null(self, pipeline):
        timed_out, error = self._set_element_state_with_timeout(
            element=pipeline,
            target_state=Gst.State.NULL,
            timeout_seconds=10.0,
            thread_name=f"live-set-null-{self._pipeline_id}",
            label="pipeline",
        )
        if timed_out or error is not None:
            return

        state_change, current_state, _ = pipeline.get_state(5 * Gst.SECOND)
        if state_change == Gst.StateChangeReturn.FAILURE:
            print(
                f"[WARN]  Pipeline '{self._pipeline_id}' failed to reach NULL state "
                f"(state={current_state.value_nick})"
            )

    def _set_element_state_with_timeout(
        self,
        element,
        target_state: Gst.State,
        timeout_seconds: float,
        thread_name: str,
        label: str,
    ):
        if element is None:
            return False, None

        result = {"returned": False, "error": None}

        def _set_state():
            try:
                element.set_state(target_state)
                result["returned"] = True
            except Exception as exc:
                result["error"] = exc

        worker = threading.Thread(
            target=_set_state,
            name=thread_name,
            daemon=True,
        )
        worker.start()
        worker.join(timeout=timeout_seconds)

        if worker.is_alive():
            print(
                f"[WARN]  {label} set_state({target_state.value_nick}) timed out "
                f"after {timeout_seconds:.1f}s",
                flush=True,
            )
            return True, None

        if result["error"] is not None:
            return False, result["error"]

        return False, None

    def _detach_capture_branch(self):
        if self._capture_branch_pad is None or self._capture_bin_sink_pad is None:
            return

        try:
            if self._capture_branch_pad.is_linked():
                self._capture_branch_pad.unlink(self._capture_bin_sink_pad)
            if self._post_tee is not None:
                self._post_tee.release_request_pad(self._capture_branch_pad)
        except Exception:
            pass
        finally:
            self._capture_branch_pad = None
            self._capture_bin_sink_pad = None

    def _remove_probe(self, pad, probe_attr: str):
        probe_id = getattr(self, probe_attr)
        if not probe_id or pad is None:
            return

        pad.remove_probe(probe_id)
        setattr(self, probe_attr, None)

    def _release_streammux_pad(self):
        if self.streammux is None or self._streammux_sink_pad is None:
            return

        try:
            self.streammux.release_request_pad(self._streammux_sink_pad)
        except Exception:
            pass
        finally:
            self._streammux_sink_pad = None

    def _clear_runtime_references(self):
        self._pipeline = None
        self._capture_appsink = None
        self._capture_bin = None
        self._capture_bin_sink_pad = None
        self._capture_branch_pad = None
        self._post_tee = None
        self._osd_pad = None
        self._osd_probe_id = None
        self._rtmp_sink_pad = None
        self._first_frame_probe_id = None
        self._streammux_sink_pad = None
        self._first_frame_sent = False

    def _build_capture_bin(self):
        capture_bin = Gst.Bin.new("live-capture-bin")

        capture_queue = Gst.ElementFactory.make("queue", "capture-queue")
        capture_conv = Gst.ElementFactory.make("nvvideoconvert", "capture-convert")
        capture_caps = Gst.ElementFactory.make("capsfilter", "capture-caps")
        capture_caps.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))
        capture_appsink = Gst.ElementFactory.make("appsink", "capture-appsink")
        capture_appsink.set_property("emit-signals", True)
        capture_appsink.set_property("sync", False)
        capture_appsink.set_property("max-buffers", 30)
        capture_appsink.set_property("drop", True)
        capture_appsink.connect("new-sample", self._on_new_sample)

        for elem in [capture_queue, capture_conv, capture_caps, capture_appsink]:
            if elem is None:
                raise RuntimeError("Failed to create live capture branch elements")
            capture_bin.add(elem)

        if not capture_queue.link(capture_conv):
            raise RuntimeError("Failed to link live capture queue to converter")
        if not capture_conv.link(capture_caps):
            raise RuntimeError("Failed to link live capture converter to caps")
        if not capture_caps.link(capture_appsink):
            raise RuntimeError("Failed to link live capture caps to appsink")

        sink_pad = capture_queue.get_static_pad("sink")
        ghost_pad = Gst.GhostPad.new("sink", sink_pad)
        capture_bin.add_pad(ghost_pad)

        self._capture_appsink = capture_appsink

        return capture_bin
    
    def get_metadata(self) -> Dict:
        return {
            "pipeline_id": self._pipeline_id,
            "pipeline_name": self._pipeline_name,
            "location_name": self._location_name,
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
            detections = []

            while l_obj:
                try:
                    obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                bbox = [
                    int(obj.rect_params.left),
                    int(obj.rect_params.top),
                    int(obj.rect_params.left + obj.rect_params.width),
                    int(obj.rect_params.top + obj.rect_params.height),
                ]

                detections.append({
                    "bbox": bbox,
                    "class_id": obj.class_id,
                    "confidence": obj.confidence,
                    "track_id": obj.object_id,
                })

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            persons = []
            if self._person_attribute_aggregator is not None:
                persons = self._person_attribute_aggregator.aggregate(detections)
                for person in persons:
                    if self._capture_decision_engine is None or self._capture_worker is None:
                        continue

                    person_id = str(person["person_id"])
                    attrs = [a["class_id"] for a in person["attributes"]]
                    self._capture_decision_engine.register(person_id, attrs)
                    triggered_labels = self._capture_decision_engine.get_triggered_labels(person_id)
                    if triggered_labels:
                        tracked_object = self._build_tracked_object(person_id, person)
                        self._capture_worker.enqueue_capture(buf.pts, tracked_object)

            person_style_by_track_id = self._preview_renderer.build_person_style_map(persons)
            l_obj = frame_meta.obj_meta_list

            while l_obj:
                try:
                    obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                self._preview_renderer.apply_object_preview(
                    batch_meta=batch_meta,
                    frame_meta=frame_meta,
                    obj=obj,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    safe_offset=safe_offset,
                    person_style_by_track_id=person_style_by_track_id,
                )

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            frame = memoryview(map_info.data)
            frame_rgba = bytearray(frame)
            np_frame = np.ndarray((height, width, 4), buffer=frame_rgba, dtype="uint8").copy()
            if self._capture_worker is not None:
                self._capture_worker.store_frame(buffer.pts, np_frame)
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def _build_tracked_object(self, person_id: str, person: dict) -> dict:
        counts = {}
        detections = 0
        if self._capture_decision_engine is not None:
            counts = self._capture_decision_engine.get_attribute_counts(person_id)
            detections = self._capture_decision_engine.get_detection_count(person_id)

        attributes_by_label = {}
        for attr in person["attributes"]:
            label = self._class_id_to_label.get(attr["class_id"], str(attr["class_id"]))
            existing = attributes_by_label.get(label)
            candidate = {
                "label": label,
                "confidence": attr.get("confidence", 0.0),
                "bbox": attr.get("bbox"),
                "count": counts.get(attr["class_id"], 0),
            }
            if existing is None or candidate["confidence"] > existing["confidence"]:
                attributes_by_label[label] = candidate

        return {
            "person_id": person_id,
            "track_id": person.get("person_id"),
            "detections": detections,
            "bbox": person["bbox"],
            "confidence": person.get("confidence", 0.0),
            "attributes": list(attributes_by_label.values()),
        }
    
    def _first_frame_probe(self, pad, info, u_data):
        if not self._first_frame_sent:
            self._first_frame_sent = True

            self._publish_status(
                PIPELINE_STATUS_RUNNING,
                "Pipeline is running (frames reached RTMP sink)"
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
