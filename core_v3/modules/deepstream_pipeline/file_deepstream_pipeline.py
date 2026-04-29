import threading
import queue
import time
from typing import Dict
import numpy as np
import pyds

from gi.repository import Gst

from .constant import (
    PIPELINE_STATUS_RUNNING,
    PIPELINE_STATUS_STARTING,
    PIPELINE_STATUS_STOPPED,
    PIPELINE_STATUS_STOPPING,
)

from .deepstream_pipeline_interface import DeepstreamPipelineInterface
from .capture_decision_engine import CaptureDecisionEngine
from .person_attribute_aggregator import PersonAttributeAggregator
from .ppe_preview_renderer import PPEPreviewRenderer
from ..drawing.FrameDrawer import FrameDrawer
from ..capture_processing_service.capture_processing_service import CaptureProcessingService
from ..triton_model_manager.triton_model_manager import TritonModelManager


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
        ai_model_id: str | None,
        worker_id: str,
        worker_source_id: str,
        location_name: str | None,
        capture_decision_engine: CaptureDecisionEngine,
        person_attribute_aggregator: PersonAttributeAggregator,
        frame_drawer: FrameDrawer,
        capture_processing_service: CaptureProcessingService,
        class_id_to_label: Dict[int, str],
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
        self.worker_id      = worker_id
        self.ai_model_id    = ai_model_id
        self.worker_source_id = worker_source_id
        self.location_name  = location_name or "LOCATION"
        self.output_width   = output_width
        self.output_height  = output_height
        self.target_fps     = target_fps
        self.encode_bitrate = encode_bitrate
        self.gpu_id         = gpu_id
        self.triton_model_manager = triton_model_manager
        self.capture_decision_engine = capture_decision_engine
        self.person_attribute_aggregator = person_attribute_aggregator
        self.frame_drawer = frame_drawer
        self.frame_drawer.location_name = self.location_name
        self.capture_processing_service = capture_processing_service
        self.class_id_to_label = class_id_to_label

        self._update_event_queue = update_event_queue
        self._status             = PIPELINE_STATUS_STOPPED
        self._first_frame_sent   = False
        self._stop_requested     = False

        self._osd_pad              = None
        self._post_tiler_tee       = None
        self._capture_branch_pad   = None
        self._capture_bin          = None
        self._capture_bin_sink_pad = None
        self._rtmp_sink_pad        = None
        self._first_frame_probe_id = None
        self._osd_probe_id         = None
        self._capture_appsink      = None
        self._uri_src              = None

        self._play_thread = None
        self._capture_worker = self.capture_processing_service.get_or_create_worker(
            pipeline_id=self.pipeline_id,
            worker_id=self.worker_id,
            worker_source_id=self.worker_source_id,
            frame_drawer=self.frame_drawer,
        )
        self._preview_renderer = PPEPreviewRenderer(
            frame_drawer=self.frame_drawer,
            class_id_to_label=self.class_id_to_label,
            location_name=self.location_name,
            person_class_id=4,
            attribute_class_ids={1, 2, 3, 5},
            preview_style_hold_seconds=3.0,
        )

        self._pipeline = self._build_pipeline()

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call)

    # ── Interface ────────────────────────────────────────────────────────────

    def play(self):
        self._stop_requested = False
        self._publish_status(PIPELINE_STATUS_STARTING, "Pipeline is starting")
        self._play_thread = threading.Thread(target=self._play_background)
        self._play_thread.start()

    def _play_background(self):
        try:
            self.triton_model_manager.request_model_access(
                self.pipeline_id, "rfdetr", self.ai_model_id
            )
            self.triton_model_manager.wait_model_till_ready("rfdetr")

            if self._stop_requested:
                try:
                    self.triton_model_manager.release_model_access(self.pipeline_id)
                except Exception as exc:
                    print(f"[WARN]  Failed to release model access after stop request: {exc}")
                try:
                    self._pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass
                self._publish_status(PIPELINE_STATUS_STOPPED, "Pipeline stopped before starting")
                return

            ret = self._pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Pipeline failed to transition to PLAYING state.")

            print(f"[INFO]  Pipeline started → {self.rtmp_uri}")

        except Exception as exc:
            self._publish_status(PIPELINE_STATUS_STOPPED, f"Pipeline error: {exc}")
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            try:
                self.triton_model_manager.release_model_access(self.pipeline_id)
            except Exception as release_exc:
                print(f"[WARN]  Failed to release model access after startup error: {release_exc}")

    def stop(self):
        self._stop_requested = True
        self._publish_status(PIPELINE_STATUS_STOPPING, "Pipeline is stopping")

        pipeline = self._pipeline
        if pipeline is None:
            self._publish_status(PIPELINE_STATUS_STOPPED, "Pipeline stopped")
            return

        try:
            self._stop_capture_branch()
            self._remove_probe(self._osd_pad, "_osd_probe_id")
            self._remove_probe(self._rtmp_sink_pad, "_first_frame_probe_id")

            try:
                self.triton_model_manager.release_model_access(self.pipeline_id)
            except Exception:
                pass

            try:
                self.capture_processing_service.stop_worker(self.pipeline_id)
            except Exception:
                pass

            self._safe_set_pipeline_null(pipeline)

            self._clear_runtime_references()

            if (
                self._play_thread
                and self._play_thread.is_alive()
                and threading.current_thread() is not self._play_thread
            ):
                self._play_thread.join(timeout=2.0)

        finally:
            self._publish_status(PIPELINE_STATUS_STOPPED, "Pipeline stopped")
            print("[INFO]  Pipeline stopped.", flush=True)

    def _stop_capture_branch(self):
        if self._capture_appsink is not None:
            self._capture_appsink.set_property("emit-signals", False)

        self._set_element_state_with_timeout(
            element=self._capture_bin,
            target_state=Gst.State.NULL,
            timeout_seconds=5.0,
            thread_name=f"capture-bin-null-{self.pipeline_id}",
            label="capture bin",
        )

        self._detach_capture_branch()

    def _safe_set_pipeline_null(self, pipeline):
        timed_out, error = self._set_element_state_with_timeout(
            element=pipeline,
            target_state=Gst.State.NULL,
            timeout_seconds=10.0,
            thread_name=f"set-null-{self.pipeline_id}",
            label="pipeline",
        )
        if timed_out or error is not None:
            return

        pipeline.get_state(5 * Gst.SECOND)

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
            if self._post_tiler_tee is not None:
                self._post_tiler_tee.release_request_pad(self._capture_branch_pad)
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

    def _clear_runtime_references(self):
        self._pipeline = None
        self._osd_pad = None
        self._post_tiler_tee = None
        self._capture_branch_pad = None
        self._capture_bin = None
        self._capture_bin_sink_pad = None
        self._rtmp_sink_pad = None
        self._first_frame_probe_id = None
        self._osd_probe_id = None
        self._capture_appsink = None
        self._uri_src = None
        self._first_frame_sent = False

    def get_metadata(self) -> Dict:
        return {
            "pipeline_id":      self.pipeline_id,
            "pipeline_name":    self.pipeline_name,
            "location_name":    self.location_name,
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
        streammux.set_property("batch-size",           1)
        streammux.set_property("width",                self.output_width)
        streammux.set_property("height",               self.output_height)
        streammux.set_property("gpu-id",               self.gpu_id)
        streammux.set_property("live-source",          False)
        streammux.set_property("batched-push-timeout", 400)
        streammux.set_property("nvbuf-memory-type",    0)
        streammux.set_property("attach-sys-ts",        False)
        pipeline.add(streammux)

        # 2. Source bin using nvurisrcbin
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
                raise RuntimeError("Failed to link source bin to streammux")
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

        # 5. nvtracker
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

        # 7. Tiler
        tiler = self._make("nvmultistreamtiler", "tiler")
        tiler.set_property("rows",    1)
        tiler.set_property("columns", 1)
        tiler.set_property("width",   self.output_width)
        tiler.set_property("height",  self.output_height)
        tiler.set_property("gpu-id",  self.gpu_id)

        # 8. Post-tiler → RGBA for OSD + tee
        post_conv = self._make("nvvideoconvert", "post-tiler-conv")
        post_conv.set_property("gpu-id", self.gpu_id)
        post_caps = self._make("capsfilter", "post-tiler-caps")
        post_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=RGBA"))
        post_tee = self._make("tee", "post-tiler-tee")
        self._post_tiler_tee = post_tee

        osd_queue = self._make("queue", "osd-queue")

        appsink = self._make("appsink", "capture-appsink")
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync",         False)
        appsink.set_property("max-buffers",  30)
        appsink.set_property("drop",         True)
        appsink.connect("new-sample", self._on_new_sample)
        self._capture_appsink = appsink
        capture_bin = self._create_capture_bin(appsink)
        self._capture_bin = capture_bin

        # 9. OSD
        nvosd = self._make("nvdsosd", "osd")
        nvosd.set_property("process-mode", 0)
        nvosd.set_property("display-text", 1)

        # 10. Encode
        enc_conv = self._make("nvvideoconvert", "enc-conv")
        enc_conv.set_property("gpu-id", self.gpu_id)
        enc_caps = self._make("capsfilter", "enc-caps")
        enc_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=I420"))

        # 11. H.264 encoder
        encoder = self._make("nvv4l2h264enc", "h264-encoder")
        encoder.set_property("bitrate",        self.encode_bitrate * 1000)
        encoder.set_property("iframeinterval", max(1, self.target_fps * 2))

        # 12. H.264 parser
        h264parse = self._make("h264parse", "h264-parse")
        h264parse.set_property("config-interval", -1)

        # 13. FLV muxer
        flvmux = self._make("flvmux", "flv-mux")
        flvmux.set_property("streamable", True)

        # 14. RTMP sink
        rtmpsink = self._make("rtmpsink", "rtmp-sink")
        rtmpsink.set_property("location", self.rtmp_uri)
        rtmpsink.set_property("sync",  False)
        rtmpsink.set_property("async", False)
        self._rtmp_sink_pad = rtmpsink.get_static_pad("sink")
        self._first_frame_probe_id = self._rtmp_sink_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._first_frame_probe,
            None,
        )

        # Add all elements
        for el in [mux_conv, mux_caps, pgie, tracker,
                   pre_conv, pre_caps, tiler,
                   post_conv, post_caps, post_tee,
                   osd_queue, capture_bin,
                   nvosd, enc_conv, enc_caps, encoder, h264parse, flvmux, rtmpsink]:
            pipeline.add(el)

        # Link main chain
        for src_el, dst_el in [
            (streammux,   mux_conv),
            (mux_conv,    mux_caps),
            (mux_caps,    pgie),
            (pgie,        tracker),
            (tracker,     pre_conv),
            (pre_conv,    pre_caps),
            (pre_caps,    tiler),
            (tiler,       post_conv),
            (post_conv,   post_caps),
            (osd_queue,   nvosd),
            (nvosd,       enc_conv),
            (enc_conv,    enc_caps),
            (enc_caps,    encoder),
            (encoder,     h264parse),
            (h264parse,   flvmux),
            (flvmux,      rtmpsink),
        ]:
            if not src_el.link(dst_el):
                raise RuntimeError(
                    f"Failed to link {src_el.get_name()} → {dst_el.get_name()}")

        # Link tee branches
        if not post_caps.link(post_tee):
            raise RuntimeError("Failed to link post-tiler-caps to tee")

        osd_pad = post_tee.get_request_pad("src_%u")
        osd_queue_pad = osd_queue.get_static_pad("sink")
        if not osd_pad or not osd_queue_pad or osd_pad.link(osd_queue_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link tee to OSD branch")

        capture_pad = post_tee.get_request_pad("src_%u")
        capture_bin_sink_pad = capture_bin.get_static_pad("sink")
        if not capture_pad or not capture_bin_sink_pad or capture_pad.link(capture_bin_sink_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link tee to capture branch")
        self._capture_branch_pad = capture_pad
        self._capture_bin_sink_pad = capture_bin_sink_pad

        # OSD probe
        self._osd_pad = nvosd.get_static_pad("sink")
        if self._osd_pad:
            self._osd_probe_id = self._osd_pad.add_probe(
                Gst.PadProbeType.BUFFER, self._osd_probe, None)

        return pipeline

    def _create_source_bin(self, index: int, uri: str) -> Gst.Bin:
        bin_name = f"source-bin-{index}"
        nbin = Gst.Bin.new(bin_name)

        uri_src = self._make("nvurisrcbin", f"uri-src-{index}")
        uri_src.set_property("uri",                 uri)
        uri_src.set_property("gpu-id",              self.gpu_id)
        uri_src.set_property("drop-frame-interval", 2)
        uri_src.set_property("file-loop",           True)
        self._uri_src = uri_src
        nbin.add(uri_src)

        def _on_pad_added(src, pad):
            caps = pad.get_current_caps() or pad.query_caps()
            if not caps:
                return
            struct = caps.get_structure(0)
            if not struct or not struct.get_name().startswith("video"):
                return
            if nbin.get_static_pad("src"):
                return

            ghost = Gst.GhostPad.new("src", pad)
            if not ghost:
                print(f"[ERROR] Failed to create ghost pad on {bin_name}", flush=True)
                return
            ghost.set_active(True)
            nbin.add_pad(ghost)
            print(f"[INFO]  Ghost pad created on {bin_name}", flush=True)

        def _child_added(child_proxy, obj, name, _):
            if name.startswith("decodebin"):
                obj.connect("child-added", _child_added, None)

        uri_src.connect("pad-added", _on_pad_added)
        uri_src.connect("child-added", _child_added, None)
        return nbin

    def _create_capture_bin(self, appsink: Gst.Element) -> Gst.Bin:
        capture_bin = Gst.Bin.new("capture-bin")
        if not capture_bin:
            raise RuntimeError("Could not create capture bin")

        capture_queue = self._make("queue", "capture-queue")
        capture_conv = self._make("nvvideoconvert", "capture-convert")
        capture_conv.set_property("gpu-id", self.gpu_id)
        capture_caps = self._make("capsfilter", "capture-caps")
        capture_caps.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))

        for el in [capture_queue, capture_conv, capture_caps, appsink]:
            capture_bin.add(el)

        for src_el, dst_el in [
            (capture_queue, capture_conv),
            (capture_conv, capture_caps),
            (capture_caps, appsink),
        ]:
            if not src_el.link(dst_el):
                raise RuntimeError(
                    f"Failed to link {src_el.get_name()} → {dst_el.get_name()}"
                )

        ghost = Gst.GhostPad.new("sink", capture_queue.get_static_pad("sink"))
        if not ghost:
            raise RuntimeError("Could not create capture-bin ghost sink pad")
        ghost.set_active(True)
        capture_bin.add_pad(ghost)
        return capture_bin

    # ── Probes ───────────────────────────────────────────────────────────────

    def _first_frame_probe(self, pad, info, user_data):
        if not self._first_frame_sent:
            self._first_frame_sent = True
            self._publish_status(
                PIPELINE_STATUS_RUNNING,
                "Pipeline is running (frames reached RTMP sink)",
            )
        return Gst.PadProbeReturn.OK

    def _osd_probe(self, pad, info, user_data):
        if self._stop_requested:
            return Gst.PadProbeReturn.DROP

        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch_meta.frame_meta_list

        frame_w = self.output_width
        frame_h = self.output_height

        def safe_offset(value: int, max_value: int):
            return max(0, min(value, max_value))

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
                    "bbox":       bbox,
                    "class_id":   obj.class_id,
                    "confidence": obj.confidence,
                    "track_id":   obj.object_id,
                })

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            persons = self.person_attribute_aggregator.aggregate(detections)
            for person in persons:
                person_id = str(person["person_id"])
                attrs = [a["class_id"] for a in person["attributes"]]

                self.capture_decision_engine.register(person_id, attrs)
                triggered_labels = self.capture_decision_engine.get_triggered_labels(person_id)
                if triggered_labels:
                    tracked_object = self._build_tracked_object(person_id, person)
                    self._capture_worker.enqueue_capture(buf.pts, tracked_object)

            person_style_by_track_id = self._preview_renderer.build_person_style_map(persons)

            # Drawing logic

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
        if self._stop_requested:
            return Gst.FlowReturn.EOS

        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width  = structure.get_value("width")
        height = structure.get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            frame_rgba = bytearray(memoryview(map_info.data))
            np_frame = np.ndarray(
                (height, width, 4), buffer=frame_rgba, dtype="uint8"
            ).copy()
            self._capture_worker.store_frame(buffer.pts, np_frame)
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def _build_tracked_object(self, person_id: str, person: dict) -> dict:
        counts = self.capture_decision_engine.get_attribute_counts(person_id)
        attributes_by_label = {}

        for attr in person["attributes"]:
            label = self.class_id_to_label.get(attr["class_id"], str(attr["class_id"]))
            existing = attributes_by_label.get(label)
            candidate = {
                "label":      label,
                "confidence": attr.get("confidence", 0.0),
                "bbox":       attr.get("bbox"),
                "count":      counts.get(attr["class_id"], 0),
            }
            if existing is None or candidate["confidence"] > existing["confidence"]:
                attributes_by_label[label] = candidate

        return {
            "person_id":  person_id,
            "track_id":   person.get("person_id"),
            "detections": self.capture_decision_engine.get_detection_count(person_id),
            "bbox":       person["bbox"],
            "confidence": person.get("confidence", 0.0),
            "attributes": list(attributes_by_label.values()),
        }

    def _bus_call(self, bus, message):
        t = message.type

        if t == Gst.MessageType.EOS:
            if self._stop_requested:
                print("[INFO]  EOS received during stop — ignoring")
                return True

            # Seek back to start for seamless loop
            print("[INFO]  EOS received — seeking to start for loop")
            success = self._pipeline.seek(
                1.0,
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                Gst.SeekType.SET, 0,
                Gst.SeekType.NONE, 0,
            )
            if not success:
                print("[WARN]  Seek to start failed")

        elif t == Gst.MessageType.WARNING:
            err, dbg = message.parse_warning()
            print(f"[WARN]  {err} | {dbg}")

        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            if self._stop_requested:
                print(f"[INFO]  Ignoring bus error during stop: {err} | {dbg}")
                return True
            print(f"[ERROR] {err} | {dbg}")
            threading.Thread(
                target=self.stop,
                name=f"error-stop-{self.pipeline_id}",
                daemon=True,
            ).start()

        return True

    # ── Status ───────────────────────────────────────────────────────────────

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

    def crop_with_padding(self, frame, bbox, padding_ratio=0.25):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_w, pad_h = int(bw * padding_ratio), int(bh * padding_ratio)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        return frame[y1:y2, x1:x2]

    def safe_bbox(self, bbox, frame_w, frame_h):
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w - 1))
        y2 = max(0, min(y2, frame_h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]
