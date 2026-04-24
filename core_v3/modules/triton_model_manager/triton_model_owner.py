import threading
from typing import Dict, Optional
from gi.repository import Gst


# ── Config Registry ───────────────────────────────────────────────────────────

MODEL_CONFIG_MAP: Dict[str, str] = {
    "yolo":   "/app/config/deepstream-inferserver-yolo.txt",
    "rfdetr": "/app/config/deepstream-inferserver-rfdetr-test.txt",
}


# ── Owner ─────────────────────────────────────────────────────────────────────

class TritonModelOwner:
    """
    Owns a single Triton model lifecycle via a minimal DeepStream pipeline.
    Model is considered ready when the first buffer reaches fakesink.
    Relies on the main program's GLib main loop.
    """

    def __init__(self, model_id: str, infer_config: str, gpu_id: int = 0):
        self._model_id    = model_id
        self._infer_config = infer_config
        self._gpu_id      = gpu_id
        self._pipeline: Optional[Gst.Pipeline] = None
        self._model_ready = threading.Event()

    def load(self):
        self._pipeline = self._build()

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call)

        self._pipeline.set_state(Gst.State.PLAYING)
        print(f"[TritonModelOwner] Model '{self._model_id}' pipeline started")

    def unload(self):
        print(f"[TritonModelOwner] Unloading model '{self._model_id}'")
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        self._model_ready.clear()
        print(f"[TritonModelOwner] Model '{self._model_id}' unloaded")

    def wait_till_ready(self, timeout: Optional[float] = None):
        print(f"[TritonModelOwner] Waiting for model '{self._model_id}' to be ready...")
        ready = self._model_ready.wait(timeout=timeout)
        if not ready:
            raise TimeoutError(
                f"[TritonModelOwner] Model '{self._model_id}' not ready after {timeout}s"
            )
        print(f"[TritonModelOwner] Model '{self._model_id}' is ready")

    def is_ready(self) -> bool:
        return self._model_ready.is_set()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _on_fakesink_handoff(self, sink, buffer, pad):
        if not self._model_ready.is_set():
            print(f"[TritonModelOwner] First buffer received for '{self._model_id}' — Triton ready")
            self._model_ready.set()

            # Throttle — model is loaded, no need to keep inferring
            infer = self._pipeline.get_by_name(f"owner-infer-{self._model_id}")
            if infer:
                infer.set_property("interval", 2147483647)

    def _build(self) -> Gst.Pipeline:
        pipeline = Gst.Pipeline.new(f"owner-pipeline-{self._model_id}")

        src = self._make("videotestsrc", f"owner-src-{self._model_id}")
        src.set_property("is-live", True)
        src.set_property("pattern", 2)
        # Force a known caps so nvstreammux accepts it
        src.set_property("num-buffers", -1)

        capsfilter = self._make("capsfilter", f"owner-caps-{self._model_id}")
        capsfilter.set_property("caps", Gst.Caps.from_string(
            "video/x-raw, format=NV12, width=64, height=64, framerate=5/1"
        ))

        nvconv = self._make("nvvideoconvert", f"owner-conv-{self._model_id}")
        nvconv.set_property("gpu-id", self._gpu_id)

        nvconv_caps = self._make("capsfilter", f"owner-conv-caps-{self._model_id}")
        nvconv_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=NV12, width=64, height=64, framerate=5/1"
        ))

        streammux = self._make("nvstreammux", f"owner-mux-{self._model_id}")
        streammux.set_property("batch-size", 1)
        streammux.set_property("width", 64)
        streammux.set_property("height", 64)
        streammux.set_property("live-source", True)
        streammux.set_property("gpu-id", self._gpu_id)
        streammux.set_property("batched-push-timeout", 400)

        infer = self._make("nvinferserver", f"owner-infer-{self._model_id}")
        infer.set_property("config-file-path", self._infer_config)
        infer.set_property("interval", 0)

        sink = self._make("fakesink", f"owner-sink-{self._model_id}")
        sink.set_property("sync", False)
        sink.set_property("signal-handoffs", True)
        sink.connect("handoff", self._on_fakesink_handoff)

        for el in [src, capsfilter, nvconv, nvconv_caps, streammux, infer, sink]:
            pipeline.add(el)

        # src → capsfilter → nvconv → nvconv_caps → streammux sink_0
        src.link(capsfilter)
        capsfilter.link(nvconv)
        nvconv.link(nvconv_caps)

        sinkpad = streammux.request_pad_simple("sink_0")
        srcpad  = nvconv_caps.get_static_pad("src")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link nvconv_caps → streammux")

        streammux.link(infer)
        infer.link(sink)

        return pipeline
    
    def _bus_call(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"[TritonModelOwner][ERROR] {self._model_id}: {err} | {dbg}")
            # Signal ready with error so wait_till_ready doesn't hang forever
            # Caller will discover the pipeline is broken when they try to use it
        elif t == Gst.MessageType.WARNING:
            err, dbg = message.parse_warning()
            print(f"[TritonModelOwner][WARN] {self._model_id}: {err} | {dbg}")
        return True

    @staticmethod
    def _make(factory: str, name: str) -> Gst.Element:
        el = Gst.ElementFactory.make(factory, name)
        if not el:
            raise RuntimeError(f"Could not create element '{factory}' as '{name}'")
        return el
