import gi
from gi.repository import Gst, GObject
from .deepstream_pipeline_interface import DeepstreamPipelineInterface

class DeepstreamPipeline(DeepstreamPipelineInterface):

    def __init__(self, rtsp_url: str, rtmp_url: str):
        Gst.init(None)

        self.rtsp_url = rtsp_url
        self.rtmp_url = rtmp_url

        self.pipeline = None

        self._is_playing = False

        self._build_pipeline()

    # ---------------------------------------------------------
    # Pipeline Builder
    # ---------------------------------------------------------

    def _build_pipeline(self):
        self.pipeline = Gst.Pipeline.new("deepstream-pipeline")

        # Elements
        self.source = Gst.ElementFactory.make("rtspsrc", "rtsp-source")
        self.rtppay = Gst.ElementFactory.make("rtph264depay", "rtppay")
        self.h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        self.decoder = Gst.ElementFactory.make("nvv4l2decoder", "decoder")
        self.streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
        self.nvinferserver = Gst.ElementFactory.make("nvinferserver", "infer")
        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convert")
        self.nvosd = Gst.ElementFactory.make("nvdsosd", "osd")
        self.nvvidconv_post = Gst.ElementFactory.make("nvvideoconvert", "convert-post")
        self.caps = Gst.ElementFactory.make("capsfilter", "caps")
        self.encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        self.h264parser_enc = Gst.ElementFactory.make("h264parse", "parser-enc")
        self.flvmux = Gst.ElementFactory.make("flvmux", "flvmux")
        self.sink = Gst.ElementFactory.make("rtmpsink", "rtmp-sink")

        elements = [
            self.source, self.rtppay, self.h264parser, self.decoder,
            self.streammux, self.nvinferserver, self.nvvidconv,
            self.nvosd, self.nvvidconv_post, self.caps,
            self.encoder, self.h264parser_enc,
            self.flvmux, self.sink
        ]

        for elem in elements:
            if not elem:
                raise RuntimeError("Failed to create GStreamer element")

            self.pipeline.add(elem)

        self._configure_elements()
        self._link_elements()
        self._setup_bus()

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    def _configure_elements(self):
        self.source.set_property("location", self.rtsp_url)
        self.source.set_property("latency", 100)

        self.streammux.set_property("width", 1920)
        self.streammux.set_property("height", 1080)
        self.streammux.set_property("batch-size", 1)
        self.streammux.set_property("batched-push-timeout", 4000000)

        self.nvinferserver.set_property(
            "config-file-path",
            "/app/config/deepstream-inferserver-yolo.txt"
        )

        self.nvosd.set_property("process-mode", 1)
        self.nvosd.set_property("display-text", 1)

        caps_str = "video/x-raw(memory:NVMM), format=I420"
        self.caps.set_property("caps", Gst.Caps.from_string(caps_str))

        self.encoder.set_property("bitrate", 4000000)
        self.encoder.set_property("iframeinterval", 30)

        self.h264parser_enc.set_property("config-interval", -1)

        self.flvmux.set_property("streamable", True)

        self.sink.set_property("location", self.rtmp_url)
        self.sink.set_property("sync", 0)

    # ---------------------------------------------------------
    # Linking
    # ---------------------------------------------------------

    def _link_elements(self):
        self.rtppay.link(self.h264parser)
        self.h264parser.link(self.decoder)

        sinkpad = self.streammux.get_request_pad("sink_0")
        srcpad = self.decoder.get_static_pad("src")
        srcpad.link(sinkpad)

        self.streammux.link(self.nvinferserver)
        self.nvinferserver.link(self.nvvidconv)
        self.nvvidconv.link(self.nvosd)
        self.nvosd.link(self.nvvidconv_post)
        self.nvvidconv_post.link(self.caps)
        self.caps.link(self.encoder)
        self.encoder.link(self.h264parser_enc)
        self.h264parser_enc.link(self.flvmux)
        self.flvmux.link(self.sink)

        # dynamic pad
        self.source.connect("pad-added", self._on_pad_added)

    def _on_pad_added(self, src, new_pad):
        caps = new_pad.get_current_caps()
        structure = caps.get_structure(0)
        name = structure.get_name()

        if name.startswith("application/x-rtp"):
            sink_pad = self.rtppay.get_static_pad("sink")
            if not sink_pad.is_linked():
                new_pad.link(sink_pad)

    # ---------------------------------------------------------
    # Bus
    # ---------------------------------------------------------

    def _setup_bus(self):
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call)

    def _bus_call(self, bus, message):
        t = message.type

        if t == Gst.MessageType.EOS:
            print("End of stream")
            self.stop()

        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"ERROR: {err}, {debug}")
            self.stop()

        return True

    # ---------------------------------------------------------
    # Interface Implementation
    # ---------------------------------------------------------

    def play(self):
        if self._is_playing:
            return

        print("Starting DeepStream pipeline...")
        self.pipeline.set_state(Gst.State.PLAYING)

        self._is_playing = True

    def stop(self):
        if not self._is_playing:
            return

        print("Stopping DeepStream pipeline...")

        self.pipeline.set_state(Gst.State.NULL)

        self._is_playing = False
