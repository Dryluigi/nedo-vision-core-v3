#!/usr/bin/env python3
"""
DeepStream Pipeline: nvurisrcbin → nvstreammux → nvinferserver → ... → RTMP

Pipeline:
  nvurisrcbin → nvstreammux
    → nvvideoconvert → caps(NV12)
    → nvinferserver
    → nvvideoconvert → caps(NV12) → nvmultistreamtiler
    → nvvideoconvert → caps(RGBA) → nvdsosd
    → nvvideoconvert → caps(I420)
    → x264enc → h264parse → flvmux → rtmpsink
"""

import sys
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import pyds

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_URIS = [
    "file:///app/videos/pos-pabrik-h265.mp4",
    # "file:///app/videos/12683529_3840_2160_30fps.mp4",
]

INFER_CONFIG  = "/app/config/deepstream-inferserver-yolo.txt"
OUTPUT_WIDTH  = 1280
OUTPUT_HEIGHT = 720
BATCH_SIZE    = len(INPUT_URIS)
GPU_ID        = 0
TARGET_FPS    = 30

# ── RTMP settings ──────────────────────────────────────────────────────────
RTMP_URI      = "rtmp://rtmp.petrokimia-gresik.com:1935/live/cctv-5?user=petrokimia&pass=OG6sM2Lsm6RW"   # ← change to your RTMP server
ENCODE_BITRATE = 4000   # kbps
ENCODE_PRESET  = 2      # 0=ultrafast … 8=veryslow (lower = faster, lower quality)
# ---------------------------------------------------------------------------


def make_element(factory, name):
    el = Gst.ElementFactory.make(factory, name)
    if not el:
        raise RuntimeError(f"Could not create '{factory}' element named '{name}'")
    return el


# ── Source bin ──────────────────────────────────────────────────────────────

def _src_bin_child_added(child_proxy, obj, name, user_data):
    if name.startswith("decodebin"):
        obj.connect("child-added", _src_bin_child_added, user_data)


def create_source_bin(index: int, uri: str) -> Gst.Bin:
    bin_name = f"source-bin-{index}"
    nbin = Gst.Bin.new(bin_name)

    uri_src = make_element("nvurisrcbin", f"uri-src-{index}")
    uri_src.set_property("uri", uri)
    uri_src.set_property("gpu-id", GPU_ID)
    uri_src.set_property("drop-frame-interval", 0)
    uri_src.set_property("file-loop", True)
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

    uri_src.connect("pad-added", _on_pad_added)
    uri_src.connect("child-added", _src_bin_child_added, uri_src)
    return nbin


# ── Metadata probe ───────────────────────────────────────────────────────────

def osd_sink_pad_buffer_probe(pad, info, _):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    l_frame = batch_meta.frame_meta_list
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
            r = obj.rect_params
            # print(
            #     f"  Frame={frame_meta.frame_num} src={frame_meta.source_id} "
            #     f"cls={obj.class_id} lbl={obj.obj_label!r} "
            #     f"conf={obj.confidence:.2f} "
            #     f"box=[{r.left:.0f},{r.top:.0f},{r.width:.0f},{r.height:.0f}]"
            # )
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


# ── Bus ──────────────────────────────────────────────────────────────────────

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[INFO] EOS received.")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, dbg = message.parse_warning()
        print(f"[WARN]  {err} | {dbg}")
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        print(f"[ERROR] {err} | {dbg}")
        loop.quit()
    return True


# ── Pipeline ─────────────────────────────────────────────────────────────────

def build_pipeline():
    pipeline = Gst.Pipeline()

    # ── 1. Streammux ────────────────────────────────────────────────────────
    streammux = make_element("nvstreammux", "stream-muxer")
    pipeline.add(streammux)
    streammux.set_property("batch-size", BATCH_SIZE)
    streammux.set_property("width", OUTPUT_WIDTH)
    streammux.set_property("height", OUTPUT_HEIGHT)
    streammux.set_property("gpu-id", GPU_ID)
    streammux.set_property("live-source", False)
    streammux.set_property("batched-push-timeout", 40000)
    streammux.set_property("nvbuf-memory-type", 0)
    streammux.set_property("attach-sys-ts", True)

    # ── 2. Sources ──────────────────────────────────────────────────────────
    for i, uri in enumerate(INPUT_URIS):
        print(f"[INFO] Adding source {i}: {uri}")
        src_bin = create_source_bin(i, uri)
        pipeline.add(src_bin)

        pad_name = f"sink_{i}"
        sinkpad = streammux.get_request_pad(pad_name)
        if not sinkpad:
            sinkpad = streammux.request_pad_simple(pad_name)
        if not sinkpad:
            raise RuntimeError(f"Could not get streammux pad '{pad_name}'")

        srcpad = src_bin.get_static_pad("src")
        if srcpad:
            if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link source {i} to streammux")
        else:
            def _link_to_mux(element, pad, sp=sinkpad, idx=i):
                if pad.get_name() != "src":
                    return
                if sp.is_linked():
                    return
                ret = pad.link(sp)
                if ret == Gst.PadLinkReturn.OK:
                    print(f"[INFO]  source-bin-{idx} → streammux sink_{idx} linked")
                else:
                    print(f"[ERROR] source-bin-{idx} → streammux link failed: {ret}")
            src_bin.connect("pad-added", _link_to_mux)

    # ── 3. NV12 caps after muxer (before nvinferserver) ─────────────────────
    mux_conv = make_element("nvvideoconvert", "mux-conv")
    mux_conv.set_property("gpu-id", GPU_ID)
    mux_caps = make_element("capsfilter", "mux-caps")
    mux_caps.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM), format=NV12"))

    # ── 4. nvinferserver ────────────────────────────────────────────────────
    pgie = make_element("nvinferserver", "primary-inference")
    pgie.set_property("config-file-path", INFER_CONFIG)

    # ── 5. Pre-tiler converter ───────────────────────────────────────────────
    pre_tiler_conv = make_element("nvvideoconvert", "pre-tiler-conv")
    pre_tiler_conv.set_property("gpu-id", GPU_ID)
    pre_tiler_caps = make_element("capsfilter", "pre-tiler-caps")
    pre_tiler_caps.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM), format=NV12"))

    # ── 6. Tiler ────────────────────────────────────────────────────────────
    tiler = make_element("nvmultistreamtiler", "tiler")
    cols = max(1, int(len(INPUT_URIS) ** 0.5))
    rows = (len(INPUT_URIS) + cols - 1) // cols
    tiler.set_property("rows", rows)
    tiler.set_property("columns", cols)
    tiler.set_property("width", OUTPUT_WIDTH)
    tiler.set_property("height", OUTPUT_HEIGHT)
    tiler.set_property("gpu-id", GPU_ID)

    # ── 7. Post-tiler converter → RGBA for OSD ──────────────────────────────
    post_tiler_conv = make_element("nvvideoconvert", "post-tiler-conv")
    post_tiler_conv.set_property("gpu-id", GPU_ID)
    post_tiler_caps = make_element("capsfilter", "post-tiler-caps")
    post_tiler_caps.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM), format=RGBA"))

    # ── 8. OSD ──────────────────────────────────────────────────────────────
    nvosd = make_element("nvdsosd", "osd")
    nvosd.set_property("process-mode", 0)
    nvosd.set_property("display-text", 1)

    # ── 9. Pre-encode converter → I420 (required by x264enc) ────────────────
    enc_conv = make_element("nvvideoconvert", "enc-conv")
    enc_conv.set_property("gpu-id", GPU_ID)
    enc_caps = make_element("capsfilter", "enc-caps")
    enc_caps.set_property("caps", Gst.Caps.from_string(
        "video/x-raw, format=I420"))          # x264enc needs system memory I420

    # ── 10. H.264 encoder ───────────────────────────────────────────────────
    # Option A: CPU encoder (always available)
    # encoder = make_element("x264enc", "h264-encoder")
    # encoder.set_property("bitrate", ENCODE_BITRATE)
    # encoder.set_property("speed-preset", ENCODE_PRESET)
    # encoder.set_property("tune", 0x00000004)   # zerolatency — important for RTMP
    # encoder.set_property("key-int-max", 60)    # keyframe every 2 s at 30 fps

    # Option B: GPU encoder (uncomment if you have nvv4l2h264enc / nvenc)
    encoder = make_element("nvv4l2h264enc", "h264-encoder")
    encoder.set_property("bitrate", ENCODE_BITRATE * 1000)
    # encoder.set_property("preset-level", 1)    # 1=UltraFastPreset
    # encoder.set_property("insert-sps-pps", 1)
    encoder.set_property("iframeinterval", 60)
    enc_caps.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM), format=I420"))  # nvv4l2h264enc accepts NVMM

    # ── 11. H.264 parser ────────────────────────────────────────────────────
    h264parse = make_element("h264parse", "h264-parse")
    h264parse.set_property("config-interval", -1)   # send SPS/PPS with every IDR

    # ── 12. FLV muxer ───────────────────────────────────────────────────────
    flvmux = make_element("flvmux", "flv-mux")
    flvmux.set_property("streamable", True)   # required for live RTMP

    # ── 13. RTMP sink ───────────────────────────────────────────────────────
    rtmpsink = make_element("rtmpsink", "rtmp-sink")
    rtmpsink.set_property("location", RTMP_URI)
    rtmpsink.set_property("sync", True)
    rtmpsink.set_property("async", False)

    # ── Add all elements ─────────────────────────────────────────────────────
    for el in [mux_conv, mux_caps, pgie,
               pre_tiler_conv, pre_tiler_caps, tiler,
               post_tiler_conv, post_tiler_caps, nvosd,
               enc_conv, enc_caps, encoder, h264parse, flvmux, rtmpsink]:
        pipeline.add(el)

    # ── Link ─────────────────────────────────────────────────────────────────
    # streammux → conv → NV12 → nvinferserver
    #           → conv → NV12 → tiler
    #           → conv → RGBA → osd
    #           → conv → I420 → x264enc → h264parse → flvmux → rtmpsink
    assert streammux.link(mux_conv),                    "streammux → mux_conv"
    assert mux_conv.link(mux_caps),                     "mux_conv → mux_caps"
    assert mux_caps.link(pgie),                         "mux_caps → pgie"
    assert pgie.link(pre_tiler_conv),                   "pgie → pre_tiler_conv"
    assert pre_tiler_conv.link(pre_tiler_caps),         "pre_tiler_conv → pre_tiler_caps"
    assert pre_tiler_caps.link(tiler),                  "pre_tiler_caps → tiler"
    assert tiler.link(post_tiler_conv),                 "tiler → post_tiler_conv"
    assert post_tiler_conv.link(post_tiler_caps),       "post_tiler_conv → post_tiler_caps"
    assert post_tiler_caps.link(nvosd),                 "post_tiler_caps → nvosd"
    assert nvosd.link(enc_conv),                        "nvosd → enc_conv"
    assert enc_conv.link(enc_caps),                     "enc_conv → enc_caps"
    assert enc_caps.link(encoder),                      "enc_caps → encoder"
    assert encoder.link(h264parse),                     "encoder → h264parse"
    assert h264parse.link(flvmux),                      "h264parse → flvmux"
    assert flvmux.link(rtmpsink),                       "flvmux → rtmpsink"

    # Metadata probe on OSD sink pad
    osd_pad = nvosd.get_static_pad("sink")
    if osd_pad:
        osd_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    return pipeline


def main():
    Gst.init(None)
    print("[INFO] Building pipeline …")
    pipeline = build_pipeline()

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print(f"[INFO] Streaming to: {RTMP_URI}")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Pipeline failed to start.")
        pipeline.set_state(Gst.State.NULL)
        sys.exit(1)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("[INFO] Interrupted.")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("[INFO] Pipeline stopped.")


if __name__ == "__main__":
    main()