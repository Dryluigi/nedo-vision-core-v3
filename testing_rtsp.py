#!/usr/bin/env python3

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

# Initialize GStreamer
Gst.init(None)

def bus_call(bus, message, loop):
    """Callback function for GStreamer bus messages"""
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    return True

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """Probe function to access buffer data"""
    return Gst.PadProbeReturn.OK

def main(rtsp_url, rtmp_url):
    # Create pipeline
    pipeline = Gst.Pipeline()
    
    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
        return
    
    # Create elements
    print("Creating elements...")
    
    # RTSP source
    source = Gst.ElementFactory.make("rtspsrc", "rtsp-source")
    
    # Depayloader
    rtppay = Gst.ElementFactory.make("rtph264depay", "rtppay")
    
    # H264 parser
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    
    # Hardware decoder
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    
    # Stream muxer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    
    # Inference server
    nvinferserver = Gst.ElementFactory.make("nvinferserver", "nvinferserver")
    
    # Converter
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    
    # OSD (On-Screen Display)
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    
    # Video converter for encoding
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    
    # Caps filter for encoder input
    caps_vidconv = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    
    # H264 encoder
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    
    # H264 parser for encoder output
    h264parser_enc = Gst.ElementFactory.make("h264parse", "h264-parser-enc")
    
    # FLV muxer for RTMP
    flvmux = Gst.ElementFactory.make("flvmux", "flv-muxer")
    
    # RTMP sink
    sink = Gst.ElementFactory.make("rtmpsink", "rtmp-sink")
    
    if not source or not rtppay or not h264parser or not decoder or not streammux or \
       not nvinferserver or not nvvidconv or not nvosd or not nvvidconv_postosd or \
       not caps_vidconv or not encoder or not h264parser_enc or not flvmux or not sink:
        sys.stderr.write("One element could not be created. Exiting.\n")
        return
    
    # Set properties
    print("Setting properties...")
    
    # RTSP source properties
    source.set_property("location", rtsp_url)
    source.set_property("latency", 100)
    
    # Stream muxer properties
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)
    
    # Inference server properties
    nvinferserver.set_property("config-file-path", "/app/config/deepstream-inferserver-yolo.txt")
    
    # OSD properties
    nvosd.set_property("process-mode", 1)  # GPU mode
    nvosd.set_property("display-text", 1)
    
    # Caps for encoder input (I420 format)
    caps_str = "video/x-raw(memory:NVMM), format=I420"
    caps_vidconv.set_property("caps", Gst.Caps.from_string(caps_str))
    
    # Encoder properties
    encoder.set_property("bitrate", 4000000)  # 4 Mbps
    encoder.set_property("iframeinterval", 30)  # I-frame every 30 frames
    
    # H264 parser properties
    h264parser_enc.set_property("config-interval", -1)
    
    # FLV muxer properties
    flvmux.set_property("streamable", True)
    
    # RTMP sink properties
    sink.set_property("location", rtmp_url)
    sink.set_property("sync", 0)
    
    # Add elements to pipeline
    print("Adding elements to Pipeline...")
    pipeline.add(source)
    pipeline.add(rtppay)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(nvinferserver)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps_vidconv)
    pipeline.add(encoder)
    pipeline.add(h264parser_enc)
    pipeline.add(flvmux)
    pipeline.add(sink)
    
    # Link elements (except source which needs pad-added signal)
    print("Linking elements in the Pipeline...")
    
    rtppay.link(h264parser)
    h264parser.link(decoder)
    
    # Link decoder to streammux
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write("Unable to get the sink pad of streammux\n")
        return
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write("Unable to get source pad of decoder\n")
        return
    srcpad.link(sinkpad)
    
    # Link rest of the pipeline
    streammux.link(nvinferserver)
    nvinferserver.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps_vidconv)
    caps_vidconv.link(encoder)
    encoder.link(h264parser_enc)
    h264parser_enc.link(flvmux)
    flvmux.link(sink)
    
    # Connect to pad-added signal for dynamic linking
    def pad_added_handler(src, new_pad):
        print(f"Received new pad '{new_pad.get_name()}' from '{src.get_name()}'")
        
        # Check if this is the RTP stream we want
        new_pad_caps = new_pad.get_current_caps()
        new_pad_struct = new_pad_caps.get_structure(0)
        new_pad_type = new_pad_struct.get_name()
        
        if new_pad_type.startswith("application/x-rtp"):
            sink_pad = rtppay.get_static_pad("sink")
            if not sink_pad.is_linked():
                ret = new_pad.link(sink_pad)
                if ret == Gst.PadLinkReturn.OK:
                    print("Link succeeded (type '%s')" % new_pad_type)
                else:
                    print("Link failed (type '%s')" % new_pad_type)
    
    source.connect("pad-added", pad_added_handler)
    
    # Create an event loop and feed GStreamer bus messages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    # Add probe to get buffer data
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write("Unable to get sink pad of nvosd\n")
    else:
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    # Start play back
    print("Starting pipeline...")
    print(f"Streaming to RTMP: {rtmp_url}")
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    # Cleanup
    print("Stopping pipeline...")
    pipeline.set_state(Gst.State.NULL)
    print("Pipeline stopped")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: %s <RTSP URL> <RTMP URL>\n" % sys.argv[0])
        sys.stderr.write("Example: %s rtsp://localhost:8554/stream rtmp://localhost:1935/live/output\n" % sys.argv[0])
        sys.exit(1)
    
    rtsp_url = sys.argv[1]
    rtmp_url = sys.argv[2]
    print(f"Using RTSP URL: {rtsp_url}")
    print(f"Using RTMP URL: {rtmp_url}")
    main(rtsp_url, rtmp_url)