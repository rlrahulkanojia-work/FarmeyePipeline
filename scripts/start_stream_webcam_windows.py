import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib
import sys

class WebcamServer:
    def __init__(self):
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")
        
        # Create a factory for the stream
        self.factory = GstRtspServer.RTSPMediaFactory()
        
        # Create the pipeline string
        pipeline = (
            "ksvideosrc device-index=0 ! "
            "videoconvert ! "
            "x264enc tune=zerolatency ! "
            "rtph264pay name=pay0 pt=96"
        )
        
        self.factory.set_launch(pipeline)
        self.factory.set_shared(True)
        
        # Attach the factory to the server
        self.server.get_mount_points().add_factory("/webcam", self.factory)
        
        # Start the server
        self.server.attach(None)

def main():
    # Initialize GStreamer
    Gst.init(sys.argv[1:])
    
    server = WebcamServer()
    print("Stream ready at rtsp://127.0.0.1:8554/webcam")
    
    try:
        # Run the main loop
        GLib.MainLoop().run()
    except KeyboardInterrupt:
        print("Stopping server...")
        sys.exit(0)

if __name__ == '__main__':
    main()