import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib
import os

class RTSPServer:
    def __init__(self, video_path, port=8554):
        Gst.init(None)
        
        self.video_path = os.path.abspath(video_path)
        self.port = port
        
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service(str(port))
        
        # Create a media factory
        self.factory = GstRtspServer.RTSPMediaFactory()
        
        # Create the pipeline
        pipeline_str = (
            f'filesrc location="{self.video_path}" ! '
            'decodebin ! videoconvert ! x264enc tune=zerolatency ! '
            'rtph264pay name=pay0 pt=96'
        )
        
        self.factory.set_launch(pipeline_str)
        self.factory.set_shared(True)
        
        # Attach the factory to the server
        self.server.get_mount_points().add_factory("/stream", self.factory)
        
        # Start the server
        self.server.attach(None)
    
    def run(self):
        self.loop = GLib.MainLoop()
        print(f"Stream ready at rtsp://127.0.0.1:{self.port}/stream")
        self.loop.run()
    
    def stop(self):
        if hasattr(self, 'loop'):
            self.loop.quit()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_video_file>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    server = RTSPServer(video_path)
    
    try:
        server.run()
    except KeyboardInterrupt:
        server.stop()