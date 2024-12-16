import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib
import platform
import sys

class WebcamRTSPServer:
    def __init__(self, device_index=0, port=8554):
        Gst.init(None)
        
        self.port = port
        
        # Determine the operating system
        self.os_type = platform.system()
        
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service(str(port))
        
        # Create a media factory
        self.factory = GstRtspServer.RTSPMediaFactory()
        
        # Create pipeline based on OS
        if self.os_type == "Linux":
            # Linux (V4L2) pipeline
            pipeline_str = (
                f'v4l2src device=/dev/video{device_index} ! '
                'video/x-raw,width=640,height=480,framerate=30/1 ! '
                'videoconvert ! x264enc tune=zerolatency ! '
                'rtph264pay name=pay0 pt=96'
            )
        elif self.os_type == "Windows":
            # Windows (DirectShow) pipeline
            pipeline_str = (
                'dshowvideosrc device-name="HD Webcam" ! '  # Replace "HD Webcam" with your webcam name
                'video/x-raw,width=640,height=480,framerate=30/1 ! '
                'videoconvert ! x264enc tune=zerolatency ! '
                'rtph264pay name=pay0 pt=96'
            )
        else:
            # macOS (AVFoundation) pipeline
            pipeline_str = (
                f'avfvideosrc device-index={device_index} ! '
                'video/x-raw,width=640,height=480,framerate=30/1 ! '
                'videoconvert ! x264enc tune=zerolatency ! '
                'rtph264pay name=pay0 pt=96'
            )
        
        self.factory.set_launch(pipeline_str)
        self.factory.set_shared(True)
        
        # Attach the factory to the server
        self.server.get_mount_points().add_factory("/webcam", self.factory)
        
        # Start the server
        self.server.attach(None)
    
    def run(self):
        self.loop = GLib.MainLoop()
        print(f"Webcam stream ready at rtsp://127.0.0.1:{self.port}/webcam")
        print("Press Ctrl+C to stop the server")
        self.loop.run()
    
    def stop(self):
        if hasattr(self, 'loop'):
            self.loop.quit()

if __name__ == "__main__":
    # Optional command line argument for device index
    device_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    server = WebcamRTSPServer(device_index=device_index)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()