# ffmpeg -re -i output.mp4 -f mpegts udp://100.115.118.128:1000
ffmpeg -re -i <input> -c:v libx264 -preset ultrafast -tune zerolatency -f mpegts udp://<tailscale-ip>:<port>
ffmpeg -re -i output.mp4 -c:v libx264 -preset ultrafast -tune zerolatency -f mpegts udp://100.115.118.128:1000
# ffmpeg -re -i <input> -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://0.0.0.0:<port>/<path>
#gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/stream0 ! rtph264depay ! rtph264pay ! udpsink host=100.114.2.124 port=1234