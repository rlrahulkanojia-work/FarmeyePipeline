ffmpeg -i udp://@:1234 -c:v copy -c:a copy output.mp4
ffmpeg -i udp://100.100.204.36:1000 -c:v copy -c:a copy output.mp4
timeout 20 gst-launch-1.0 -e rtspsrc location=rtsp://100.100.204.36:8554/webcam ! queue ! rtph264depay ! h264parse ! matroskamux ! filesink location=output.mkv