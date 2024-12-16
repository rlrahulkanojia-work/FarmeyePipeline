import cv2
import time

# Open RTSP stream
cap = cv2.VideoCapture("rtsp://100.100.204.36:8554/webcam") #farmhub IP

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Set start time
start_time = time.time()
duration = 30  # Duration in seconds

while True:
    ret, frame = cap.read()
    if ret:
        
        out.write(frame)
        
        if time.time() - start_time > duration:
            break
    else:
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()