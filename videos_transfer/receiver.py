import cv2

# Replace port with the same port as sender
udp_url = "udp://@:1234"

cap = cv2.VideoCapture(udp_url)

if not cap.isOpened():
    print("Failed to open UDP stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    print(frame.shape)

cap.release()
cv2.destroyAllWindows()
