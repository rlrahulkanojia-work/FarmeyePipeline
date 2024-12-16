#!/bin/bash

# Default settings
INPUT_DEVICE="0"  # Default webcam
FRAMERATE="30"    # Default framerate
RESOLUTION="1280x720"  # Default resolution
RTSP_PORT="8554"  # Default RTSP port
STREAM_NAME="webcam"  # Default stream name

# Function to check and install dependencies
check_dependencies() {
    # Check FFmpeg
    if ! command -v ffmpeg &> /dev/null; then
        echo "FFmpeg is not installed. Installing..."
        brew install ffmpeg
    fi

    # Check RTSP Simple Server
    if ! command -v rtsp-simple-server &> /dev/null; then
        echo "RTSP Simple Server is not installed. Installing..."
        brew install rtsp-simple-server
    fi
}

# Function to start RTSP server
start_server() {
    echo "Starting RTSP server..."
    # Kill any existing rtsp-simple-server process
    pkill rtsp-simple-server 2>/dev/null
    # Start the server in background
    rtsp-simple-server &
    # Wait for server to start
    sleep 2
}

# Function to list available video devices
list_devices() {
    echo "Available video devices:"
    ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep "\[AVFoundation"
}

# Function to start streaming
start_stream() {
    echo "Starting RTSP stream on rtsp://localhost:${RTSP_PORT}/${STREAM_NAME}"
    ffmpeg -f avfoundation -framerate ${FRAMERATE} \
        -video_size ${RESOLUTION} -i "${INPUT_DEVICE}:none" \
        -c:v libx264 -preset ultrafast -tune zerolatency \
        -b:v 1000k -f rtsp \
        -rtsp_transport tcp \
        rtsp://localhost:${RTSP_PORT}/${STREAM_NAME}
}

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    pkill rtsp-simple-server
    exit 0
}

# Set up cleanup trap
trap cleanup EXIT

# Check and install dependencies
check_dependencies

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--device)
            INPUT_DEVICE="$2"
            shift 2
            ;;
        -f|--framerate)
            FRAMERATE="$2"
            shift 2
            ;;
        -r|--resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        -p|--port)
            RTSP_PORT="$2"
            shift 2
            ;;
        -n|--name)
            STREAM_NAME="$2"
            shift 2
            ;;
        -l|--list)
            list_devices
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Start the server and stream
start_server
start_stream