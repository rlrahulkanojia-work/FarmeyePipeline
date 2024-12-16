import cv2
import os
import argparse

def time_to_seconds(time_str):
    """
    Convert time in MM:SS format to seconds.
    
    :param time_str: Time string in MM:SS format (e.g., '02:30' means 2 minutes 30 seconds or 'end' for full duration).
    :return: Time in seconds.
    """
    if time_str.lower() == 'end':
        return -1  # Use -1 to indicate the end of the video
    minutes, seconds = map(int, time_str.split(":"))
    return minutes * 60 + seconds

def extract_frames(video_path, output_folder, start_time="00:00", end_time="-1:-1", fps=1):
    """
    Extract frames from a video within a specified time range (MM:SS format).
    
    :param video_path: Path to the video file.
    :param output_folder: Folder to save extracted frames.
    :param start_time: Start time in MM:SS format for frame extraction (default is '00:00').
    :param end_time: End time in MM:SS format for frame extraction (default is '-1:-1', which means the full video).
    :param fps: Number of frames per second to extract (default is 1 frame per second).
    """
    # Convert times from MM:SS to seconds
    start_time_sec = time_to_seconds(start_time)
    end_time_sec = time_to_seconds(end_time)

    # Check if output folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Get the video frame rate and duration
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_duration = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps
    
    if end_time_sec == -1 or end_time_sec > video_duration:
        end_time_sec = video_duration  # Set end_time to the video duration if not specified

    # Convert start and end times to frame numbers
    start_frame = int(start_time_sec * video_fps)
    end_frame = int(end_time_sec * video_fps)

    # Set the current frame position to the start_frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    frame_number = 0

    while current_frame <= end_frame:
        # Read the frame
        ret, frame = video_capture.read()
        if not ret:
            break  # Exit if no frame is read
        
        # Save the frame at the specified fps
        if current_frame % int(video_fps / fps) == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_number}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_number += 1

        current_frame += 1

    # Release the video capture
    video_capture.release()
    print(f"Extracted {frame_number} frames from {start_time} to {end_time}.")

if __name__ == "__main__":
    # Set up argparse to handle CLI arguments
    parser = argparse.ArgumentParser(description="Extract frames from a video within a specified time range (MM:SS format).")
    
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("output_folder", type=str, help="Folder to save extracted frames.")
    parser.add_argument("--start_time", type=str, default="00:00", help="Start time in MM:SS format for frame extraction (default is '00:00').")
    parser.add_argument("--end_time", type=str, default="-1:-1", help="End time in MM:SS format for frame extraction (default is '-1:-1', which means the full video).")
    parser.add_argument("--fps", type=int, default=1, help="Number of frames per second to extract (default is 1 frame per second).")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the frame extraction function with the parsed arguments
    extract_frames(args.video_path, args.output_folder, args.start_time, args.end_time, args.fps)
