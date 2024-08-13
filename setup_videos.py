import os
import subprocess


def extract_frames(video_path, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # FFmpeg command to extract frames
    ffmpeg_command = [
        'ffmpeg',
        '-i', video_path,             # Input video file
        '-q:v', '2',                  # High-quality JPEG frames
        '-start_number', '0',         # Start numbering frames at 00000
        os.path.join(output_dir, '%05d.jpg')  # Output frame format
    ]

    # Run the FFmpeg command
    subprocess.run(ffmpeg_command, check=True)


def extract_frames_for_all_videos(directory_path):
    # List all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Check for video file extensions
            video_path = os.path.join(directory_path, filename)
            video_name, _ = os.path.splitext(filename)  # Get the video name without the extension
            output_dir = os.path.join(directory_path, video_name)  # Set output directory to video name
            extract_frames(video_path, output_dir)  # Extract frames for this video


if __name__ == "__main__":
    # Example usage:
    directory_path = 'videos'  # Path to the directory containing videos
    extract_frames_for_all_videos(directory_path)
