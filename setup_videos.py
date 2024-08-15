import os
import subprocess


def extract_frames(video_path, output_dir, max_frames=None):
    # Modify output directory name if max_frames is specified
    if max_frames is not None:
        output_dir = f"{output_dir}_{max_frames}"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Base FFmpeg command to extract frames
    ffmpeg_command = [
        'ffmpeg',
        '-i', video_path,             # Input video file
        '-q:v', '2',                  # High-quality JPEG frames
        '-start_number', '0',         # Start numbering frames at 00000
        os.path.join(output_dir, '%05d.jpg')  # Output frame format
    ]

    # Add the limit on the number of frames if max_frames is specified
    if max_frames is not None:
        ffmpeg_command.insert(-1, '-frames:v')
        ffmpeg_command.insert(-1, str(max_frames))

    # Run the FFmpeg command
    subprocess.run(ffmpeg_command, check=True)


def extract_frames_for_all_videos(directory_path, max_frames=None):
    # List all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Check for video file extensions
            video_path = os.path.join(directory_path, filename)
            video_name, _ = os.path.splitext(filename)  # Get the video name without the extension
            output_dir = os.path.join(directory_path, video_name)  # Set output directory to video name
            extract_frames(video_path, output_dir, max_frames)  # Extract frames for this video


if __name__ == "__main__":
    # Example usage:
    directory_path = 'videos'  # Path to the directory containing videos
    max_frames = 100  # Optional: specify the maximum number of frames to extract
    extract_frames_for_all_videos(directory_path, max_frames)
