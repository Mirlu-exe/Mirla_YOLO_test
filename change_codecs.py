import subprocess
import pathlib
import sys
import time

# --- Configuration ---
# Set the directory containing the videos you want to process.
# If you leave this as '.', it will use the same directory the script is run from.
SOURCE_DIR = pathlib.Path('/home/nishiyamalab/PycharmProjects/Mirla_YOLO_test/unseen_videos/Comparision Video results')
"""'/home/nishiyamalab/Videos/pinnawala_sample_videos'"""
OUTPUT_SUFFIX = "_h264.mp4"
# A list of video extensions to look for.
VIDEO_EXTENSIONS = ['.MP4', '.mov', '.mkv', '.avi', '.webm', '.flv', '.ts', '.mp4']

# --- FFmpeg Settings ---
# -c:v libx264: Sets the video codec to H.264.
# -crf 23: Sets the quality factor. 23 is a good default (lower = better quality/bigger file).
# -preset medium: Controls the encoding speed vs. compression ratio. 'medium' is a good balance.
# -c:a aac: Sets the audio codec to AAC (standard for MP4).
# -b:a 128k: Sets the audio bitrate.
FFMPEG_OPTIONS = [
    '-c:v', 'libx264',
    '-crf', '23',
    '-preset', 'medium',
    '-c:a', 'aac',
    '-b:a', '128k',
    '-movflags', '+faststart' # Optimizes the file for web streaming
]

def re_encode_videos():
    """
    Finds video files in the source directory and re-encodes them to H.264/AAC MP4 format
    using FFmpeg.
    """
    try:
        # Check if FFmpeg is available
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        print("✅ FFmpeg is installed and accessible.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ERROR: FFmpeg is not installed or not in your system's PATH.")
        print("Please install FFmpeg to use this script.")
        sys.exit(1)

    print(f"\nScanning directory: {SOURCE_DIR.resolve()}")
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        # Use rglob to find files recursively (if needed), or glob for current directory only.
        video_files.extend(SOURCE_DIR.glob(f'*{ext}'))

    if not video_files:
        print("🛑 No video files found with the specified extensions.")
        return

    print(f"Found {len(video_files)} video file(s) to process.")

    for i, input_file in enumerate(video_files):
        print(f"\n--- [{i + 1}/{len(video_files)}] Processing: {input_file.name} ---")

        # Create the output filename
        # Example: video.mov -> video_h264.mp4
        output_file_name = input_file.stem + OUTPUT_SUFFIX
        output_file = input_file.parent / output_file_name

        if output_file.exists():
            print(f"⏩ Output file already exists: {output_file.name}. Skipping.")
            continue

        # Construct the full FFmpeg command
        # ffmpeg -i <input_file> ... <output_file>
        command = [
            'ffmpeg',
            '-i', str(input_file), # Input file path
            *FFMPEG_OPTIONS,       # The list of encoding options
            '-y',                  # Automatically overwrite output files (if they exist, though we checked above)
            str(output_file)       # Output file path
        ]

        # Execute the command
        start_time = time.time()
        print(f"   Executing: {' '.join(command)}")
        try:
            # Note: FFmpeg prints progress to stderr, so we capture stdout, and let stderr print live.
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=sys.stderr)
            duration = time.time() - start_time
            print(f"✨ SUCCESS: Re-encoded to {output_file.name} in {duration:.2f} seconds.")

        except subprocess.CalledProcessError as e:
            print(f"❌ FAILED to re-encode {input_file.name}. FFmpeg exited with error code {e.returncode}.")
            # Note: Detailed FFmpeg error output is usually printed to stderr directly.
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    re_encode_videos()
