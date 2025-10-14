#!/usr/bin/env python3
"""
Organize recordings by date - combines and archives daily recordings.

This script:
1. Scans the recordings directory for MP4 files
2. Groups videos by date (format: motion_YYYYMMDD_HHMMSS.mp4)
3. For each day:
   - If date directory exists: appends new videos to existing combined video
   - If date directory is new: combines all videos into one file (ordered by time)
   - Creates subdirectory with the date
   - Moves all videos and the combined video to that subdirectory
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict
import re
import tempfile


def parse_video_filename(filename):
    """
    Extract date and time from video filename.

    Expected format: motion_20251014_153045.mp4 or manual_20251014_153045.mp4

    Returns:
        tuple: (date_str, datetime_obj) or (None, None) if parsing fails
    """
    # Pattern: prefix_YYYYMMDD_HHMMSS.mp4
    pattern = r'^(?:motion|manual)_(\d{8})_(\d{6})\.mp4$'
    match = re.match(pattern, filename)

    if not match:
        return None, None

    date_str = match.group(1)  # YYYYMMDD
    time_str = match.group(2)  # HHMMSS

    try:
        # Parse into datetime
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        return date_str, dt
    except ValueError:
        return None, None


def get_video_duration(video_path):
    """
    Get video duration using ffprobe.

    Returns:
        float: Duration in seconds, or 0 if unable to determine
    """
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass

    return 0


def combine_videos(video_files, output_path):
    """
    Combine multiple videos into one using ffmpeg concat demuxer.

    Args:
        video_files: List of Path objects, ordered by time
        output_path: Path object for output file

    Returns:
        bool: True if successful, False otherwise
    """
    if len(video_files) == 0:
        print("  ❌ No videos to combine")
        return False

    if len(video_files) == 1:
        print(f"  ℹ️  Only one video for this day, skipping combination")
        return False

    # Create temporary file list for ffmpeg concat
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        for video in video_files:
            # Convert to absolute path and escape single quotes
            abs_path = video.resolve()
            escaped_path = str(abs_path).replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")

    try:
        print(f"  🎬 Combining {len(video_files)} videos...")
        print(f"     Output: {output_path.name}")

        # Use concat demuxer (fastest, no re-encoding)
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',  # Copy streams without re-encoding
            '-y',  # Overwrite output file
            str(output_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            # Verify output file exists and has reasonable size
            if output_path.exists() and output_path.stat().st_size > 1000:
                print(f"  ✅ Combined video created: {output_path.name}")
                return True
            else:
                print(f"  ❌ Combined video is invalid or too small")
                if output_path.exists():
                    output_path.unlink()
                return False
        else:
            # Show last 500 chars of error (most relevant part)
            error_msg = result.stderr.strip()
            if len(error_msg) > 500:
                error_msg = "..." + error_msg[-500:]
            print(f"  ❌ ffmpeg failed:")
            for line in error_msg.split('\n')[-10:]:  # Last 10 lines
                if line.strip():
                    print(f"     {line}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ❌ ffmpeg timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"  ❌ Error during combination: {e}")
        return False
    finally:
        # Clean up temp file
        try:
            os.unlink(concat_file)
        except:
            pass


def organize_recordings(recordings_dir="recordings"):
    """
    Main function to organize recordings by date.

    Args:
        recordings_dir: Directory containing recordings (default: "recordings")
    """
    recordings_path = Path(recordings_dir).resolve()  # Convert to absolute path

    if not recordings_path.exists():
        print(f"❌ Recordings directory not found: {recordings_dir}")
        return

    print(f"📁 Scanning recordings directory: {recordings_dir}")
    print("="*60)

    # Group videos by date
    videos_by_date = defaultdict(list)
    skipped_files = []

    for file_path in recordings_path.glob("*.mp4"):
        # Skip files in subdirectories (already organized)
        if file_path.parent != recordings_path:
            continue

        date_str, dt = parse_video_filename(file_path.name)

        if date_str is None:
            skipped_files.append(file_path.name)
            continue

        videos_by_date[date_str].append((dt, file_path))

    if skipped_files:
        print(f"\n⚠️  Skipped {len(skipped_files)} files with unexpected names:")
        for name in skipped_files[:5]:
            print(f"   - {name}")
        if len(skipped_files) > 5:
            print(f"   ... and {len(skipped_files) - 5} more")

    if not videos_by_date:
        print(f"\n✅ No videos to organize")
        return

    print(f"\n📊 Found videos from {len(videos_by_date)} day(s) to organize")
    print("="*60)

    # Process each day
    for date_str in sorted(videos_by_date.keys()):
        videos = videos_by_date[date_str]

        # Sort videos by time
        videos.sort(key=lambda x: x[0])
        video_files = [path for _, path in videos]

        # Format date for display
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        formatted_date = date_obj.strftime("%Y-%m-%d")

        print(f"\n📅 Processing {formatted_date} ({len(videos)} new video(s))")

        # Calculate total duration of new videos
        total_duration = sum(get_video_duration(v) for v in video_files)
        duration_str = f"{int(total_duration // 60)}m {int(total_duration % 60)}s" if total_duration > 0 else "unknown"
        print(f"   New videos duration: {duration_str}")

        # Create subdirectory if it doesn't exist
        subdir = recordings_path / date_str
        subdir_existed = subdir.exists()
        subdir.mkdir(exist_ok=True)

        if subdir_existed:
            print(f"   📂 Directory already exists: {subdir.name}/")
        else:
            print(f"   📂 Created directory: {subdir.name}/")

        # Check if combined video already exists
        combined_path = subdir / f"combined_{date_str}.mp4"
        existing_combined = combined_path.exists()

        combined_success = False
        videos_to_combine = video_files.copy()

        if existing_combined:
            print(f"   🔄 Found existing combined video, will update it")
            # Add existing combined video at the beginning (it contains all previous videos)
            # But we need to figure out where to insert new videos chronologically
            # For simplicity, we'll just append new videos at the end
            # This assumes new videos are chronologically after the existing combined video
            videos_to_combine = [combined_path] + video_files

        # Combine videos if we have multiple files
        if len(videos_to_combine) > 1:
            # Create temporary combined file
            temp_combined = subdir / f"combined_{date_str}_temp.mp4"
            combined_success = combine_videos(videos_to_combine, temp_combined)

            if combined_success:
                # Replace old combined video with new one
                if existing_combined:
                    try:
                        combined_path.unlink()
                        print(f"   🗑️  Removed old combined video")
                    except Exception as e:
                        print(f"   ⚠️  Could not remove old combined: {e}")

                try:
                    temp_combined.rename(combined_path)
                    print(f"   ✅ Updated combined video")
                except Exception as e:
                    print(f"   ❌ Failed to rename combined video: {e}")
                    combined_success = False
        elif len(video_files) == 1 and not existing_combined:
            print(f"   ℹ️  Only one video, no combination needed")

        # Move new videos to subdirectory
        print(f"  📦 Moving {len(video_files)} video(s) to {subdir.name}/")
        moved_count = 0
        for video_path in video_files:
            try:
                dest_path = subdir / video_path.name
                video_path.rename(dest_path)
                moved_count += 1
            except Exception as e:
                print(f"  ❌ Failed to move {video_path.name}: {e}")

        if moved_count == len(video_files):
            print(f"  ✅ Moved all videos successfully")
        else:
            print(f"  ⚠️  Moved {moved_count}/{len(video_files)} videos")

        # Summary
        if combined_success and combined_path.exists():
            combined_size_mb = combined_path.stat().st_size / (1024 * 1024)
            print(f"  💾 Combined video size: {combined_size_mb:.1f} MB")

    print("\n" + "="*60)
    print("✅ Organization complete!")


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Organize motion detection recordings by date',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Organize recordings in default directory
  python organize_recordings.py

  # Organize recordings in custom directory
  python organize_recordings.py --dir /path/to/recordings

This script will:
  1. Group videos by date (from filename)
  2. Combine all videos from each day into one file
  3. Create subdirectories by date (YYYYMMDD)
  4. Move original and combined videos to subdirectories
  5. If date directory exists: append new videos to combined video

Safe to run multiple times - appends new videos to existing combined files
"""
    )

    parser.add_argument(
        '--dir',
        type=str,
        default='recordings',
        help='Recordings directory (default: recordings)'
    )

    args = parser.parse_args()

    try:
        organize_recordings(args.dir)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
