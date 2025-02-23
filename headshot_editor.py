import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2
import pytesseract
import numpy as np
import subprocess
from fuzzywuzzy import fuzz  # Requires `pip install fuzzywuzzy`
from moviepy.video.fx.all import crop

# Tesseract Configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(image):
    """Enhances text clarity without losing details."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply sharpening filter to highlight the text
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


def is_similar_to_headshot(text):
    """Checks if the detected text contains or is similar to 'HEADSHOT'."""
    text = text.upper().replace(" ", "")
    return "HEADSHOT" in text


def convert_to_mp4(input_path, output_path):
    """Converts MKV to MP4 using ffmpeg."""
    subprocess.run(["ffmpeg", "-i", input_path, "-codec", "copy", output_path], check=True)


def crop_to_phone_format(clip):
    """Crops a video to 9:16 aspect ratio for vertical display."""
    width, height = clip.size
    target_width = int(height * (9 / 16))

    if target_width > width:
        target_width = width  # Prevent invalid crop

    if target_width % 2 != 0:
        target_width -= 1  

    x1 = max((width - target_width) // 2, 0)
    x2 = min(x1 + target_width, width)

    return crop(clip, x1=x1, y1=0, x2=x2, y2=height)


def detect_headshot(video_path, output_basename, debug=False, max_gap=None, transition_duration=0.5, pre_headshot_duration=7, post_headshot_duration=2, max_video_duration=60):
    if max_gap is None:
        max_gap = pre_headshot_duration

    if video_path.endswith(".mkv"):
        mp4_path = video_path.replace(".mkv", ".mp4")
        convert_to_mp4(video_path, mp4_path)
        video_path = mp4_path

    clip = VideoFileClip(video_path)

    # Detect video resolution to set correct ROI
    _, height = clip.size
    if height == 1080:
        ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 1100, 800, 1400, 900
    elif height == 1440:
        ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 1466, 1066, 1866, 1200
    else:
        print(f"Warning: Unexpected resolution {height}p. Using default 1080p ROI.")
        ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 1100, 800, 1400, 900  # Default to 1080p ROI

    highlights = []
    last_detection_time = -10
    current_clip_start = None
    current_clip_end = None

    if debug:
        os.makedirs("debug_frames", exist_ok=True)

    for current_time in range(0, int(clip.duration)):
        frame = clip.get_frame(current_time)

        roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

        if roi is None or roi.size == 0:
            print(f"Error: Empty ROI at {current_time}s")
            continue

        processed_roi = preprocess_image(roi)

        if debug:
            cv2.imwrite(f"debug_frames/roi_{current_time}.png", roi)
            cv2.imwrite(f"debug_frames/processed_{current_time}.png", processed_roi)

        text = pytesseract.image_to_string(
            processed_roi,
            config='--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        ).strip()

        if debug:
            print(f"Time: {current_time}s → Detected text: '{text}'")

        if is_similar_to_headshot(text) and (current_time - last_detection_time) > 2:
            last_detection_time = current_time

            if current_clip_start is None:
                current_clip_start = current_time - pre_headshot_duration
                current_clip_end = current_time + post_headshot_duration
            elif current_time - current_clip_end <= max_gap:
                current_clip_end = current_time + post_headshot_duration
            else:
                highlights.append(clip.subclip(current_clip_start, current_clip_end))
                current_clip_start = current_time - pre_headshot_duration
                current_clip_end = current_time + post_headshot_duration

    if current_clip_start is not None:
        highlights.append(clip.subclip(current_clip_start, current_clip_end))

    if not highlights:
        print(f"Error: No 'HEADSHOT' detected in {video_path}! Check debug_frames/.")
        return

    # Splitting into multiple videos if needed
    video_index = 1
    current_duration = 0
    current_clips = []

    for highlight in highlights:
        highlight_duration = highlight.duration

        if current_duration + highlight_duration > max_video_duration:
            # Save the current video and start a new one
            final_clip = concatenate_videoclips(current_clips, method="compose")
            portrait_clip = crop_to_phone_format(final_clip)

            output_file = f"{output_basename}_part{video_index}.mp4"
            portrait_clip.write_videofile(output_file, codec="libx264", audio_codec="aac", threads=64, preset="ultrafast", ffmpeg_params=["-pix_fmt", "yuv420p"])

            # Reset for new video
            video_index += 1
            current_duration = 0
            current_clips = []

        current_clips.append(highlight)
        current_duration += highlight_duration

    if current_clips:
        final_clip = concatenate_videoclips(current_clips, method="compose")
        portrait_clip = crop_to_phone_format(final_clip)

        output_file = f"{output_basename}_part{video_index}.mp4"
        portrait_clip.write_videofile(output_file, codec="libx264", audio_codec="aac", threads=64, preset="ultrafast", ffmpeg_params=["-pix_fmt", "yuv420p"])

    clip.reader.close()
    clip.audio.reader.close_proc()


# **Batch Processing for Multiple Videos**
def process_multiple_videos(input_folder="input_videos", output_folder="output_videos", debug=False):
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(input_folder) if f.endswith((".mp4", ".mkv"))]

    if not video_files:
        print("No videos found in input folder.")
        return

    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"highlight_{video_file}")

        print(f"Processing {input_path} → {output_path}")
        detect_headshot(input_path, output_path, debug=debug)

    print("Batch processing complete!")


# Run batch processing
process_multiple_videos(debug=True)
