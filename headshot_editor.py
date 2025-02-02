from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2
import pytesseract
import os
import numpy as np
from fuzzywuzzy import fuzz  # Requires `pip install fuzzywuzzy`

# Tesseract configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Adjusted ROI (based on tests)
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 1100, 800, 1400, 900  # x1, y1, x2, y2

def preprocess_image(image):
    """Enhance text clarity without losing details."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply sharpening filter to highlight text
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened

def is_similar_to_headshot(text):
    """Check if the detected text contains or is similar to 'HEADSHOT'."""
    text = text.upper().replace(" ", "")  # Convert to uppercase and remove spaces

    # 1️⃣ Check if "HEADSHOT" is exactly in the text
    if "HEADSHOT" in text:
        return True

    # 2️⃣ Compare similarity if there are extra characters
    # similarity = fuzz.partial_ratio(text, "HEADSHOT")  # Allows extra characters
    # return similarity >= 98  # Accepts results that are at least 98% similar

def detect_headshot(video_path, output_path, debug=False, max_gap=2, transition_duration=0.5, pre_headshot_duration=7, post_headshot_duration=2):
    clip = VideoFileClip(video_path)
    highlights = []
    last_detection_time = -10  # Avoid duplicate detections
    current_clip_start = None  # Start of current sequence
    current_clip_end = None    # End of current sequence

    if debug:
        os.makedirs("debug_frames", exist_ok=True)

    for current_time in range(0, int(clip.duration)):
        frame = clip.get_frame(current_time)
        roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        processed_roi = preprocess_image(roi)

        if debug:
            cv2.imwrite(f"debug_frames/roi_{current_time}.png", roi)  # Original
            cv2.imwrite(f"debug_frames/processed_{current_time}.png", processed_roi)  # Processed

        text = pytesseract.image_to_string(
            processed_roi,
            config='--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        ).strip()

        if debug:
            print(f"Time: {current_time}s → Detected Text: '{text}'")

        if is_similar_to_headshot(text) and (current_time - last_detection_time) > 2:
            last_detection_time = current_time

            if current_clip_start is None:  # If it's the first detection, start a new sequence
                current_clip_start = current_time - pre_headshot_duration
                current_clip_end = current_time + post_headshot_duration
            elif current_time - current_clip_end <= max_gap:  # If it's close to the last sequence
                current_clip_end = current_time + post_headshot_duration  # Extend the sequence
            else:  # If it's too far, add the previous sequence and start a new one
                highlights.append(clip.subclip(current_clip_start, current_clip_end))
                current_clip_start = current_time - pre_headshot_duration
                current_clip_end = current_time + post_headshot_duration

    # Add the last sequence if it exists
    if current_clip_start is not None:
        highlights.append(clip.subclip(current_clip_start, current_clip_end))

    if not highlights:
        print("Error: No 'HEADSHOT' detected! Check debug_frames/.")
        return

    # Apply transition between clips
    final_clips = []
    for i in range(len(highlights)-1):
        final_clips.append(highlights[i].crossfadeout(transition_duration))  # Transition out
    final_clips.append(highlights[-1])  # Add the last clip without transition

    final_clip = concatenate_videoclips(final_clips, method="compose")
    
    # Optimize video processing
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=16, preset="ultrafast")
    clip.reader.close()
    clip.audio.reader.close_proc()

# Run with debug enabled
detect_headshot("input_video.mp4", "highlights_output.mp4", debug=True)
