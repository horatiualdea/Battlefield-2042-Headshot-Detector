# Battlefield 2042 Headshot Highlight Detection

This project is designed to detect "HEADSHOT" text in Battlefield 2042 gameplay videos and extract the corresponding highlights with simple transitions between them.

## Features

- Detects "HEADSHOT" text in gameplay.
- Extracts highlights before and after each detected headshot.
- Creates smooth transitions between video segments.
- Adjustable time for pre-headshot and post-headshot footage.
- Fast video processing using MoviePy.

## Requirements

- Python 3.x
- Tesseract-OCR (for text detection)
- Fuzzywuzzy (for fuzzy text matching)
- OpenCV (for image processing)
- MoviePy (for video editing)

### Installation

1. Install Python 3.x: [Download Python](https://www.python.org/downloads/)
2. Install Tesseract-OCR:
   - Windows: [Download Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt install tesseract-ocr`
3. Install the required Python packages:
   ```bash
   pip install moviepy opencv-python pytesseract fuzzywuzzy
4. Set the path to the Tesseract executable in the script:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Change as needed

### Usage

Run the script with the following command:
    ```bash
    python headshot_highlight_detection.py

The script will process the video and generate a highlight reel based on detected headshots. 
You can adjust the parameters like the duration before and after a headshot, as well as the transition duration between clips.

### Customization

pre_headshot_duration: Time before the "HEADSHOT" text appears to be included in the highlight.
post_headshot_duration: Time after the "HEADSHOT" text appears to be included in the highlight.
transition_duration: Duration of the crossfade transition between clips.

### Troubleshooting

Ensure that Tesseract-OCR is installed correctly and the path is set properly in the script.
If the script fails to detect any headshots, try adjusting the ROI coordinates to match the area where the "HEADSHOT" text appears in your game.