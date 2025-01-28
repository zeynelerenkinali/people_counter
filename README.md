# People Counter with YOLOv8

This project is a people counting system using YOLOv8 for object detection. The system processes video input, detects people, and tracks them across frames using the SORT tracking algorithm.

## Features

- **YOLOv8 Object Detection**: Utilizes the YOLOv8 model for real-time person detection.
- **Object Tracking**: Tracks detected people with the SORT algorithm.
- **Multiple Count Zones**: Counts people crossing predefined lines in two different regions.
- **Masked Region**: Uses a mask to focus detection on specific areas of the video.
- **Overlay Graphics**: Displays custom overlay graphics on the output video.

## Requirements

- Python 3.7+
- OpenCV
- cvzone
- torch
- numpy
- Sort tracking algorithm

You can install the necessary packages using:

```bash
pip install opencv-python-headless ultralytics torch numpy sort cvzone
```
## Usage

    Clone the repository: Clone this repository to your local machine.

git clone https://github.com/yourusername/people-counter-yolov8.git
cd people-counter-yolov8

Download YOLOv8 weights: Download the YOLOv8 model weights (yolov8l.pt) and place them in the yolo_weights/ directory.

Run the project: Execute the script to process the video input and count the number of people crossing the predefined lines.

    python people_counter.py

## How It Works

    The system loads a pre-trained YOLOv8 model from the yolo_weights/ directory to detect people in the input video.
    A mask image (mask.png) is used to focus detection on specific regions of the video.
    The SORT tracking algorithm tracks the detected people across video frames.
    The system counts how many people cross predefined lines in two regions of the video (with different color-coded lines).
    Custom overlay graphics are displayed on the output video, along with the people count.

## Demo

A demo video (videos/people.mp4) is processed using the YOLOv8 people counter. The output displays the number of people crossing each of the two defined zones.

To view the output video with the people count, run the project as shown in the Usage section.

## References

    YOLOv8 by Ultralytics
    SORT Tracking Algorithm
    OpenCV Documentation
