# Industrial Vision Position Detection

This repository contains a Python script for detecting and annotating positions of objects in test images using template matching and contour analysis.

## Features

- Loads template and test images.
- Performs binarization and contour extraction.
- Matches shapes between template and test images.
- Filters detected objects based on size deviation.
- Annotates detected objects with bounding boxes, centers, and angles.
- Corrects angles of detected objects.
- Saves annotated images and writes detection results to a text file.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

Install the required Python packages using pip:

```bash
pip install opencv-python-headless numpy
