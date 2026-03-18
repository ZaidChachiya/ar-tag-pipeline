# ar-tag-pipeline
A robust computer vision pipeline designed to detect fiducial markers (AR Tags) in video streams and perform planar Augmented Reality (AR) tasks.


***

# Augmented Reality Tag Detection and Rendering Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Fast%20Math-lightgrey.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Video%20I%2FO-green.svg)

## 📌 What is this?

This project is a robust, "from scratch" computer vision pipeline designed to detect planar Augmented Reality (AR) fiducial tags in video streams, decode them, and overlay arbitrary 2D media onto them in real-time. 

A major highlight of this project is that it is built from **first principles**. High-level library abstractions commonly used for these tasks (like `cv2.findHomography`, `cv2.warpPerspective`, or built-in contour detectors) were intentionally avoided. Instead, the mathematical and algorithmic foundations were implemented manually using linear algebra and NumPy.

### Key Features
* **Custom Image Processing Pipeline:** Includes custom grayscale conversion, global thresholding, and morphological boundary extraction using boolean logic.
* **Stack-Based Connected Components:** Implements a custom iterative Depth First Search (DFS) algorithm to find contiguous regions without hitting Python's recursion limits.
* **Robust Corner Detection:** Utilizes a custom *Longest Diagonal and Perpendicular Distance* algorithm to securely find the four corners of a tag, ignoring pixel aliasing and border noise.
* **Homography from Scratch:** Computes the $3 \times 3$ projective transformation matrix (Homography) by solving linear equations (`Ah = b`) to orthorectify the tag.
* **Inverse Warping & Decoding:** Corrects perspective distortion to accurately read the tag's orientation and decode its embedded 4-bit ID using adaptive thresholding.
* **Real-time 2D Overlay (Task 2):** Projects a user-defined template image perfectly onto the moving AR tag in the video feed using forward warping and bounding-box optimization.

---

## ⚙️ Installation

### Prerequisites
You will need **Python 3.x** installed on your system. 

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/ar-tag-detection.git
cd ar-tag-detection
```

### Step 2: Install Dependencies
The project relies on `numpy` for matrix operations and `opencv-python` strictly for video capturing, image reading, and GUI display.

```bash
pip install numpy opencv-python
```

---

## 🚀 How to Use

The main entry point for the application is `main.py`. It requires a template image to overlay onto the detected AR tags. You can either use a pre-recorded video or your live webcam feed.

### Command Line Arguments
* `--video`: (Optional) Path to the input video file. If not provided, the script defaults to the primary webcam (`0`).
* `--template`: (Required) Path to the image you want to overlay on top of the AR tag.

### Example Usage

**1. Using a Live Webcam**
```bash
python main.py --template assets/my_overlay_image.jpg
```

**2. Using a Video File**
```bash
python main.py --video assets/test_video.mp4 --template assets/my_overlay_image.jpg
```

### Controls during execution
* The application will open two windows:
  * `task1_frame`: Displays the raw video feed with bounding boxes, tag IDs, and orientation rotation text overlaid on detected tags.
  * `task2_frame`: Displays the final Augmented Reality output with your template perfectly mapped onto the tags.
* Press the **`q`** key while focused on the video windows to quit the application safely.

---

## 📂 Code Structure

* **`main.py`**: Handles video input/output, parses command-line arguments, and manages the execution loop for rendering the frames.
* **`utils.py`**: The powerhouse of the repository. Contains all the heavy mathematical lifting and custom algorithm implementations, including:
  * `extract_boundary()`: Morphological edge detection.
  * `get_connected_components()`: Stack-based component grouping.
  * `get_quad_corners()`: Geometric corner extraction.
  * `compute_homography()` & `apply_inverse_homography()`: Custom projective geometry logic.
  * `decode_tag()`: Parses the binary 8x8 grid of the AR tag.
  * `warp_overlay()`: Renders the Augmented Reality template.

---
*Note: This project was developed as an exercise in understanding the core mathematical fundamentals of Augmented Reality pipelines without relying on "black box" standard library functions.*

## 🎥 Output Video Demonstration

You can view the output videos demonstrating the AR tag detection and rendering pipeline in action here:

[**View Output Video Folder on Google Drive**](https://drive.google.com/drive/folders/1KGwmqdVYgbvTd4eskOS7nA9Krc8kBRTM)
