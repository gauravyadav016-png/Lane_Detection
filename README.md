# Lane Detection in Videos

This project provides a web application for detecting lanes in videos using OpenCV and Streamlit. The application allows users to upload video files, processes the video to detect lanes, and then enables users to download the processed video.

## Features

- Detects lanes in video frames.
- Uses OpenCV for image processing and lane detection.
- Simple web interface built with Streamlit.
- Supports video file formats such as MP4.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gauravyadav016-png/Lane_Detection.git
   cd lane-detection


2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload your video file using the provided interface.

4. Click the "Process Video" button to detect lanes in the video and then download the processed video.

## Code Overview

- `app.py`: The main application file that sets up the Streamlit interface and handles file uploads and video processing.
- `LaneDetection`: A class that provides methods to process video frames and detect lanes.
- `LaneLineHistory`: A class to keep track of lane lines history for more stable lane detection.

## Dependencies

- `streamlit`: For building the web application interface.
- `numpy`: For numerical operations.
- `opencv-python`: For image processing and lane detection.
- `moviepy`: For handling video file operations.
- `tempfile`: For handling temporary file storage.


## Input Sample
https://github.com/gauravyadav016-png/Lane_Detection/assets/77223217/b86a95f3-3fd4-479c-81c9-8f72324f35f3


## Output Sample
https://github.com/gauravyadav016-png/Lane_Detection/assets/77223217/95c00d01-db03-4af7-ab4a-830835ff6c40

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
