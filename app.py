import streamlit as st
import numpy as np
import cv2
from moviepy import editor
import tempfile

# Class to keep track of lane lines history
class LaneLineHistory:
    def __init__(self, max_history=10):
        self.left_lines = []
        self.right_lines = []
        self.max_history = max_history

    def add_lines(self, left_line, right_line):
        if left_line is not None:
            self.left_lines.append(left_line)
            if len(self.left_lines) > self.max_history:
                self.left_lines.pop(0)
        if right_line is not None:
            self.right_lines.append(right_line)
            if len(self.right_lines) > self.max_history:
                self.right_lines.pop(0)

    def get_average_lines(self):
        left_line_avg = np.mean(self.left_lines, axis=0) if self.left_lines else None
        right_line_avg = np.mean(self.right_lines, axis=0) if self.right_lines else None
        return left_line_avg, right_line_avg

# Function to select the region of interest in the image
def region_selection(image):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.45, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.55, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Function to perform Hough Line Transform
def hough_transform(image):
    rho = 1
    theta = np.pi / 180
    threshold = 15
    minLineLength = 40
    maxLineGap = 20
    lines = cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)
    return lines if lines is not None else []

# Function to average slope and intercept of lines
def average_slope_intercept(lines, img_width):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue  # Ignore vertical lines
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if length < 30:  # Discard short lines
                continue
            if slope < 0 and x1 < img_width / 2 and x2 < img_width / 2:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            elif slope > 0 and x1 > img_width / 2 and x2 > img_width / 2:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane

# Function to convert slope and intercept to pixel points
def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    if slope == 0:
        x1 = x2 = int(intercept)
    else:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

# Initialize the LaneLineHistory object
lane_history = LaneLineHistory()

# Function to generate lane lines from the detected lines
def lane_lines(image, lines):
    img_width = image.shape[1]
    left_lane, right_lane = average_slope_intercept(lines, img_width)
    lane_history.add_lines(left_lane, right_lane)
    left_lane_avg, right_lane_avg = lane_history.get_average_lines()
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    left_line = pixel_points(y1, y2, left_lane_avg)
    right_line = pixel_points(y1, y2, right_lane_avg)
    return left_line, right_line

# Function to draw lane lines on the image
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

# Function to process each video frame
def frame_processor(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.equalizeHist(grayscale)  # Histogram Equalization
    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    low_t = 50
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)
    region = region_selection(edges)
    hough = hough_transform(region)
    result = draw_lane_lines(image, lane_lines(image, hough))
    return result

# Function to process the video
def process_video(input_path, output_path):
    input_video = editor.VideoFileClip(input_path, audio=False)
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(output_path, audio=False)

# Streamlit app to upload and process video
def lane_detection_app():
    st.title("Lane Detection in Videos")
    st.write("Upload a video to detect lanes and then download the processed video.")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        st.write("Video uploaded successfully. Click 'Process Video' to start lane detection.")
        
        if st.button("Process Video"):
            output_path = "output.mp4"
            process_video(tfile.name, output_path)
            
            with open(output_path, 'rb') as f:
                st.download_button('Download Processed Video', f, file_name='output.mp4')

if __name__ == "__main__":
    lane_detection_app()
