import streamlit as st
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import tempfile

class LaneDetection:
    """
    Class to handle lane detection operations.
    """
    def __init__(self, max_history=10):
        self.lane_history = LaneLineHistory(max_history)

    def process_frame(self, image):
        """
        Process a single frame to detect lanes.
        """
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.equalizeHist(grayscale)  # Histogram Equalization
        blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        region = self.region_selection(edges)
        lines = self.hough_transform(region)
        result = self.draw_lane_lines(image, self.lane_lines(image, lines))
        return result

    def region_selection(self, image):
        """
        Define the region of interest in the image for lane detection.
        """
        mask = np.zeros_like(image)
        ignore_mask_color = 255
        rows, cols = image.shape[:2]
        vertices = np.array([[
            [cols * 0.1, rows * 0.95],
            [cols * 0.45, rows * 0.6],
            [cols * 0.55, rows * 0.6],
            [cols * 0.9, rows * 0.95]
        ]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        return cv2.bitwise_and(image, mask)

    def hough_transform(self, image):
        """
        Apply Hough Line Transform to detect lines in the image.
        """
        return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=15,
                               minLineLength=40, maxLineGap=20)

    def lane_lines(self, image, lines):
        """
        Generate lane lines from detected lines.
        """
        img_width = image.shape[1]
        left_lane, right_lane = self.average_slope_intercept(lines, img_width)
        self.lane_history.add_lines(left_lane, right_lane)
        left_lane_avg, right_lane_avg = self.lane_history.get_average_lines()
        y1 = image.shape[0]
        y2 = int(y1 * 0.6)
        left_line = self.pixel_points(y1, y2, left_lane_avg)
        right_line = self.pixel_points(y1, y2, right_lane_avg)
        return left_line, right_line

    def draw_lane_lines(self, image, lines, color=[255, 0, 0], thickness=12):
        """
        Draw lane lines on the image.
        """
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line, color, thickness)
        return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

    def average_slope_intercept(self, lines, img_width):
        """
        Calculate the average slope and intercept of the lines.
        """
        left_lines, left_weights, right_lines, right_weights = [], [], [], []
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

    def pixel_points(self, y1, y2, line):
        """
        Convert slope and intercept to pixel points.
        """
        if line is None:
            return None
        slope, intercept = line
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return ((x1, y1), (x2, y2))

class LaneLineHistory:
    """
    Class to keep track of lane lines history.
    """
    def __init__(self, max_history=10):
        self.left_lines = []
        self.right_lines = []
        self.max_history = max_history

    def add_lines(self, left_line, right_line):
        """
        Add new lines to the history.
        """
        if left_line is not None:
            self.left_lines.append(left_line)
            if len(self.left_lines) > self.max_history:
                self.left_lines.pop(0)
        if right_line is not None:
            self.right_lines.append(right_line)
            if len(self.right_lines) > self.max_history:
                self.right_lines.pop(0)

    def get_average_lines(self):
        """
        Calculate average lines from the history.
        """
        left_line_avg = np.mean(self.left_lines, axis=0) if self.left_lines else None
        right_line_avg = np.mean(self.right_lines, axis=0) if self.right_lines else None
        return left_line_avg, right_line_avg

def process_video(input_path, output_path):
    """
    Process the video to detect lanes.
    """
    lane_detector = LaneDetection()
    input_video = VideoFileClip(input_path, audio=False)
    processed = input_video.fl_image(lane_detector.process_frame)
    processed.write_videofile(output_path, audio=False)

def lane_detection_app():
    """
    Streamlit app to upload and process video for lane detection.
    """
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

