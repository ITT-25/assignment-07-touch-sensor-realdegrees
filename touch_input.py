import time
import json
import socket
from typing import Tuple, List
import cv2
import click
import numpy as np
from src.fingertip_data import FingertipData
from src.fingertip_events import FingertipEventDetector
from src.fingertip_history import FingertipHistory

width = 800
height = 450
dt = 1 / 60.0

class TouchEventBroadcaster:
    """Broadcast touch events via UDP"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5700):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
    def broadcast_event(self, event: dict):
        """Broadcast an event as JSON via UDP"""
        try:
            message = json.dumps(event)
            self.sock.sendto(message.encode(), (self.host, self.port))
        except Exception as e:
            print(f"Error broadcasting event: {e}")
            
    def close(self):
        """Close the socket"""
        self.sock.close()

def preprocess_frame(frame: np.ndarray, threshold_value: int = 60) -> Tuple[np.ndarray, np.ndarray, int, int]:
    height, width = frame.shape[:2]
    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cut 5% from all sides
    cut_h = int(0.05 * height)
    cut_w = int(0.05 * width)
    gray = gray[cut_h:height - cut_h, cut_w:width - cut_w]
    frame = frame[cut_h:height - cut_h, cut_w:width - cut_w]
    height, width = gray.shape[:2]

    # Apply Gaussian blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Equalize the histogram to improve contrast
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)
    
    # Use global thresholding but with calibrated threshold value
    _, thresh = cv2.threshold(equalized, threshold_value, 255, cv2.THRESH_BINARY)

    # keep only continous black areas over a certain size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
    
    
    return frame, thresh, width, height

def process_contours(thresh: np.ndarray, width: int, height: int) -> list:
    # find contours in thresholded image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours by size
    area_scale_factor = (width * height) / (1600 * 900)
    lower_bound = int(3000 * area_scale_factor)
    upper_bound = int(35000 * area_scale_factor)
    contours = [c for c in contours if lower_bound <
                cv2.contourArea(c) < upper_bound]

    # smooth contours
    return [cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True) for c in contours]

def detect_fingertips(contours: List[np.ndarray]) -> List[FingertipData]:
        fingertips = []
        for c in contours:
            # Fit an ellipse to the contour
            if len(c) >= 5:  # Need at least 5 points to fit an ellipse
                # Calc circularity
                area = cv2.contourArea(c)
                perimeter = cv2.arcLength(c, True)
                circularity = 4 * np.pi * (area / (perimeter ** 2))

                # Calc solidity
                hull = cv2.convexHull(c)
                area = cv2.contourArea(c)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area != 0 else 0

                if solidity > 0.97 and circularity > 0.5:
                    # Find the minimum enclosing circle
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    radius = int(radius)
                    fingertips.append(FingertipData((int(x), int(y)), radius, circularity, c, time.time()))
                    
        return fingertips
    

@click.command()
@click.option(
    "--video-id",
    "-c",
    default=0,
    help="ID of the webcam you want to use",
    type=int,
    show_default=True,
)
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode")
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to broadcast events to",
    type=str,
    show_default=True,
)
@click.option(
    "--port",
    "-p",
    default=5700,
    help="Port to broadcast events to",
    type=int,
    show_default=True,
)
@click.option(
    "--calibration-frames",
    default=5,
    help="Number of frames to use for brightness calibration",
    type=int,
    show_default=True,
)
def main(video_id: int, debug: bool, host: str, port: int, calibration_frames: int) -> None:
    global width, height, dt
    
    print(f"Starting webcam capture with camera ID: {video_id}")
    print(f"Broadcasting events to {host}:{port}")
    
    cap = cv2.VideoCapture(video_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {video_id}")
        return

    # Perform brightness calibration
    threshold_value = calibrate_threshold(cap, calibration_frames)
    
    fingertip_tracker = FingertipHistory(0.8, dt)
    fingertip_event_detector = FingertipEventDetector(dt)
    event_broadcaster = TouchEventBroadcaster(host, port)

    def capture_loop(dt: float) -> None:
        ret, frame = cap.read()
        if not ret:
            return
        else:
            # Flip vertically for correct orientation
            frame = cv2.flip(frame, 0)

        # Process the frame, find contours and fingertips, handle mouse
        frame, thresh, width, height = preprocess_frame(frame, threshold_value)
        contours = process_contours(thresh, width, height)            
        fingertips = detect_fingertips(contours)
        fingertip_tracker.update(fingertips)
        events = fingertip_event_detector.detect_events(fingertip_tracker.get_stable_fingertips())
        
        # Invert all "movement": "y" values in events to match the pyglet coordinate system in the target app
        if "movement" in events:
            events["movement"]["y"] = height - events["movement"]["y"]

        event_broadcaster.broadcast_event(events)
    
        
        # Get stable fingertips for drawing
        stable_histories = fingertip_tracker.get_stable_fingertips()
        
        # Draw the stable fingertips on the original frame
        for history in stable_histories:
            if len(history) > 0:
                latest_tip = history[-1]
                center = latest_tip.center
                radius = latest_tip.radius
                
                # Draw current fingertip
                cv2.circle(frame, center, radius, (0, 255, 0), 2) # enclosing circle
                cv2.drawContours(frame, [latest_tip.contour], -1, (0, 0, 255), 2) # contour
                cv2.circle(frame, center, 15 if events.get("tap") else 4, (255, 0, 0), -1)  # center point

        # Display event information on frame
        if events:
            y_offset = 30
            for event_type, event_data in events.items():
                event_text = f"{event_type}: {event_data}"
                cv2.putText(frame, event_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
        
        # Display threshold value in debug mode
        cv2.putText(frame, f"Threshold: {threshold_value}", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Insert the thresholded image at bot right PiP
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thresh = cv2.resize(thresh, (160, 90))
        frame[height-90:height, width-160:width] = thresh[0:90, 0:160]
        
        cv2.imshow("Detected Box", frame)

    try:
        while True:
            capture_loop(dt)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or Escape
                break
            time.sleep(dt)
    finally:
        # Clean up
        event_broadcaster.close()
        cap.release()
        cv2.destroyAllWindows()


def calibrate_threshold(cap: cv2.VideoCapture, num_frames: int) -> int:
    """Calibrate the threshold value based on the average brightness of the scene."""
    
    print(f"Starting calibration... analyzing {num_frames} frames")
    brightness_values = []
    
    # Apply same preprocessing as in preprocess_frame before calculating threshold
    # Skip the first 20 frames to allow camera to initialize and adjust
    print("Waiting for camera to initialize...")
    for _ in range(20):
        ret, _ = cap.read()
        if not ret:
            print("Warning: Could not read frame during camera initialization")
        time.sleep(0.1)
    print("Camera initialized, starting calibration...")
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i+1} during calibration")
            continue
            
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cut_h = int(0.05 * height)
        cut_w = int(0.05 * width)
        gray = gray[cut_h:height - cut_h, cut_w:width - cut_w]
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        avg_brightness = np.mean(blurred)
        brightness_values.append(avg_brightness)
        
        print(f"Frame {i+1}: brightness = {avg_brightness:.1f}")
        
        time.sleep(0.1)
    
    if not brightness_values:
        print("Warning: No frames captured during calibration, using default threshold")
        return 55  # Default middle value
    
    # Calculate average brightness across all frames
    avg_brightness = np.mean(brightness_values)
    print(f"Average brightness: {avg_brightness:.1f}")

    # Map the average brightness (0-255) to the range below
    lower_bound = 35
    upper_bound = 70
    threshold = lower_bound + (avg_brightness / 255) * (upper_bound - lower_bound)
    threshold = int(threshold)
    
    # Ensure threshold is within bounds
    threshold = max(lower_bound, min(upper_bound, threshold))

    print(f"Calculated threshold: {threshold}")
    print("Calibration complete!")
    return threshold


if __name__ == "__main__":
    main()
