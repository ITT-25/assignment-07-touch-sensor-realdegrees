import time
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

def preprocess_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
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
    clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)
    
    # Use global thresholding but with equalized image
    _, thresh = cv2.threshold(equalized, 60, 255, cv2.THRESH_BINARY)

    # keep only continous black areas over a certain size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return frame, thresh, width, height

def process_contours(thresh: np.ndarray, width: int, height: int) -> list:
    # find contours in thresholded image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours by size
    area_scale_factor = (width * height) / (1600 * 900)
    lower_bound = int(3500 * area_scale_factor)
    upper_bound = int(20000 * area_scale_factor)
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

                if solidity > 0.97 and circularity > 0.8:
                    # Find the minimum enclosing circle
                    (center_point, radius) = cv2.minEnclosingCircle(c)
                    # Convert to integer
                    radius = int(radius)
                    center = tuple(map(int, center_point))
                    fingertips.append(FingertipData(center, radius, c, time.time()))
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
def main(video_id: int, debug: bool) -> None:
    global width, height, dt
    
    print(f"Starting webcam capture with camera ID: {video_id}")
    cap = cv2.VideoCapture(video_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fingertip_tracker = FingertipHistory(1, dt)
    fingertip_event_detector = FingertipEventDetector(dt)

    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {video_id}")
        return

    def capture_loop(dt: float) -> None:
        ret, frame = cap.read()
        if not ret:
            return
        else:
            # Flip frame
            frame = cv2.flip(frame, 1)

        # Process the frame, find contours and fingertips, handle mouse
        frame, thresh, width, height = preprocess_frame(frame)
        contours = process_contours(thresh, width, height)            
        fingertips = detect_fingertips(contours)
        fingertip_tracker.update(fingertips)
        events = fingertip_event_detector.detect_events(fingertip_tracker.get_stable_fingertips())
        
        if events:
            print(f"Detected events: {events}")
        
        # Get stable fingertips for drawing
        stable_histories = fingertip_tracker.get_stable_fingertips()
        
        # Draw the stable fingertips on the original frame
        for history in stable_histories:
            if len(history) > 0:
                latest_tip = history[-1]
                center = latest_tip.center
                radius = latest_tip.radius
                
                # Draw current fingertip
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.drawContours(frame, [latest_tip.contour], -1, (0, 0, 255), 2)
        
        # Display event information on frame
        if events:
            y_offset = 30
            for event in events:
                cv2.putText(frame, f"Event: {event}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_offset += 25

        # Insert the thresholded image at bot right PiP
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thresh = cv2.resize(thresh, (160, 90))
        frame[height-90:height, width-160:width] = thresh[0:90, 0:160]
        
        cv2.imshow("Detected Box", frame)

    while True:
        capture_loop(dt)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # Q or Escape
            break
        time.sleep(dt)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
