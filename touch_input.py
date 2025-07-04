import time
from typing import Tuple
import cv2
import click
import numpy as np

width = 800
height = 450

def preprocess_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    height, width = frame.shape[:2]
    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cut 5% from all sides
    cut = int(0.05 * height)
    gray = gray[cut:height-cut, cut:width-cut]
    frame = frame[cut:height-cut, cut:width-cut]
    height, width = gray.shape[:2]

    # Apply Gaussian blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Equalize the histogram to improve contrast
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)
    
    # Use global thresholding but with equalized image
    _, thresh = cv2.threshold(equalized, 52, 255, cv2.THRESH_BINARY)

    # keep only continous black areas over a certain size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2) # remove speckles (for fingertip)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4) # close holes (for hand)
    
    return frame, thresh, width, height

def process_contours(thresh: np.ndarray, width: int, height: int) -> list:
    # find contours in thresholded image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours by size
    area_scale_factor = (width * height) / (1600 * 900)
    lower_bound = int(3000 * area_scale_factor)
    upper_bound = int(17000 * area_scale_factor)
    contours = [c for c in contours if lower_bound <
                cv2.contourArea(c) < upper_bound]

    # smooth contours
    return [cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True) for c in contours]

def detect_fingertips(contours):
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
                    # replace the counter with a circle at the centroid of the ellipse and an equal area to the countour
                    radius = int((cv2.contourArea(c) / 3.14) ** 0.5)
                    center = tuple(map(int, cv2.minEnclosingCircle(c)[0]))
                    fingertips.append((center, radius, c))
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
    global width, height
    
    print(f"Starting webcam capture with camera ID: {video_id}")
    cap = cv2.VideoCapture(video_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        # TODO: convert fingertips to mouse events and broadacast them with dippid
        
        # Draw the filtered contours on the original frame
        for center, radius, contour in fingertips:
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

        # Insert the thresholded image at bot right PiP
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thresh = cv2.resize(thresh, (160, 90))
        frame[height-90:height, width-160:width] = thresh[0:90, 0:160]
        
        cv2.imshow("Detected Box", frame)

    dt = 1 / 60.0
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
