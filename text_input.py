import time
import click
import cv2
import numpy as np
from DIPPID import SensorUDP
from src.recognizer import Recognizer
    
@click.command()
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode")
@click.option(
    "--port",
    "-p",
    default=5700,
    help="Port to broadcast events to",
    type=int,
    show_default=True,
)
@click.option(
    "--confidence-threshold",
    "-c",
    default=0.9,
    help="Confidence threshold for auto-typing (0.0-1.0)",
    type=float,
    show_default=True,
)
@click.option(
    "--detection-timer",
    "-t",
    default=1.5,
    help="The amount of time in seconds to wait after the last input before making a prediction (seconds)",
    type=float,
    show_default=True,
)
def main(debug: bool, port: int, confidence_threshold: float, detection_timer: float) -> None:
    sensor = SensorUDP(port)
    recognizer = Recognizer(confidence_threshold=confidence_threshold, detection_timer=detection_timer)

    def handle_movement(data):
        recognizer.on_point((data['x'], 450 - data['y'])) # Invert y because we are receiving pyglet adjusted coordinates

    sensor.register_callback('movement', handle_movement)

    dt = 1/60  # Target frame rate

    def loop(dt: float) -> None:
        # Draw the points on a preview image
        img = np.zeros((450, 800, 3), dtype=np.uint8)
        for point in recognizer.points:
            cv2.circle(img, point, 5, (0, 255, 0), -1)
            
        # Create visualization image
        if recognizer.points:
            # Process live rasterization
            raster_img = recognizer._preprocess_and_rasterize(recognizer.points)
            raster_img = np.transpose(raster_img)
            
            # Resize for display
            raster_size = 100 
            raster_img_resized = cv2.resize(raster_img, (raster_size, raster_size))
            raster_img_color = cv2.cvtColor(raster_img_resized, cv2.COLOR_GRAY2BGR) * 255
            
            # Position for live raster
            live_x = 10
            live_y = 80 + raster_size + 30
            
            # Add live raster with border
            cv2.rectangle(img, (live_x-2, live_y-2), (live_x+raster_size+2, live_y+raster_size+2), (0, 255, 0), 2)
            img[live_y:live_y+raster_size, live_x:live_x+raster_size] = raster_img_color
            cv2.putText(img, "Live Raster:", (live_x, live_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # Show waiting message
            cv2.putText(img, "Waiting for input from DIPPID...", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            
        # Show prediction if available
        if recognizer.prediction.char:
            # Display prediction text
            cv2.putText(img, f"Predicted: {recognizer.prediction.char} ({recognizer.prediction.confidence:.2f})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "Rasterized Image:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Prepare final raster display
            raster_size = 100
            final_x = 10
            final_y = 80
            
            # Get rasterized image and convert to color
            raster_img = cv2.resize(recognizer.prediction.rasterized_image[0, ..., 0], (raster_size, raster_size))
            raster_img_color = cv2.cvtColor(raster_img, cv2.COLOR_GRAY2BGR) * 255
            
            # Set border color based on confidence
            border_color = (0, 255, 0) if recognizer.prediction.confidence > recognizer.confidence_threshold else (0, 0, 255)
            
            cv2.rectangle(img, (final_x-2, final_y-2), (final_x+raster_size+2, final_y+raster_size+2), border_color, 2)
            img[final_y:final_y+raster_size, final_x:final_x+raster_size] = raster_img_color

        # Display the visualization
        if debug:
            cv2.imshow("Preview", img)

    try:
        while True:
            loop(dt)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or Escape
                break
            time.sleep(dt)
    finally:
        # Clean up
        sensor.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
