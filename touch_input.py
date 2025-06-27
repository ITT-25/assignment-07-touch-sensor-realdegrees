import time
import cv2
import click

@click.command()
@click.option(
    "--video-id",
    "-c",
    default=0,
    help="ID of the webcam you want to use",
    type=int,
    show_default=True,
)
@click.option(
    "--cam-width", "-w", default=640, help="Width of the webcam frame", type=int, show_default=True
)
@click.option(
    "--cam-height",
    "-h",
    default=480,
    help="Height of the webcam frame",
    type=int,
    show_default=True,
)
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode")
def main(video_id: int, cam_width: int, cam_height: int, debug: bool) -> None:
    print(f"Starting webcam capture with camera ID: {video_id}")
    cap = cv2.VideoCapture(video_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

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

        h, w = frame.shape[:2]
        cv2.imshow("Webcam Capture", frame)

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
