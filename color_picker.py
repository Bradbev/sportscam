import cv2
import numpy as np
import argparse
import sys

# Global variable to store the current frame
current_frame = None

def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse clicks on the video frame.
    When the left mouse button is clicked, it samples the pixel color
    and prints RGB, HSV, and a suggested HSV range to the console.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_frame is not None:
            # Get the BGR color of the clicked pixel
            bgr_color = current_frame[y, x]
            
            # Convert BGR to RGB
            rgb_color = bgr_color[::-1]
            
            # Convert BGR to HSV
            # Create a 1x1 pixel image for conversion
            hsv_color = (cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]).astype(int)

            print(f"\nClicked at (x={x}, y={y})")
            print(f"RGB: {rgb_color}")
            print(f"HSV: {hsv_color}")

            # Define a tolerance for creating an HSV range
            h_tolerance = 10
            sv_tolerance = 50

            # Calculate lower and upper HSV bounds for cv2.inRange
            lower_h = max(0, hsv_color[0] - h_tolerance)
            upper_h = min(179, hsv_color[0] + h_tolerance)
            
            lower_s = max(0, hsv_color[1] - sv_tolerance)
            upper_s = min(255, hsv_color[1] + sv_tolerance)
            
            lower_v = max(0, hsv_color[2] - sv_tolerance)
            upper_v = min(255, hsv_color[2] + sv_tolerance)

            lower_bound = np.array([lower_h, lower_s, lower_v])
            upper_bound = np.array([upper_h, upper_s, upper_v])

            print(f"lower = np.array({lower_bound})")
            print(f"upper = np.array({upper_bound})")

def main():
    """
    Main function to parse arguments, open video, and handle playback.
    """
    global current_frame

    parser = argparse.ArgumentParser(description="A script to pick colors from a video file.")
    parser.add_argument("video_file", help="The path to the video file to process.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_file}")
        sys.exit(1)

    window_name = 'Color Picker - Press SPACE to pause, ESC to exit'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            current_frame = frame

        cv2.imshow(window_name, current_frame)

        key = cv2.waitKey(25) & 0xFF

        if key == ord(' '):  # Space bar to pause/resume
            paused = not paused
        elif key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main().astype(int)