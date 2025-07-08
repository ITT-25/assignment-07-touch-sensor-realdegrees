# Camera-Based Touch Sensor

Instructions on how to launch and configure the application are located in [README.md](README.md).  

![](docs/example_task_1.gif)

## 1. Design Decisions

### 1.1 "Hardware"

Since the FoV of the camera was unable to capture the entire surface of the plexiglass, I decided to draw a rectangle on a sheet of paper doubling as a diffuser.
This also makes it a lot easier to assemble the setup since you only have to put the camera in the box and can then place the plexiglass with the paper on top of it and align the rectangle with the camera view when launching the program.  
**The physical rectangle is not used for any detection or calibration steps, it is purely for visual reference for the user!**

### 1.2 Calibration

For the calibration step at the start I decided to wait for a few frames before capturing the average brightness because the first few frames somehow started extremely dark consistently. After these frames are skipped the app takes the average brightness of a few frames and maps this brightness to a value between 30 and 70 (arbitrary based on different lighting conditions I tested in) to be used as a threshold for the binary thresholding applied in later steps.

### 1.3 Image Preprocessing and Fingertip Detection

After playing around with a few basic preprocessing techniques like blurring, contrast enhancement and thresholding, I found that finding contours was still pretty rough and I often had issues with the palm or finger shadows being picked up. Despite that the image is still preprocessed with a Gaussian blur to reduce noise and a grayscale conversion to simplify the image.

#### 1.3.1 CLAHE

As a solution to this I found CLAHE (Contrast Limited Adaptive Histogram Equalization) to be very effective. It enhances the contrast of the image in a way that is adaptive to local regions, which helps in distinguishing the fingertips from the background and shadows. In combination with closing and opening the image to remove speckles and merge areas with holes I was then able to cleanly seperate all contours based on their size and only pass the ones that are likely to be fingertips.

#### 1.3.2 History

On top of that I implemented a _history_ system that keeps track of a fingertip based on position delta. This way I could filter out any noise that might occur if a fraction of the palm or a shadow was picked up as a fingertip for a few frames.

#### 1.3.3 Circularity Score

To further improve the classification of fingertips I used the enclosing circle of the contour (after slightly smoothing the countour) to check the circularity of the contour. The circularity describes how close the contour is to a perfect circle. For one this helped me filter out contours that are not fingertips but it also gave me a score that I could use to determine if a finger is pressed down or not since the fingertip has a rather round shape compared to the oval shape when pressing a little harder. 

### 1.4 Event Detection

For movement events I simply used the center of the enclosing circle described above. I didn't feel the need to apply any smoothing to the history of the fingertip since it is supposed to work 1:1 in real-time and the jitter was not bad at all. For tap detection I tried to detect patterns in the history of a fingertip, e.g. a pattern of `high circularity -> low circularity -> high circularity` matched with a pattern of `low radius -> high radius -> low radius` would indicate a tap. However, this was not very reliable and resulted in a lot of false positives. Instead I decided to implement a simple cooldown for taps and check if the average radius and circularity of the last few history entries matched certain thresholds. Low circularity and high radius indicates a tap so the system continously registers taps (with a cooldown) as long as a finger is pressed down.

### 1.5 Data Broadcasting

The data is broadcasted using `DIPPID` with the required structure and works fine in combination with [fitts_law.py](fitts_law.py).
The y coordinates are manually flipped before broadcasting to match the pyglet coordinate system.  
The events that are broadcasted are also printed on the preview window.

### 1.6 Aspect Ratio

I found that a 16:9 aspect ratio makes the most sense since ultimately this app is supposed to map the movement of the finger 1:1 to a screen and most screens are 16:9 and the camera is as well. In order to properly match up the coordinates I had to slightly modify the code in [fitts_law.py](fitts_law.py) to match the camera's aspect ratio.

## 2. Building Process

I closed the box on the bottom to provide a stable base for the camera. For consistency when reassembling I marked the position and rotation of the camera on the inside of the box. The cable was simply routed through one of the 4 corners but doesn't matter as long as the camera is in the designated position. I flipped the cardboard that makes up the top of the box to the inside to provide more stability. The plexiglass was then put on top with a sheet of paper taped to it to act as a diffuser. As described in the first section I then drew a rectangle on the paper to indicate the touch area that the camera is able to physically see. During development I also taped down the camera however this is not necessary as long as nobody yanks on the cable.

**Note: Initially I folded the top "flaps" of the box upwards so that the camera's FoV would be able to capture the entire surface but decided against it because the entire thing was unstable and not really practical/convenient to use. In my opinion drawing the visible area on the paper is a lot more practical and easier to assemble.**

## 3. Usage Guide

Instructions on how to launch and configure the application are located in [README.md](README.md).

Assemble the box so that all marked areas inside the box line up. Put the camera at the designated spot and route the cable through one of the corners. Put the plexiglass on top of the box with the paper taped to it and align the rectangle with the camera view. Launch the application and let it calibrate (If fingertip detection does not work as seen in the example gif, adjust the lighting or trick the app by putting your hand on the touch area during calibration to achieve different thresholds).

Once the app is launched and calibrated you will see the preview window. Put a fingertip on the touch area and it should show the detected contour, enclosing circle and center point around your fingertip. A small debug view in the bottom right also shows the preprocessed image at the step before searching for contours.