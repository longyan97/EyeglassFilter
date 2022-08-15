import argparse
import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat
import dlib
import numpy as np
import time


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords




parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, default=5, help="Strength of Gaussian blur specified by sigma (default: 5)")
parser.add_argument("--filter", choices=["none", "blur"], default="blur")
parser.add_argument("--fps", action="store_true", help="output fps every second")
args = parser.parse_args()

# Set up webcam capture.
vc = cv2.VideoCapture(1)

if not vc.isOpened():
    raise RuntimeError('Could not open video source')


# Query final capture device values (may be different from preferred settings).
width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = vc.get(cv2.CAP_PROP_FPS)
print(f'Webcam capture started ({width}x{height} @ {fps_in}fps)')

fps_out = 30
resample_factor = .25

# External resources of facial landmark detector used for locating glass areas
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('utils/shape_predictor_68_face_landmarks.dat')


with pyvirtualcam.Camera(width, height, fps_out, fmt=PixelFormat.BGR, print_fps=args.fps) as cam:
    print(f'Virtual cam started: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')

    while True:
        # Read frame from webcam.
        ret, image = vc.read()
        if not ret:
            raise RuntimeError('Error fetching frame')

        if args.filter == "none":
            pass

        else: 
            # Get the region of interest, i.e., the glass areas
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (0, 0), fx=resample_factor, fy=resample_factor)
            rects, scores, idx = detector.run(gray, 0)
            if len(rects) >= 1:
                rect_widths = [x.width() for x in rects]
                rect = rects[rect_widths.index(max(rect_widths))]
                shape = predictor(gray, rect)
                face_image = dlib.get_face_chip(image, shape, size=600)
                shape_np = np.zeros((68, 2), dtype="int")
                for i in range(0, 68):
                    shape_np[i] = (int(shape.part(i).x / resample_factor), int(shape.part(i).y / resample_factor))
                key_points = shape_np[[1, 2, 16, 17]]

                newrect = [shape_np[19], shape_np[24], shape_np[1], shape_np[30], shape_np[15]]
                
                glass_rect = cv2.minAreaRect(key_points)
                newglass = (glass_rect[0], glass_rect[1], 88)
                glass_corners = cv2.boxPoints(newglass).astype(int)

                # Get current frame and output multiple frames but return current frame
                bottomright = (0,0)
                topleft = (0,0)
                for i, (x, y) in enumerate(glass_corners):
                    if i == 2:
                        bottomright = (x,y)
                    if i == 0:
                        topleft = (x,y)
                imgcpy = image.copy()
                roi = cv2.rectangle(imgcpy, topleft, bottomright, (0,255,0))
                np_roi = image[topleft[1]:bottomright[1], topleft[0]:bottomright[0]]

                # blur lens area using Gaussian filters
                if args.filter == "blur":
                    blur = cv2.GaussianBlur(src=np_roi, ksize=(0,0), sigmaX=args.sigma) 
                    image[topleft[1]:bottomright[1], topleft[0]:bottomright[0]] = blur


        # Send to virtual cam.
        cam.send(image)
        # Wait until it's time for the next frame.
        cam.sleep_until_next_frame()


