# if you need send the points please from laser import trajectory

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time


class Laser1:
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64,
                    help="max buffer size")
    args = vars(ap.parse_args())
    trajectory = []
    # define the lower and upper boundaries of the colors
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    pts = deque(maxlen=args["buffer"])

    def run_laser(self,frame):
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        # Perform morphological operations to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        hsv = cv2.cvtColor(closed, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                # Append the center coordinates to the trajectory list
                self.trajectory.append(center)
                # Print the current center coordinates
            # print(center)
            # update the points queue
            self.pts.appendleft(center)
            # trajectory = np.array(trajectory)
            # print(trajectory)
            # loop over the set of tracked points
        for i in range(1, len(self.pts)):
            # if either of the tracked points are None, ignore
            # them
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue

                # otherwise, compute the thickness of the line and
                # draw the connecting lines
            thickness = int(np.sqrt(self.args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)
            return frame

    #
    # # if a video path was not supplied, grab the reference
    # # to the webcam
    # if not args.get("video", False):
    #     vs = VideoStream(src=0).start()
    #
    # # otherwise, grab a reference to the video file
    # else:
    #     vs = cv2.VideoCapture(args["video"])
    #
    # # allow the camera or video file to warm up
    # time.sleep(2.0)
    #
    # # Create an empty list to store the trajectory points
    # trajectory = []
    #
    # # keep looping
    # while True:
    #     # grab the current frame
    #     frame = vs.read()
    #
    #     # handle the frame from VideoCapture or VideoStream
    #     frame = frame[1] if args.get("video", False) else frame
    #
    #     # if we are viewing a video and we did not grab a frame,
    #     # then we have reached the end of the video
    #     if frame is None:
    #         break
    #
    #     # resize the frame, blur it, and convert it to the HSV
    #     # color space
    #     frame = imutils.resize(frame, width=600)
    #     blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    #
    #     # Perform morphological operations to remove small noise
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #     opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    #     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    #
    #     hsv = cv2.cvtColor(closed, cv2.COLOR_BGR2HSV)
    #
    #     # construct a mask for the color "green", then perform
    #     # a series of dilations and erosions to remove any small
    #     # blobs left in the mask
    #     mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    #     mask = cv2.erode(mask, None, iterations=2)
    #     mask = cv2.dilate(mask, None, iterations=2)
    #
    #     # find contours in the mask and initialize the current
    #     # (x, y) center of the ball
    #     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    #                             cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = imutils.grab_contours(cnts)
    #     center = None
    #
    #     # only proceed if at least one contour was found
    #     if len(cnts) > 0:
    #         # find the largest contour in the mask, then use
    #         # it to compute the minimum enclosing
    #         c = max(cnts, key=cv2.contourArea)
    #         ((x, y), radius) = cv2.minEnclosingCircle(c)
    #         M = cv2.moments(c)
    #         center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #
    #         # only proceed if the radius meets a minimum size
    #         if radius > 10:
    #             # draw the circle and centroid on the frame,
    #             # then update the list of tracked points
    #             cv2.circle(frame, (int(x), int(y)), int(radius),
    #                        (0, 255, 255), 2)
    #             cv2.circle(frame, center, 5, (0, 0, 255), -1)
    #             # Append the center coordinates to the trajectory list
    #             trajectory.append(center)
    #             # Print the current center coordinates
    #         # print(center)
    #         # update the points queue
    #         pts.appendleft(center)
    #         # trajectory = np.array(trajectory)
    #         # print(trajectory)
    #         # loop over the set of tracked points
    #     for i in range(1, len(pts)):
    #         # if either of the tracked points are None, ignore
    #         # them
    #         if pts[i - 1] is None or pts[i] is None:
    #             continue
    #
    #             # otherwise, compute the thickness of the line and
    #             # draw the connecting lines
    #         thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
    #         cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    #
    #         # show the frame to our screen
    #     cv2.imshow("Frame", frame)
    #     key = cv2.waitKey(1) & 0xFF
    #
    #     # if the 'q' key is pressed, stop the loop
    #     if key == ord("q"):
    #         break
    #
    #     # if we are not using a video file, stop the camera video stream
    # if not args.get("video", False):
    #     vs.stop()
    #
    #     # otherwise, release the camera
    # else:
    #     vs.release()
    #     # Convert the trajectory list to a NumPy array for further processing if needed
    #
    #     # close all windows
    # cv2.destroyAllWindows()
    #
    #
