import math

import torch

from gaze_tracking import GazeTracking
from collections import deque
import numpy as np
import argparse
import imutils
import time
import mediapipe as mp
import run
#import Laser
import face_expersion
import skleaton as sk
import cv2

from utils.general import non_max_suppression, scale_coords, check_img_size, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from models.experimental import attempt_load


class Game1:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    fr = face_expersion.FaceRecognition1()
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64,
                    help="max buffer size")
    args = vars(ap.parse_args())
    trajectory = []

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    pts = deque(maxlen=args["buffer"])
    time.sleep(2.0)

    # Initialize Mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Initialize variables
    circle_counter = 0
    prev_angle = 0
    count = -1
    is_circle_detected = False
    is_first_detection = True
    is_initial_circle_detected = False
    gaze = GazeTracking()
    yo = run.yolo()
    face_name=""
    face_expersion_now=""
    yolo_now =[]
    imgsz, stride, device, half ,model ,names ,colors , classes = yo.run()

    classes_to_filter = ['train', 'person',
                         'cell phone']  # You can give a list of classes to filter by name ['train', 'person']

    opt = {
        "weights": "weight/yolov7.pt",  # Path to weights file; default weights are for the YOLOv7 "nano" model
        "yaml": "data/coco.yaml",
        "img-size": 640,  # Default image size
        "conf-thres": 0.25,  # Confidence threshold for inference
        "iou-thres": 0.45,  # NMS IoU threshold for inference
        "device": '0',  # Device to run the model on, e.g., '0' or '0,1,2,3' for GPU, or 'cpu' for CPU
        "classes": classes_to_filter  # List of classes to filter or None
    }
    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)



    def run(self):
        video_capture = cv2.VideoCapture(0)

        with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7) as hands:
            while video_capture.isOpened():
                success, frame = video_capture.read()
                if not success:
                    break

                # Convert the image to RGB
                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

                # Process the image
                results = hands.process(frame)

                # Draw hand landmarks on the image
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        # Check if the detected hand is the left hand
                        if hand_handedness.classification[0].label == 'Left':
                            self.mp_drawing.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                            # Get the finger nail landmarks
                            thumb_nail = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                            index_nail = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            middle_nail = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                            ring_nail = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
                            pinky_nail = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

                            # Calculate the angle formed by the finger nail landmarks
                            angle = math.degrees(
                                math.atan2(index_nail.y - pinky_nail.y, index_nail.x - pinky_nail.x)
                                - math.atan2(thumb_nail.y - ring_nail.y, thumb_nail.x - ring_nail.x)
                            )

                            # Check if the finger trajectory forms a circle
                            self.count = sk.is_circle_trajectory(angle)

                else:
                    self.is_circle_detected = False
                    if self.is_initial_circle_detected:
                        self.is_first_detection = True

                # Display the count on top of the video
                cv2.putText(frame, f"Count: {self.count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                ###################


                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                txt = face_expersion.deep_face(frame)
                self.face_expersion_now=txt
                self.fr.process_current_frame1(frame)
                self.face_name = self.fr.name_now

                # Laser

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

                #############
                # Gaze Tracking

                 # We send this frame to GazeTracking to analyze it
                self.gaze.refresh(frame)

                frame = self.gaze.annotated_frame()
                text = ""

                if self.gaze.is_blinking():
                    text = "Blinking"
                elif self.gaze.is_right():
                    text = "Looking right"
                elif self.gaze.is_left():
                    text = "Looking left"
                elif self.gaze.is_center():
                    text = "Looking center"

                cv2.putText(frame, text, (48, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

                left_pupil = self.gaze.pupil_left_coords()
                right_pupil = self.gaze.pupil_right_coords()
                cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                                (147, 58, 31), 1)
                cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                                (147, 58, 31), 1)

                for (top, right, bottom, left), name in zip(self.fr.face_locations, self.fr.face_names):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    # Create the frame with the name
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                    cv2.putText(frame, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                #####################
                with torch.no_grad():
                    fps = video_capture.get(cv2.CAP_PROP_FPS)
                    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
                    # class_counts = {class_name: 0 for class_name in self.classes_to_filter}
                    if success:
                        img = self.letterbox(frame, self.imgsz, stride=self.stride)[0]
                        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                        img = np.ascontiguousarray(img)
                        img = torch.from_numpy(img).to(self.device)
                        img = img.half() if self.half else img.float()  # uint8 to fp16/32
                        img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)
                        t1 = time_synchronized()
                        pred = self.model(img, augment=False)[0]

                        pred = non_max_suppression(pred, self.opt['conf-thres'], self.opt['iou-thres'], classes=self.classes,
                                                   agnostic=False)
                        t2 = time_synchronized()

                        t2 = time_synchronized()

                        # Reset counts for each frame
                        class_counts = {class_name: 0 for class_name in self.classes_to_filter}
                        for i, det in enumerate(pred):
                            s = ''
                            s += '%gx%g ' % img.shape[2:]  # print string
                            gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
                            if len(det):
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                                for c in det[:, -1].unique():
                                    n = (det[:, -1] == c).sum()  # detections per class
                                    class_counts[self.names[int(c)]] += n  # increment class count
                                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                                for *xyxy, conf, cls in reversed(det):
                                    label = f'{self.names[int(cls)]} {conf:.2f}'

                                    plot_one_box(xyxy, frame, label=label, color=self.colors[int(cls)], line_thickness=3)

                        # Print class counts on the screen
                        class_counts_str = ', '.join(
                            [f"{class_name}: {class_counts[class_name]}" for class_name in self.classes_to_filter])
                        self.yolo_now=class_counts_str
                        cv2.putText(frame, class_counts_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Show the image
                cv2.imshow('Fire system Game', frame)
                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        # while True:
        #     ret,frame = video_capture.read()
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        #     for (x, y, w, h) in faces:
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        #     txt = face_expersion.deep_face(frame)
        #
        #     self.fr.process_current_frame1(frame)
        #
        #
        #     #Laser
        #
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
        #     mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
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
        #             self.trajectory.append(center)
        #             # Print the current center coordinates
        #         # print(center)
        #         # update the points queue
        #         self.pts.appendleft(center)
        #         # trajectory = np.array(trajectory)
        #         # print(trajectory)
        #         # loop over the set of tracked points
        #     for i in range(1, len(self.pts)):
        #         # if either of the tracked points are None, ignore
        #         # them
        #         if self.pts[i - 1] is None or self.pts[i] is None:
        #             continue
        #
        #             # otherwise, compute the thickness of the line and
        #             # draw the connecting lines
        #         thickness = int(np.sqrt(self.args["buffer"] / float(i + 1)) * 2.5)
        #         cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)
        #
        #     #############
        #     #skealton
        #
        #
        #
        #
        #
        #
        #     for (top, right, bottom, left), name in zip(self.fr.face_locations, self.fr.face_names):
        #         # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        #         top *= 4
        #         right *= 4
        #         bottom *= 4
        #         left *= 4
        #         # Create the frame with the name
        #         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #         cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        #         cv2.putText(frame, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        #
        #     cv2.imshow('Face Recognition and expression', frame)
        #
        #     # # Hit 'q' on the keyboard to quit!
        #     # if cv2.waitKey(1) == ord('q'):
        #     #     break
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()