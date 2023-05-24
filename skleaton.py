import cv2
import mediapipe as mp
import math

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







# Function to check if the angle formed by three points is close to 180 degrees
def is_angle_close_to_180(angle):
    return math.isclose(angle, 180, rel_tol=30)

# Function to check if the finger trajectory forms a circle
def is_circle_trajectory(angle):
    global circle_counter, prev_angle, count, is_circle_detected, is_first_detection, is_initial_circle_detected

    if is_angle_close_to_180(angle) and is_angle_close_to_180(prev_angle):
        circle_counter += 1
        if circle_counter >= 10 and not is_circle_detected and not is_first_detection and is_initial_circle_detected:
            count += 1
            is_circle_detected = True

    else:
        circle_counter = 0
        is_circle_detected = False

    prev_angle = angle
    is_first_detection = False
    is_initial_circle_detected = True

    return count

def start_hand_tracking():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Set up Mediapipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert the image to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # Process the image
            results = hands.process(image)

            # Draw hand landmarks on the image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Check if the detected hand is the left hand
                    if hand_handedness.classification[0].label == 'Left':
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Get the finger nail landmarks
                        thumb_nail = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        index_nail = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        middle_nail = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                        ring_nail = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                        pinky_nail = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                        # Calculate the angle formed by the finger nail landmarks
                        angle = math.degrees(
                            math.atan2(index_nail.y - pinky_nail.y, index_nail.x - pinky_nail.x)
                            - math.atan2(thumb_nail.y - ring_nail.y, thumb_nail.x - ring_nail.x)
                        )

                        # Check if the finger trajectory forms a circle
                        is_circle_trajectory(angle)

            else:
                is_circle_detected = False
                if is_initial_circle_detected:
                    is_first_detection = True

            # Display the count on top of the video
            cv2.putText(image, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show the image
            cv2.imshow('Hand Tracking', image)
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the webcam and destroy the windows
    cap.release()
    cv2.destroyAllWindows()
