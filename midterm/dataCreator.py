import cv2
import csv
import mediapipe as mp
import numpy as np
import glob
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Initialize the data format
def init_posing_format(posing):
    landmarks = ["class"]
    for val in range(1, len(posing.pose_landmarks.landmark) + 1):
        landmarks.extend(['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)])
    print(landmarks)

    with open("pose_label.csv", mode='w', newline='') as f:
        csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL).writerow(landmarks)


def capture_pose(lmark, class_name):
    pose = lmark.pose_landmarks.landmark
    pose_ar = list(
        np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten()
    )
    pose_ar.insert(0, class_name)
    print(pose_ar)

    with open("pose_label.csv", mode='a', newline='') as f:
        csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL).writerow(pose_ar)


# Getting pose by using webcam
def get_pose_realtime():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # pass by reference.
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            cv2.imshow('Press ESC to initialize the format', image)

            if cv2.waitKey(5) & 0xFF == 27:
                try:
                    init_posing_format(results)
                    break
                except:
                    print("Initialization failed. Try again.")
                    continue
    cap.release()


# Collect label by leading videos
def get_pose_video(name):
    videos = glob.glob(name + "/*.mp4")

    if len(videos) == 0:
        print("Could not find any videos. exit the function.")
        return

    print(videos)
    for i in videos:
        cap = cv2.VideoCapture(i)

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print(name, "finished")
                    break

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # pass by reference.
                results = pose.process(image)

                # Draw the pose annotation on the image.
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                cv2.imshow("Capturing pose " + i + "...", image)
                try:
                    capture_pose(results, name)
                except:
                    print("Failed to capture frame.")
                    continue
                cv2.waitKey(40)
        cap.release()


if __name__ == "__main__":
    while True:
        num = input("Select your job\n 1. Initialize the pose data \n 2. Add class\n")
        if num == "1":
            get_pose_realtime()
            break
        elif num == "2":
            name = input("Write down class name here: ")
            get_pose_video(name)
            break
        else:
            print("Wrong number detected. Try again.")

    cv2.destroyAllWindows()
