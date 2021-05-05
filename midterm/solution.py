import pickle
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


def posing_realtime(my_model):
    f = open(my_model, "rb")
    model = pickle.load(my_model)
    f.close()

    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as pose:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                pose_ar = np.array(
                    [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]
                ).flatten()
                pose_ar = [list(pose_ar)]
                x_captured = pd.DataFrame(pose_ar)
                pose_predict = model.predict(x_captured)[0]
                pose_predict_prob = model.predict_proba(x_captured)[0]
                print("Detected: ", pose_predict, pose_predict_prob)

                cur_pose = results.pose_landmarks.landmark
                shoulder_x = cur_pose[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
                shoulder_y = cur_pose[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                shoulder = np.array((shoulder_x, shoulder_y))
                cv_window = np.array([1280, 720])
                crd_left = tuple((shoulder * cv_window).astype(int))

                # Draw predicted data
                cv2.rectangle(
                    image,
                    (crd_left[0], crd_left[1] + 5),
                    (crd_left[0] + len(pose_predict) * 20, crd_left[1] - 35),
                    (255, 255, 255),
                    -1
                )

                cv2.putText(
                    image,
                    pose_predict + " : " + str(round(pose_predict_prob[np.argmax(pose_predict_prob)], 2)),
                    crd_left,
                    cv2.FONT_ITALIC,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )
            except:
                print("Error occurred")

            cv2.imshow('Press ESC to initialize the format', image)
            if cv2.waitKey(5) & 0xFF == 27:
                print("Exiting")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    posing_realtime("gunnerDetector.pkl")