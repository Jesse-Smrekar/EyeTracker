import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from pygrabber.dshow_graph import FilterGraph

"""
    Set these debugging params if desired
"""
SHOW_EYE_TRACE = True
SHOW_IRIS_TRACE = True
SELECTED_INPUT_DEVICE = 0


print("#########################\nList of available input devices\n#########################")
input_devices = FilterGraph().get_input_devices()
available_cameras = {}
for device_index, device_name in enumerate(input_devices):
    print(str(device_index), " : ", device_name)
print("#########################")

# SELECTED_INPUT_DEVICE = int(input("Input the number for the desired camera: "))


CAMERA = cv2.VideoCapture(SELECTED_INPUT_DEVICE)
FACE = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
SCREEN_W, SCREEN_H = pyautogui.size()

# Don't ask
LEFT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_LANDMARKS = [33, 7, 163, 144, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = list(range(474,478))
RIGHT_IRIS = list(range(469, 473))


while True:
    _, raw = CAMERA.read()
    frame = cv2.flip(raw, 1)

    # exit on 'ESC' key press
    key = cv2.waitKey(1)
    if key == 27:
        break

    # generate overlay frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_landmarks = FACE.process(rgb_frame).multi_face_landmarks

    height, width = frame.shape[:2]

    if face_landmarks:
        scaled_points = []
        for point in face_landmarks[0].landmark:
            scaled_points.append(np.multiply([point.x, point.y], [width, height]).astype(int))

        scaled_points = np.asarray(scaled_points)
        if SHOW_EYE_TRACE:
            cv2.polylines(frame, [scaled_points[LEFT_EYE_LANDMARKS]], True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.polylines(frame, [scaled_points[RIGHT_EYE_LANDMARKS]], True, (0, 255, 0), 1, cv2.LINE_AA)

        (l_eye_center_x, l_eye_center_y), l_eye_radius = cv2.minEnclosingCircle(scaled_points[LEFT_IRIS])
        (r_eye_center_x, r_eye_center_y), r_eye_radius = cv2.minEnclosingCircle(scaled_points[RIGHT_IRIS])
        l_eye_center = np.array([l_eye_center_x, l_eye_center_y], dtype=np.int32)
        r_eye_center = np.array([r_eye_center_x, r_eye_center_y], dtype=np.int32)

        if SHOW_IRIS_TRACE:
            cv2.circle(frame, l_eye_center, int(l_eye_radius), (255, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, r_eye_center, int(r_eye_radius), (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow(input_devices[SELECTED_INPUT_DEVICE], frame)

CAMERA.release()
cv2.destroyAllWindows()