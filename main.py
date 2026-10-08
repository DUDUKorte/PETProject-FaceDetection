import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import text
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("porra")
    exit()
ld_frame_timestamp_ms = 0
ps_frame_timestamp_ms = 1


### LandMarker options
lm_model_path = './face_landmarker.task'
lm_BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the video mode:
lm_options = FaceLandmarkerOptions(
    base_options=lm_BaseOptions(model_asset_path=lm_model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes = True
    )

landmarker = FaceLandmarker.create_from_options(lm_options)

### Gesture Options
gt_model_path = './gesture_recognizer.task'
gt_BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the video mode:
gt_options = GestureRecognizerOptions(
    base_options=gt_BaseOptions(model_asset_path=gt_model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands = 2
    )

gt_recognizer = GestureRecognizer.create_from_options(gt_options)

### Pose optioms
ps_model_path = './pose_landmarker_full.task'
ps_BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
ps_options = PoseLandmarkerOptions(
    base_options=ps_BaseOptions(model_asset_path=ps_model_path),
    running_mode=VisionRunningMode.VIDEO)

ps_landmarker = PoseLandmarker.create_from_options(ps_options)


### Global attribbutes
current_gesture = "None"
current_expression = "None"

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    ## os landmarks bonitinhos
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
    ## Dsenha o contorno do rosto e dos olhos/sobrancelha (Linha grossa branca)
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
    ## Desenha iris esqueda
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    ## Desenha a iris da direita
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

  return annotated_image

def get_face_blendshapes(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  return face_blendshapes_names[face_blendshapes_scores.index(max(face_blendshapes_scores))]

def get_landmark_frame(mp_image, frame):
    global current_expression

    face_landmarker_result = landmarker.detect_for_video(mp_image, ld_frame_timestamp_ms)
    
    annotated_image = draw_landmarks_on_image(frame, face_landmarker_result)
            
    if(face_landmarker_result.face_blendshapes):
        current_expression = str(get_face_blendshapes(face_landmarker_result.face_blendshapes[0]))
        #cv2.putText(annotated_image, get_face_blendshapes(face_landmarker_result.face_blendshapes[0]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 210, 250), 2)

    return annotated_image

def draw_gesture_landmarks_on_image(frame, gesture_landmarks_result):
    annotated_image = frame.copy()

    for hand_landmark in gesture_landmarks_result:
        drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmark,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style()
            )

    return annotated_image

def get_gesture_frame(mp_image, frame):
    global current_gesture
    gesture_recognition_result = gt_recognizer.recognize_for_video(mp_image, ld_frame_timestamp_ms)
    
    if(gesture_recognition_result.gestures):
        top_gesture = gesture_recognition_result.gestures[0][0]
        gesture_landmarks_result = gesture_recognition_result.hand_landmarks
        
        annotated_image = draw_gesture_landmarks_on_image(frame, gesture_landmarks_result)
        
        current_gesture = str(top_gesture.category_name)
        #cv2.putText(annotated_image, str(top_gesture.category_name), (40, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 210, 250), 2)
        return annotated_image

    return frame

def draw_pose_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    excluded_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22]

    for pose_landmarks in pose_landmarks_list:
        # "remove" os landmarks que estão na lista de excluidos (na vdd só deixa eles invisíveis)
        for index in excluded_indexes:
            pose_landmarks[index].visibility = 0

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style)

    return annotated_image


def get_pose_frame(mp_image, frame):
    pose_landmarker_result = ps_landmarker.detect_for_video(mp_image, ps_frame_timestamp_ms)
    
    pose_frame = draw_pose_landmarks_on_image(frame, pose_landmarker_result)

    return pose_frame

if __name__ == '__main__':
    while(True):
        ret, frame = cap.read()

        if not ret:
            print("Deu ruim")
            break

        ld_frame_timestamp_ms += int(cap.get(cv2.CAP_PROP_FPS))
        ps_frame_timestamp_ms += int(cap.get(cv2.CAP_PROP_FPS))
    
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        #Get frame with face landmarks
        landmark_frame = get_landmark_frame(mp_image, frame)
        #Get frame with gesture landmarks
        gesture_frame = get_gesture_frame(mp_image, landmark_frame)
        #Get frame with pose landmarks
        pose_frame = get_pose_frame(mp_image, gesture_frame)

        cv2.putText(pose_frame, current_expression, (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 210, 250), 2)
        cv2.putText(pose_frame, current_gesture, (40, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 210, 250), 2)

        cv2.imshow('frame', pose_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()