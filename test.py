import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'models/gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

recognizer = GestureRecognizer.create_from_options(options)

image_path = "assets/image.png"
bgr_image = cv2.imread(image_path)
if bgr_image is None:
    raise FileNotFoundError(f"Impossible de lire l'image à l'emplacement : {image_path}")

# MediaPipe attend généralement des images en RGB
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

# Conversion en objet mp.Image
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

# Reconnaissance de gestes
gesture_recognition_result = recognizer.recognize(mp_image)
print(gesture_recognition_result)


# Connections entre les 21 points de la main (mêmes indices que la solution "hands")
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # Pouce
    (0, 5), (5, 6), (6, 7), (7, 8),   # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Majeur
    (0, 13), (13, 14), (14, 15), (15, 16),# Annulaire
    (0, 17), (17, 18), (18, 19), (19, 20) # Auriculaire
]

annotated_image = bgr_image.copy()
height, width, _ = annotated_image.shape

# Pour chaque main détectée :
for hand_idx, landmarks in enumerate(gesture_recognition_result.hand_landmarks):
    for connection in HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx   = connection[1]
        x1 = int(landmarks[start_idx].x * width)
        y1 = int(landmarks[start_idx].y * height)
        x2 = int(landmarks[end_idx].x * width)
        y2 = int(landmarks[end_idx].y * height)
        cv2.line(
            annotated_image, (x1, y1), (x2, y2),
            color=(0, 255, 0), thickness=2
        )

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(
            annotated_image, (x, y), radius=5,
            color=(0, 0, 255), thickness=-1
        )

    # Annotation du geste reconnu (s’il y en a un)
    if len(gesture_recognition_result.gestures) > hand_idx:
        if len(gesture_recognition_result.gestures[hand_idx]) > 0:
            top_gesture = gesture_recognition_result.gestures[hand_idx][0]
            gesture_name = top_gesture.category_name
            gesture_score = top_gesture.score

            # Récupérer la main droite ou gauche
            handedness_category = gesture_recognition_result.handedness[hand_idx][0]
            handedness_name = handedness_category.category_name  # "Right" ou "Left"

            text_to_display = f"{gesture_name} ({gesture_score*100:.1f}%) - {handedness_name}"
            cv2.putText(
                annotated_image, 
                text_to_display, 
                (30, 30 + 30*hand_idx), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (255, 0, 0), 
                2
            )


cv2.imshow("Résultat ", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Pour sauvegarder l'image annotée :
# cv2.imwrite("annotated_image.png", annotated_image)
