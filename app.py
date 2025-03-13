import os
import cv2
import mediapipe as mp
from flask import Flask, render_template, request, redirect, url_for

# Configuration de Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ANNOTATED_FOLDER"] = "static/annotated"

# Création des dossiers si inexistants
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["ANNOTATED_FOLDER"], exist_ok=True)

# Configuration de MediaPipe
model_path = "models/gesture_recognizer.task"
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

recognizer = GestureRecognizer.create_from_options(options)

# Détection et annotation
def process_image(image_path):
    # Charger l’image
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        return None, "Erreur de chargement de l'image"

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Reconnaissance du geste
    result = recognizer.recognize(mp_image)
    if not result.gestures or not result.gestures[0]:
        return None, "Aucun geste détecté"

    top_gesture = result.gestures[0][0]
    gesture_name = top_gesture.category_name
    gesture_score = top_gesture.score

    # Dessin des landmarks
    height, width, _ = bgr_image.shape
    for landmarks in result.hand_landmarks:
        for landmark in landmarks:
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(bgr_image, (x, y), 5, (0, 255, 0), -1)

    # Ajouter du texte
    cv2.putText(
        bgr_image,
        f"{gesture_name} ({gesture_score:.2f})",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Sauvegarde de l'image annotée
    annotated_path = os.path.join(app.config["ANNOTATED_FOLDER"], os.path.basename(image_path))
    cv2.imwrite(annotated_path, bgr_image)
    
    return annotated_path, f"{gesture_name} ({gesture_score:.2f})"

# Route principale
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            annotated_path, gesture_info = process_image(filepath)
            return render_template("index.html", image_url=annotated_path, result=gesture_info)

    return render_template("index.html", image_url=None, result=None)

if __name__ == "__main__":
    app.run(debug=True)
