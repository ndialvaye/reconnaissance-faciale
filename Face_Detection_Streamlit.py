import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import tempfile
from datetime import datetime
from uuid import uuid4

# === Dossier pour visages connus ===
data_dir = "known_faces"
os.makedirs(data_dir, exist_ok=True)

# === Chargement des visages connus ===
def load_known_faces():
    known_encodings = []
    known_names = []
    for file in os.listdir(data_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(data_dir, file))
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(file)[0])
    return known_encodings, known_names

known_face_encodings, known_face_names = load_known_faces()

# === Initialisation de Viola-Jones ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Interface Streamlit ===
st.title("Détection et reconnaissance de visages - Viola-Jones + Deep Learning")

st.markdown("""
## Instructions
- Activez la webcam.
- Ajustez les paramètres de détection `scaleFactor` et `minNeighbors`.
- Choisissez la couleur du rectangle de détection.
- Cliquez sur **Capturer l'image** pour sauvegarder une image.
- Si un visage est inconnu, vous pouvez l'enregistrer avec un nom.
""")

# Paramètres utilisateur
color = st.color_picker("Couleur du rectangle", "#00FF00")
color_bgr = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
scaleFactor = st.slider("scaleFactor (Viola-Jones)", 1.1, 2.0, 1.3, step=0.1)
minNeighbors = st.slider("minNeighbors (Viola-Jones)", 1, 10, 5)
capture = st.button("📸 Capturer l'image")
stframe = st.empty()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Erreur : Impossible d'accéder à la caméra.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur lors de la lecture de la caméra.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détection avec Viola-Jones
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        for (x, y, w, h) in faces:
            top, right, bottom, left = y, x + w, y + h, x
            face_image = rgb_frame[top:bottom, left:right]

            encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
            name = "Inconnu"
            if encoding:
                matches = face_recognition.compare_faces(known_face_encodings, encoding[0])
                if True in matches:
                    match_index = matches.index(True)
                    name = known_face_names[match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), color_bgr, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_bgr, 2)

            if name == "Inconnu":
                form_id = str(uuid4())
                with st.form(key=form_id):
                    new_name = st.text_input("Nom du visage inconnu :")
                    submitted = st.form_submit_button("Enregistrer")
                    if submitted and new_name:
                        save_path = os.path.join(data_dir, f"{new_name}.jpg")
                        face_rgb = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)
                        cv2.imwrite(save_path, face_rgb)
                        st.success(f"Visage de {new_name} enregistré.")
                        known_face_encodings, known_face_names = load_known_faces()
                        break

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if capture:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(tempfile.gettempdir(), f"capture_{timestamp}.png")
            cv2.imwrite(file_path, frame)
            st.success(f"Image capturée : {file_path}")
            st.image(file_path, caption="Image sauvegardée", use_column_width=True)
            break

    cap.release()
