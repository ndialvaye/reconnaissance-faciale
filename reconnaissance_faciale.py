import cv2
import streamlit as st
import numpy as np

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in an image
def detect_faces(image, scale_factor, min_neighbors, rect_color):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Draw rectangles around faces with selected color
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), rect_color, 2)

    return image, faces

# Function to save image with faces detected
def save_image(image):
    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    save_path = "output_image.jpg"
    cv2.imwrite(save_path, rgb_image)
    return save_path

# Streamlit app
def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("""
    This app uses the Viola-Jones algorithm for face detection. You can adjust parameters and save the result.
    - Use the button below to start detecting faces using your webcam.
    - Adjust the parameters such as `scaleFactor` and `minNeighbors`.
    - Pick a color for the rectangles that highlight detected faces.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    # If file is uploaded
    if uploaded_file:
        # Convert to an image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Parameters for face detection
        scale_factor = st.slider("Scale Factor", 1.1, 1.5, 1.3, 0.01)
        min_neighbors = st.slider("Min Neighbors", 1, 10, 5)
        rect_color = st.color_picker("Pick a color for the rectangles", "#FF0000")

        # Convert hex color to BGR format
        bgr_color = tuple(int(rect_color[i:i+2], 16) for i in (1, 3, 5))

        # Detect faces and draw rectangles
        detected_image, faces = detect_faces(image, scale_factor, min_neighbors, bgr_color)

        st.image(detected_image, channels="BGR", caption="Processed Image", use_column_width=True)

        st.write(f"Number of faces detected: {len(faces)}")

        # Save the processed image
        save_path = save_image(detected_image)
        st.write(f"Image saved to: {save_path}")
        st.download_button(label="Download Image", data=open(save_path, "rb").read(), file_name=save_path, mime="image/jpeg")

    # Button to start webcam face detection (optional)
    if st.button("Start Webcam"):
        run_webcam()

def run_webcam():
    st.write("Webcam is starting...")

    # Start webcam and detect faces in real-time
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        detected_frame, faces = detect_faces(frame, 1.3, 5, (0, 0, 255))  # Red color by default

        # Show the frame with detected faces
        st.image(detected_frame, channels="BGR", use_column_width=True)

        # If no faces are detected, break the loop
        if len(faces) > 0:
            break

    cap.release()
    st.write("Webcam stopped.")

# Run the app
if __name__ == "__main__":
    app()
