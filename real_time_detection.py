
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("facial_expression_model.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

labels = {0: "Happy", 1: "sad"}

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))  # Resize to match model input
        face = face / 255.0  # Normalize pixel values
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Predict expression
        prediction = model.predict(face)
        label = labels[int(prediction[0] > 0.5)]  # Threshold: 0.5

        # Draw rectangle and label around the face
        color = (0, 255, 0) if label == "Smile" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow("Facial Expression Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()