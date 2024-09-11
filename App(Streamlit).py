import streamlit as st
import pickle
import cv2
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import mediapipe as mp
import os

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Load the data to get the maximum length
data_dict = pickle.load(open('./data.pickle', 'rb'))
max_length = max(len(sample) for sample in data_dict['data'])

# Define the label dictionary
labels_dict = {0: 'OK', 1: 'PEACE', 2: 'THUMBS UP'}

# Streamlit app
st.title("Hand Sign Detection")
st.write("This application uses a model to predict hand signs: **OK**, **PEACE**, and **THUMBS UP**.")

# Add a button for camera access
start_camera = st.button("Turn on Camera")
stop_camera = False

if start_camera:
    st.write("Camera is ON. Detecting hand signs...")
    
    # Add a button to stop the camera
    stop_button = st.button("Stop Camera")

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while not stop_camera:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to read from camera.")
            break

        # Prepare the frame for hand detection
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Extract landmark coordinates
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Pad the data to match the training feature size
            data_padded = pad_sequences([data_aux], maxlen=max_length, padding='post', dtype='float32')
            prediction = model.predict(data_padded)

            predicted_character = labels_dict[int(prediction[0])]

            # Calculate bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Display the frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")

        # Check if the stop button is pressed
        if stop_button:
            stop_camera = True

    cap.release()
    st.write("Camera is OFF.")
else:
    st.write("Click the button above to start the camera.")
