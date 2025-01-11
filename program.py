import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import tkinter as tk
from tkinter import StringVar, ttk
from PIL import Image, ImageTk

# Load the trained image classification model and class labels
model = load_model("HMDB51.keras")
class_labels = np.load('classes.npy')

# Initialize Mediapipe for pose and hand landmarks
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize GUI for action selection
root = tk.Tk()
root.title("Action Detection Selector")
root.geometry("800x500")  # Adjusted size for both frames

# Container frames for layout
left_frame = tk.Frame(root, width=700, height=700, bg='white')  # Larger frame for video
left_frame.grid(row=0, column=0, sticky="nswe")
right_frame = tk.Frame(root, width=300, height=700)
right_frame.grid(row=0, column=1, sticky="nswe")

# Configure column and row weights for resizing
root.grid_columnconfigure(0, weight=3)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)

selected_action = StringVar()
selected_action.set("sit")  # Default action

# Canvas for real-time action detection
canvas = tk.Canvas(left_frame, width=640, height=480)
canvas.pack()

# Function to start real-time detection based on selected action
def start_detection():
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for Mediapipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            # Convert frame back to BGR for displaying
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            # Preprocess the frame for prediction
            prediction_image = cv2.resize(frame, (150, 150))
            prediction_image = prediction_image.astype('float32') / 255.0
            prediction_image = np.expand_dims(prediction_image, axis=0)

            # Predict the class
            prediction = model.predict(prediction_image, verbose=0)
            predicted_label = class_labels[np.argmax(prediction)]

            # Display the selected action only if it matches
            if predicted_label == selected_action.get():
                display_text = f"Action Detected: {predicted_label}"
            else:
                display_text = "No matching action detected"

            # Display the prediction
            cv2.putText(image, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert image to PIL format for tkinter
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection

            # Refresh the tkinter window
            root.update_idletasks()
            root.update()

            # Exit on pressing 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Create GUI components in the right frame
label = tk.Label(right_frame, text="Select Action to Detect:")
label.pack(pady=5)

# Use radio buttons for action selection
actions = ["sit", "stand", "jump", "run", "situp", "walk"]
for action in actions:
    tk.Radiobutton(right_frame, text=action.capitalize(), variable=selected_action, value=action, indicator= 0 , background="light blue", width=30).pack(anchor=tk.W, pady=2)

start_button = tk.Button(right_frame, text="Start Detection", command=start_detection)
start_button.pack(pady=10)

exit_button = tk.Button(right_frame, text="Exit", command=root.destroy)
exit_button.pack(pady=5)

# Run the GUI event loop
root.mainloop()
