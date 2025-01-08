import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import numpy as np
import cv2
import glob
from PIL import Image
from collections import deque
import supervision as sv
from inference import get_model
from time import time
import streamlit as st
import gradio as gr
from datetime import datetime

detection_api_key = "V12deDQVnSZqs4PkLX0Y"
class ViolenceDetectionModel(nn.Module):
    def __init__(self, sequence_length, hidden_size=256):
        super(ViolenceDetectionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove final classification layer
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Final classification layer

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.resnet(x)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return torch.sigmoid(out)

# Violence detection and detection models
classification_model = ViolenceDetectionModel(sequence_length=7, hidden_size=512)
classification_model.load_state_dict(torch.load(r"notebook/best_violence_detection_model.pth"))
classification_model.eval()
# , map_location=torch.device('cpu')
detection_model = get_model(model_id="emergency-response/1", api_key=detection_api_key)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize buffers and logging
frame_buffer = deque(maxlen=30)  # Buffer to hold frames
logs = []  # Log for timestamps of detected violence

# Streamlit UI elements
st.title("Real-Time Violence Detection")
st.write("Live video feed and violence detection with timestamp logs")

video_placeholder = st.empty()  # For live video
logs_placeholder = st.empty()  # For detection logs

def classify_frames(frame_buffer):
    """Function to classify violence in a series of frames."""
    frames = list(frame_buffer)
    frames_tensor = torch.stack([transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for frame in frames])
    frames_tensor = frames_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = classification_model(frames_tensor)
        predicted = (outputs > 0.5).float()

    return predicted.item() == 1  # Return True if violence is detected

# Capture video (either webcam or video file)
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Append frame to the buffer for classification
    frame_buffer.append(frame)

    # Display the live video in Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB")

    # Perform violence classification every time the buffer is full
    if len(frame_buffer) == frame_buffer.maxlen:
        if classify_frames(frame_buffer):
            # Log the timestamp when violence is detected
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logs.append(f"Violence detected at {timestamp}")
            st.write(logs[-1])  # Print the latest log to the Streamlit UI

    # Display the logs in Streamlit
    logs_placeholder.text("\n".join(logs))

    # Exit condition for the loop (press 'q' to stop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
