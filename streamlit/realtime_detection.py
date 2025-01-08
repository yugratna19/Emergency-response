# from PIL import Image
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# import cv2
# import numpy as np
# import time

# # Model Definition with LSTM
# class ViolenceDetectionModel(nn.Module):
#     def __init__(self, sequence_length, hidden_size=256):
#         super(ViolenceDetectionModel, self).__init__()
#         self.resnet = models.resnet18(pretrained=True)
#         self.resnet.fc = nn.Identity()  # Remove final classification layer
#         self.sequence_length = sequence_length
        
#         self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=1, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)  # Final classification layer

#     def forward(self, x):
#         batch_size, seq_len, c, h, w = x.size()
#         x = x.view(batch_size * seq_len, c, h, w)
#         x = self.resnet(x)
#         x = x.view(batch_size, seq_len, -1)
#         lstm_out, _ = self.lstm(x)
#         out = lstm_out[:, -1, :]
#         out = self.fc(out)
#         return torch.sigmoid(out)

# # Function to convert frame to tensor for model input
# def preprocess_frame(frame, transform):
#     frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     frame = transform(frame)
#     return frame

# # Function to perform real-time inference and stream video in Streamlit
# def real_time_inference_streamlit(video_placeholder, model, transform, sequence_length=5):
#     # Open the webcam
#     cap = cv2.VideoCapture(0)  # 0 is usually the default for webcam
#     frame_buffer = []  # To store frames for sequence input

#     if not cap.isOpened():
#         print("Error: Unable to access the webcam.")
#         return
    
#     print("Webcam accessed successfully. Starting real-time violence detection...")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame.")
#             break

#         # Preprocess the frame
#         processed_frame = preprocess_frame(frame, transform)
#         frame_buffer.append(processed_frame)

#         # Keep only the last `sequence_length` frames
#         if len(frame_buffer) > sequence_length:
#             frame_buffer.pop(0)

#         # If enough frames have been collected for a sequence, run inference
#         if len(frame_buffer) == sequence_length:
#             input_tensor = torch.stack(frame_buffer)  # Stack frames into a sequence
#             input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (1, sequence_length, C, H, W)

#             # Run inference
#             with torch.no_grad():
#                 model.eval()
#                 outputs = model(input_tensor)
#                 predicted = (outputs > 0.5).float()

#             # Interpret the prediction
#             if predicted.item() == 1:
#                 label = "Violence Detected"
#                 color = (0, 0, 255)  # Red for violence
#             else:
#                 label = "No Violence"
#                 color = (0, 255, 0)  # Green for non-violence

#             # Display the label on the frame
#             cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

#         # Convert the frame to RGB for Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Display the frame in the Streamlit interface
#         video_placeholder.image(frame_rgb, channels="RGB")

#         # Streamlit refreshes the image automatically, so we can use time.sleep for pacing
#         time.sleep(0.03)

#     # Release the webcam when the loop finishes
#     cap.release()

# from PIL import Image
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# import cv2
# import time
# from save_10sec import save_next_10_seconds

# # Model Definition with LSTM
# class ViolenceDetectionModel(nn.Module):
#     def __init__(self, sequence_length, hidden_size=256):
#         super(ViolenceDetectionModel, self).__init__()
#         self.resnet = models.resnet18(pretrained=True)
#         self.resnet.fc = nn.Identity()  # Remove final classification layer
#         self.sequence_length = sequence_length
        
#         self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=1, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)  # Final classification layer

#     def forward(self, x):
#         batch_size, seq_len, c, h, w = x.size()
#         x = x.view(batch_size * seq_len, c, h, w)
#         x = self.resnet(x)
#         x = x.view(batch_size, seq_len, -1)
#         lstm_out, _ = self.lstm(x)
#         out = lstm_out[:, -1, :]
#         out = self.fc(out)
#         return torch.sigmoid(out)

# # Function to convert frame to tensor for model input
# def preprocess_frame(frame, transform):
#     frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     frame = transform(frame)
#     return frame

# # Function to perform real-time inference and stream video in Streamlit
# def real_time_inference_streamlit(video_placeholder, model, transform, sequence_length=5):
#     # Open the webcam
#     cap = cv2.VideoCapture(0)  # 0 is usually the default for webcam
#     frame_buffer = []  # To store frames for sequence input

#     if not cap.isOpened():
#         print("Error: Unable to access the webcam.")
#         return
    
#     print("Webcam accessed successfully. Starting real-time violence detection...")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame.")
#             break

#         # Preprocess the frame
#         processed_frame = preprocess_frame(frame, transform)
#         frame_buffer.append(processed_frame)

#         # Keep only the last `sequence_length` frames
#         if len(frame_buffer) > sequence_length:
#             frame_buffer.pop(0)

#         # If enough frames have been collected for a sequence, run inference
#         if len(frame_buffer) == sequence_length:
#             input_tensor = torch.stack(frame_buffer)  # Stack frames into a sequence
#             input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (1, sequence_length, C, H, W)

#             # Run inference
#             with torch.no_grad():
#                 model.eval()
#                 outputs = model(input_tensor)
#                 predicted = (outputs > 0.5).float()

#             # Interpret the prediction
#             if predicted.item() == 1:
#                 label = "Violence Detected"
#                 color = (0, 0, 255)  # Red for violence
#                 save_next_10_seconds(output_path="violence_clip.mp4")  # Trigger 10-second video recording
#             else:
#                 label = "No Violence"
#                 color = (0, 255, 0)  # Green for non-violence

#             # Display the label on the frame
#             cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

#         # Convert the frame to RGB for Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Display the frame in the Streamlit interface
#         video_placeholder.image(frame_rgb, channels="RGB")

#         # Streamlit refreshes the image automatically, so we can use time.sleep for pacing
#         time.sleep(0.03)

#     # Release the webcam when the loop finishes
#     cap.release()


from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import time
from save_10sec import *
import streamlit as st

# Model Definition with LSTM
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

# Function to convert frame to tensor for model input
def preprocess_frame(frame, transform):
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = transform(frame)
    return frame

# Function to perform real-time inference and stream CCTV video
def real_time_inference_streamlit(video_placeholder, model, transform, sequence_length=5, stream_url="http://194.44.38.196:8083/mjpg/video.mjpg"):
    cap = cv2.VideoCapture(stream_url)  # Use IP camera stream

    frame_buffer = []  # To store frames for sequence input

    if not cap.isOpened():
        st.error("Error: Could not open the live stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame, transform)
        frame_buffer.append(processed_frame)

        # Keep only the last `sequence_length` frames
        if len(frame_buffer) > sequence_length:
            frame_buffer.pop(0)

        # If enough frames have been collected for a sequence, run inference
        if len(frame_buffer) == sequence_length:
            input_tensor = torch.stack(frame_buffer)  # Stack frames into a sequence
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (1, sequence_length, C, H, W)

            # Run inference
            with torch.no_grad():
                model.eval()
                outputs = model(input_tensor)
                predicted = (outputs > 0.5).float()

            # Interpret the prediction
            if predicted.item() == 1:
                label = "Violence Detected"
                color = (0, 0, 255)  # Red for violence
                # save_stream_for_duration(video_source=stream_url, output_path="cctv_clip.mp4", duration=10, fps=30)
  # Trigger 10-second video recording
            else:
                label = "No Violence"
                color = (0, 255, 0)  # Green for non-violence

            # Display the label on the frame
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Convert the frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in the Streamlit interface
        video_placeholder.image(frame_rgb, channels="RGB")

        # Streamlit refreshes the image automatically, so we can use time.sleep for pacing
        time.sleep(0.03)

    cap.release()

# def real_time_inference_streamlit(video_placeholder, model, transform, sequence_length=5):
#     cap = cv2.VideoCapture(0)  # Use laptop webcam (index 0)

#     frame_buffer = []  # To store frames for sequence input

#     if not cap.isOpened():
#         st.error("Error: Could not open the webcam.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to grab frame.")
#             break

#         # Preprocess the frame
#         processed_frame = preprocess_frame(frame, transform)
#         frame_buffer.append(processed_frame)

#         # Keep only the last `sequence_length` frames
#         if len(frame_buffer) > sequence_length:
#             frame_buffer.pop(0)

#         # If enough frames have been collected for a sequence, run inference
#         if len(frame_buffer) == sequence_length:
#             input_tensor = torch.stack(frame_buffer)  # Stack frames into a sequence
#             input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (1, sequence_length, C, H, W)

#             # Run inference
#             with torch.no_grad():
#                 model.eval()
#                 outputs = model(input_tensor)
#                 predicted = (outputs > 0.5).float()

#             # Interpret the prediction
#             if predicted.item() == 1:
#                 label = "Violence Detected"
#                 color = (0, 0, 255)  # Red for violence
#                 save_next_10_seconds(output_path="violence_clip.mp4")  # Trigger 10-second video recording
#             else:
#                 label = "No Violence"
#                 color = (0, 255, 0)  # Green for non-violence

#             # Display the label on the frame
#             cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

#         # Convert the frame to RGB for Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Display the frame in the Streamlit interface
#         video_placeholder.image(frame_rgb, channels="RGB")

#         # Streamlit refreshes the image automatically, so we can use time.sleep for pacing
#         time.sleep(0.03)

#     cap.release()


# Streamlit UI for starting detection
def start_detection():
    st.title("Real-time CCTV Violence Detection")

    video_placeholder = st.empty()

    # Add a button to start the detection
    if st.button("Start Realtime Detection"):
        # Load the model and set the transforms
        sequence_length = 5
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        model = ViolenceDetectionModel(sequence_length=sequence_length)

        real_time_inference_streamlit(video_placeholder, model, transform)