import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import numpy as np
import glob
import torch
import cv2
import os
from collections import deque
from PIL import Image
import supervision as sv


#Define ViolenceDetection Model
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

detection_api_key = "V12deDQVnSZqs4PkLX0Y"

# Function to classify frames
def classify_frames(classification_model, frame_buffer, transform):
    frames = list(frame_buffer)
    frames_tensor = torch.stack([transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for frame in frames])
    frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = classification_model(frames_tensor)
        predicted = (outputs > 0.5).float()

    return predicted.item() == 1  # True if violence detected

#Detection and annotation function
def detect_violence(detection_model, frames):
    annotated_frames = []
    for frame in frames:
        results = detection_model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frames.append(annotated_frame)

    return annotated_frames

# Process video function
def process_video(video_file, classification_model, detection_model):
    # Frame transformation pipeline
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    # Buffers for past frames and future frames
    frame_buffer = deque(maxlen=150)  # Buffer for past frames
    future_frames_buffer = []         # Buffer for future frames
    
    violence_detected = False         # To check if violence is detected in past frames
    non_violence_active = False       # To handle non-violence detection similarly
    violence_active = False           # To check if violence detection continues for future frames
    
    cap = cv2.VideoCapture(video_file)  # Open video file
    output_frames = []                  # List to store annotated frames
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    future_frames_limit = fps * 5       # Number of frames for the next 5 seconds (~150 frames at 30fps)
    future_frames_collected = 0         # To track how many future frames have been processed
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if violence_active:
            # Collect future frames for violence detection continuation
            future_frames_buffer.append(frame)
            future_frames_collected += 1

            if future_frames_collected >= future_frames_limit:
                violence_active = False  # Stop after collecting 5 seconds worth of frames

            # Perform object detection and violence classification for future frames
            annotated_future_frame = detect_violence(detection_model, [frame])[0]
            output_frames.append(annotate_frame(annotated_future_frame, "Violence"))

        elif non_violence_active:
            # Collect future frames for non-violence continuation
            future_frames_buffer.append(frame)
            future_frames_collected += 1

            if future_frames_collected >= future_frames_limit:
                non_violence_active = False  # Stop after collecting 5 seconds worth of frames

            # Annotate non-violence frames without object detection (no bounding boxes)
            output_frames.append(annotate_frame(frame, "Non-Violence"))

        else:
            # Add the current frame to the past frame buffer
            frame_buffer.append(frame)
            
            # When buffer is full, perform violence or non-violence detection on past frames
            if len(frame_buffer) == frame_buffer.maxlen:
                violence_detected = classify_frames(classification_model, frame_buffer, transform)
                
                if violence_detected:
                    violence_active = True
                    future_frames_collected = 0  # Reset future frames count
                    
                    # Collect past frames and annotate them as violent, with bounding boxes
                    past_frames = list(frame_buffer)
                    annotated_past_frames = detect_violence(detection_model, past_frames)
                    output_frames.extend([annotate_frame(f, "Violence") for f in annotated_past_frames])
                    
                    # Clear future frames buffer to start fresh
                    future_frames_buffer.clear()
                else:
                    non_violence_active = True
                    future_frames_collected = 0  # Reset future frames count
                    
                    # Collect past frames and annotate them as non-violent (no bounding boxes)
                    past_frames = list(frame_buffer)
                    output_frames.extend([annotate_frame(f, "Non-Violence") for f in past_frames])

                    # Clear future frames buffer to start fresh
                    future_frames_buffer.clear()

    cap.release()

    # Write all annotated frames to an output video file
    output_video_path = "output_annotated.mp4"
    height, width, layers = output_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in output_frames:
        out.write(frame)

    out.release()
    return output_video_path

def annotate_frame(frame, label):
    """
    Adds a label ('Violence' or 'Non-Violence') to the frame.
    This assumes the frame already contains bounding boxes for violence, but not for non-violence.
    """
    # Set text color (red for violence, green for non-violence)
    color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
    
    # Draw the label on the top-left corner of the frame
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return frame