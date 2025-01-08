import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time

# Load the YOLOv10 model
model = YOLO("yolov10s.pt")

# Define a function to check for alerts
def check_for_alerts(detections):
    person_count = 0
    suspicious_objects = []
    
    # Iterate over the detected objects
    for det in detections:
        label = det['label']
        
        # Check if a person is detected
        if label == 'person':
            person_count += 1
        
        # Check for suspicious objects like cell phone or book
        elif label in ['cell phone', 'book']:
            suspicious_objects.append(label)
    
    # Define alert messages
    alert_messages = []
    
    # Alert if no person or more than one person is detected
    if person_count == 0:
        alert_messages.append("ALERT: No person detected!")
    elif person_count > 1:
        alert_messages.append("ALERT: More than one person detected!")
    
    # Alert if suspicious objects are detected
    if suspicious_objects:
        alert_messages.append(f"ALERT: Suspicious objects detected: {', '.join(suspicious_objects)}")
    
    return alert_messages

# Streamlit app layout
st.title("Online Proctoring System with YOLOv10")
st.markdown("Detects real-time objects such as persons, cell phones, and books.")

# Start button to begin detection
start_detection = st.button("Start Detection")

# Initialize webcam
cap = None

if start_detection:
    # Open the default webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if cap is None or not cap.isOpened():
        st.error("Error: Could not open video source.")
    else:
        stframe = st.empty()  # Placeholder for displaying video frames
        status_placeholder = st.empty()  # Placeholder for alerts
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                st.error("Error: Could not read frame from video source.")
                break
            
            # Perform object detection on the current frame
            results = model.predict(frame, conf=0.25)
            
            detections = []
            for result in results:
                # Get bounding boxes and class names for each detection
                for box in result.boxes:
                    # Extract class label and confidence
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    confidence = box.conf[0]

                    # Store detection results
                    detections.append({
                        'label': label,
                        'confidence': float(confidence),
                        'bbox': box.xyxy[0].cpu().numpy()
                    })
            
            # Check for alerts
            alert_messages = check_for_alerts(detections)
            
            # Display alert messages in the Streamlit app
            if alert_messages:
                status_placeholder.error("\n".join(alert_messages))
            else:
                status_placeholder.success("Status: All clear!")
            
            # Draw bounding boxes on the frame
            for det in detections:
                x1, y1, x2, y2 = [int(c) for c in det['bbox']]
                label = det['label']
                confidence = det['confidence']
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Convert the frame (OpenCV uses BGR, but Streamlit uses RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame in the Streamlit app
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Control the frame rate (e.g., 30 FPS)
            time.sleep(1/30)
        
        # Release the webcam after the loop
        cap.release()
        cv2.destroyAllWindows()