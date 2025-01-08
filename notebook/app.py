import streamlit as st
from detection import ViolenceDetectionModel, process_video
import torch
from torchvision import transforms
import os
from inference import get_model  # Assuming you have this to load detection model

# Set page configuration to wide layout
st.set_page_config(layout="wide")

# Title
st.title("Real-Time Violence Detection and Object Detection")

# Create a layout with two columns
col1, col2 = st.columns([2, 1])  # Col1 takes 2/3 of the width, Col2 takes 1/3

# Load the violence detection model and define the transform
sequence_length = 30  # Set your sequence length
hidden_size = 512  # Set your hidden size
classification_model = ViolenceDetectionModel(sequence_length, hidden_size)

# Load the model weights
classification_model.load_state_dict(torch.load('best_violence_detection_model.pth',map_location=torch.device('cpu')))
classification_model.eval()  # Set the model to evaluation mode

# Define the frame transformation (same as in detection.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),           # Convert to tensor
])

# Load the object detection model
detection_api_key = "V12deDQVnSZqs4PkLX0Y"  # Your detection API key
detection_model = get_model(model_id="emergency-response/1", api_key=detection_api_key)

# Path for the saved video
saved_video_path = "streamlit/output_annotated.mp4"
output_video_path=""

# Add a button or selectbox to switch between modes
mode = st.radio("Select Input Mode", ('Real-Time Video', 'Pre-recorded Video'))

if mode == 'Real-Time Video':
    # Column 1: For Live Video Feed
    with col1:
        st.subheader("Live Video")

        # Create a placeholder for the video stream (You can implement real-time detection here)
        video_placeholder = st.empty()

        # Call the real-time video processing function with model, transform, and placeholder
        # Assuming you have `real_time_inference_streamlit()` implemented
        # real_time_inference_streamlit(video_placeholder, classification_model, transform)

        st.write("Violence Classified at: <Timestamp Placeholder>")

    # Column 2: For the saved video
    with col2:
        st.subheader("Saved Violence Clip")

        # Check if the saved video file exists, then display it
        if os.path.exists(saved_video_path):
            st.video(saved_video_path)
        else:
            st.write("No video saved yet.")

        # Example usage of saving a video
        if st.button("Trigger and Save Next 10 Seconds"):
            # Assuming you have implemented a `save_next_10_seconds()` function
            # save_next_10_seconds(output_path=saved_video_path)
            st.success("10-second video has been saved!")

elif mode == 'Pre-recorded Video':
        
    # Column 1: Pre-recorded Video Feed
    with col1:
        st.subheader("Pre-recorded Video")

        # Create a file uploader for the user to upload a video file
        video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

        if video_file is not None:
            # Show uploaded video
            st.video(video_file)

            # Save uploaded video to a temporary location
            temp_video_path = f"./temp_{video_file.name}"
            with open(temp_video_path, "wb") as f:
                f.write(video_file.getbuffer())

            # Process the video and get the output path using the detection logic
            output_video_path = process_video(temp_video_path, classification_model, detection_model)
            
            # Clean up temporary files
            os.remove(temp_video_path)

    # Column 2: For the saved video
    with col2:
        st.subheader("Saved Violence Clip")
        
        # Check if the saved video file exists, then display it
        if os.path.exists(output_video_path):
            st.write(output_video_path)
            st.success("Video processed successfully!")
            os.system('ffmpeg -i {} -vcodec libx264 {}'.format('output_annotated.mp4', 'output_annoted1.mp4'))
            # fmpeg -i input_video_created_by_OpenCV.mp4 -vcodec libx264 output_video_that_streamlit_can_play.mp4
            st.video("output_annoted1.mp4")
        else:
            st.write("No video saved yet.")
