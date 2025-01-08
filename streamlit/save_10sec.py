# import cv2
# import time

# def save_next_10_seconds(video_path='', output_path="output.mp4", fps=30):
#     # Initialize the video capture
#     if video_path:  # Use input video file if a valid path is provided
#         cap = cv2.VideoCapture(video_path)
#     else:  # Use the webcam or real-time feed if no path or empty string is provided
#         cap = cv2.VideoCapture(0)

#     # Check if the video opened successfully
#     if not cap.isOpened():
#         print(f"Error: Could not open video source {'webcam' if video_path == '' else video_path}.")
#         return

#     # Get the video width and height for output video
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Define the codec and create VideoWriter object to save the next 10 seconds of video as MP4
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#     event_happened = False
#     start_time = 0

#     print("Press 'e' to trigger the event and save the next 10 seconds of the video.")
    
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         if ret:
#             # Display the video feed (optional)
#             cv2.imshow('Live Feed', frame)

#             # Check if 'e' is pressed to trigger the event
#             if cv2.waitKey(1) & 0xFF == ord('e'):
#                 print("Event triggered! Starting to record the next 10 seconds.")
#                 event_happened = True
#                 start_time = time.time()

#             # If event happened, start saving frames for the next 10 seconds
#             if event_happened:
#                 elapsed_time = time.time() - start_time
#                 out.write(frame)  # Write current frame to output video
#                 if elapsed_time >= 10:  # Stop after 10 seconds
#                     print("10 seconds of video recorded.")
#                     event_happened = False  # Reset the event state

#             # Exit the loop if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break

#     # Release everything when done
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# # Example Usage:

# # To use a video file, pass its path like this:
# save_next_10_seconds(video_path="")

# import cv2
# import time

# def save_next_10_seconds(video_path='', output_path="output.mp4", fps=30):
#     # Initialize the video capture
#     if video_path:  # Use input video file if a valid path is provided
#         cap = cv2.VideoCapture(video_path)
#     else:  # Use the webcam or real-time feed if no path or empty string is provided
#         cap = cv2.VideoCapture(0)

#     # Check if the video opened successfully
#     if not cap.isOpened():
#         print(f"Error: Could not open video source {'webcam' if video_path == '' else video_path}.")
#         return

#     # Get the video width and height for output video
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Define the codec and create VideoWriter object to save the next 10 seconds of video as MP4
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#     event_happened = True  # Trigger recording immediately when this function is called
#     start_time = time.time()

#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         if ret:
#             # Display the video feed (optional)
#             cv2.imshow('Live Feed', frame)

#             # If event happened, start saving frames for the next 10 seconds
#             elapsed_time = time.time() - start_time
#             out.write(frame)  # Write current frame to output video
#             if elapsed_time >= 10:  # Stop after 10 seconds
#                 print("10 seconds of video recorded.")
#                 break

#             # Exit the loop if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break

#     # Release everything when done
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


import cv2
import time
import os

# def save_next_10_seconds(video_path=0, output_path="output.mp4", fps=15):
#     # Initialize the video capture
#     cap = cv2.VideoCapture(video_path)  # Use webcam if video_path is 0

#     # Check if the video opened successfully
#     if not cap.isOpened():
#         print(f"Error: Could not open video source {'webcam' if video_path == 0 else video_path}.")
#         return

#     # Get the video width and height for output video
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Define the codec and create VideoWriter object to save the next 10 seconds of video as MP4
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#     start_time = time.time()

#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         if ret:
#             # If event happened, start saving frames for the next 10 seconds
#             elapsed_time = time.time() - start_time
#             out.write(frame)  # Write current frame to output video
#             if elapsed_time >= 10:  # Stop after 10 seconds
#                 print("10 seconds of video recorded.")
#                 break

#             # Exit the loop if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break
#     os.system('ffmpeg -i {} -vcodec libx264 {}'.format(output_path, '10_sec_saved.mp4'))
#     # Release everything when done
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

def save_stream_for_duration(video_source, output_path="output.mp4", duration=10, fps=30):
    # Initialize the video capture
    cap = cv2.VideoCapture(video_source)

    # Check if the video source opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}.")
        return

    # Get the video width and height for the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object to save the video as MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            # Write the current frame to the output video
            out.write(frame)

            # Stop recording after the specified duration
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                print(f"{duration} seconds of video recorded.")
                break

            # Optional: Exit the loop early if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    os.system('ffmpeg -i {} -vcodec libx264 {}'.format(output_path, '10_sec_saved.mp4'))        
    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()