import os

import cv2


def get_frames_at_timestamps(video_path, timestamps, output_dir):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return False

    # Get the video's frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not get the frame rate.")
        return False

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for timestamp in timestamps:
        # Calculate the frame number at the given timestamp
        frame_number = int(fps * timestamp)

        # Set the video position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame at timestamp {timestamp}.")
            continue

        # Create a filename for the output image
        output_image_path = os.path.join(output_dir, f'frame_at_{timestamp:.2f}_seconds.png')

        # Save the frame to the file
        cv2.imwrite(output_image_path, frame)
        print(f"Saved frame at {timestamp} seconds to {output_image_path}")

    # Release the video capture object
    cap.release()

    return True


def cut_video(video_path, start_time, end_time, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Calculate the frame numbers for the start and end times
    start_frame = int(fps * start_time)
    end_frame = int(fps * end_time)

    # Set up the video writer for the output video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Set the video position to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame

    # Read and write frames until the end frame is reached
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    # Release the video capture and writer objects
    cap.release()
    out.release()

    print(f"Video cut from {start_time} to {end_time} seconds saved as {output_path}")
    return True


def save_frames_between_timestamps(video_path, start_time, end_time, resolution, output_dir):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not get the frame rate.")
        return False

    # Calculate the frame numbers for the start and end times
    start_frame = int(fps * start_time)
    end_frame = int(fps * end_time)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set the video position to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame

    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        # Create a filename for the output image
        output_image_path = os.path.join(output_dir, f'frame_{current_frame:06d}.png')
        # Save the frame to the file
        if current_frame % resolution == 0:
            cv2.imwrite(output_image_path, frame)
            print(f"Saved frame {current_frame} to {output_image_path}")
        current_frame += 1
    # Release the video capture object
    cap.release()
    return True


output_dir = 'output_frames'
output_file = 'HD CCTV Camera_cut.mp4'
input_file = 'HD CCTV Camera_cut.mp4'

# get_frames_at_timestamps('HD CCTV Camera.mp4', [ts for ts in range(30)], output_dir)
# cut_video('HD CCTV Camera.mp4', 15, 45, output_file)
save_frames_between_timestamps(input_file, 5, 13, 4, output_dir)
