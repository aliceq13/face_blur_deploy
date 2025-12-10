import os
import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Mock Django settings if needed
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        MEDIA_ROOT=os.path.join(os.getcwd(), 'media'),
        YOLO_FACE_MODEL_PATH=os.path.join(os.getcwd(), 'models', 'yolov11s-face.pt'),
        LOGGING_CONFIG=None
    )
    django.setup()

from apps.videos.video_blurring import VideoBlurrer

def create_dummy_video(filename, duration=1, fps=30):
    """Create a dummy video with a moving rectangle (simulating a face)"""
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for i in range(duration * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a moving rectangle
        x = int(i * 5) % (width - 50)
        y = 200
        cv2.rectangle(frame, (x, y), (x+50, y+50), (255, 255, 255), -1)
        
        out.write(frame)
        
    out.release()
    return filename

def test_video_blurring():
    print("Testing VideoBlurrer...")
    
    # 1. Create dummy video
    video_path = "test_input.mp4"
    create_dummy_video(video_path)
    print(f"Created dummy video: {video_path}")
    
    output_path = "test_output.mp4"
    
    # 2. Initialize VideoBlurrer
    # Note: We need a real YOLO model path. Assuming it exists or we mock it.
    # If YOLO model doesn't exist, we might need to skip or mock.
    yolo_path = settings.YOLO_FACE_MODEL_PATH
    if not os.path.exists(yolo_path):
        print(f"Warning: YOLO model not found at {yolo_path}. Skipping actual processing test.")
        # Create a dummy file to pass the check if we were just testing logic, 
        # but VideoBlurrer loads it immediately.
        # So we will just return here.
        return

    blurrer = VideoBlurrer(yolo_model_path=yolo_path, device='cpu')
    
    # 3. Run process_video
    # We pass empty face_models, so it should treat all faces as unknown -> blur them (based on logic)
    face_models = [] 
    
    success = blurrer.process_video(
        video_path=video_path,
        output_path=output_path,
        face_models=face_models,
        progress_callback=lambda x: print(f"Progress: {x}%")
    )
    
    if success:
        print("Video processing successful!")
        if os.path.exists(output_path):
            print(f"Output file created: {output_path}")
        else:
            print("Error: Output file not found.")
    else:
        print("Video processing failed.")
        
    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(output_path):
        os.remove(output_path)

if __name__ == "__main__":
    test_video_blurring()
