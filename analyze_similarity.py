import os
import sys
import django
from django.conf import settings

# Setup Django environment
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_blur_web.settings')
django.setup()

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from apps.videos.face_recognizer import FaceRecognizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_video_similarity(video_path, target_thumbnail_path, output_dir, sample_rate=5):
    """
    Analyzes video frames, detects faces, and calculates cosine similarity with the target thumbnail.
    Saves cropped face images with similarity score in the filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize Models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # YOLO for detection
    yolo_path = '/app/models/yolov11s-face.pt'
    if not os.path.exists(yolo_path):
        logger.error(f"YOLO model not found at {yolo_path}")
        return

    logger.info(f"Loading YOLO from {yolo_path}")
    yolo = YOLO(yolo_path)
    
    # FaceRecognizer for embedding (uses 12m model automatically)
    recognizer = FaceRecognizer(model_name='adaface', device=device)
    
    # 2. Process Target Thumbnail
    logger.info(f"Loading target thumbnail: {target_thumbnail_path}")
    target_img = cv2.imread(target_thumbnail_path)
    if target_img is None:
        logger.error("Failed to load target thumbnail")
        return
        
    target_embedding = recognizer.get_embedding(target_img)
    if target_embedding is None:
        logger.error("Failed to extract embedding from target thumbnail")
        return
        
    # 3. Process Video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Processing video: {video_path} ({total_frames} frames, {fps} fps)")
    
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sample_rate == 0:
            # Detect faces
            results = yolo(frame, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf.item())
                    
                    # Crop face
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    face_img = frame[y1:y2, x1:x2]
                    
                    # Get embedding
                    embedding = recognizer.get_embedding(face_img)
                    
                    if embedding is not None:
                        # Calculate similarity
                        sim = np.dot(embedding, target_embedding)
                        
                        # Create side-by-side comparison image
                        # Resize both to same height (e.g., 160) for better visualization
                        display_h = 160
                        
                        # Resize target
                        h_t, w_t = target_img.shape[:2]
                        scale_t = display_h / h_t
                        target_resized = cv2.resize(target_img, (int(w_t * scale_t), display_h))
                        
                        # Resize detected face
                        h_f, w_f = face_img.shape[:2]
                        scale_f = display_h / h_f
                        face_resized = cv2.resize(face_img, (int(w_f * scale_f), display_h))
                        
                        # Concatenate
                        combined = cv2.hconcat([target_resized, face_resized])
                        
                        # Add text (Similarity Score)
                        cv2.putText(combined, f"Sim: {sim:.4f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Save image
                        # Filename format: frame_{idx}_sim_{score}.jpg
                        filename = f"frame_{frame_idx:04d}_sim_{sim:.4f}.jpg"
                        save_path = os.path.join(output_dir, filename)
                        cv2.imwrite(save_path, combined)
                        
                        logger.info(f"Frame {frame_idx}: Saved {filename}")
                        saved_count += 1
                        
        frame_idx += 1
        
    cap.release()
    logger.info(f"Analysis complete. Saved {saved_count} face images to {output_dir}")

if __name__ == '__main__':
    # Configuration for Docker environment
    # Paths inside the container (mounted at /app)
    VIDEO_PATH = "/app/media/videos/b5bc6a53-e407-4ea3-95bf-89a4c5b6a46d.mp4"
    
    # Searching for a thumbnail...
    thumbnail_dir = "/app/media/faces/thumbnails"
    target_thumb = None
    
    if os.path.exists(thumbnail_dir):
        for root, dirs, files in os.walk(thumbnail_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    target_thumb = os.path.join(root, file)
                    break
            if target_thumb:
                break
            
    if target_thumb:
        OUTPUT_DIR = "/app/media/analysis_result"
        print(f"Target thumbnail found: {target_thumb}")
        print(f"Output directory: {OUTPUT_DIR}")
        analyze_video_similarity(VIDEO_PATH, target_thumb, OUTPUT_DIR)
    else:
        print("No thumbnail found in /app/media/faces/thumbnails. Please ensure a thumbnail exists.")
