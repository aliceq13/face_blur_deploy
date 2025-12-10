
import logging
import numpy as np
import cv2
import torch
from insightface.app import FaceAnalysis

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu():
    print("="*50)
    print("Checking GPU Availability...")
    print(f"Torch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current Device: {torch.cuda.get_device_name(0)}")
    print("="*50)

    try:
        print("\nInitializing InsightFace...")
        # providers 설정 확인
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        app = FaceAnalysis(name='buffalo_l', providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        print("\nInsightFace Initialized.")
        
        # 실제 모델이 어떤 Provider를 쓰고 있는지 확인 (ONNX Runtime)
        # Detection Model
        det_model = app.models['detection']
        print(f"\nDetection Model Providers: {det_model.session.get_providers()}")
        
        # Recognition Model
        rec_model = app.models['recognition']
        print(f"Recognition Model Providers: {rec_model.session.get_providers()}")
        
        if 'CUDAExecutionProvider' in det_model.session.get_providers():
            print("\nSUCCESS: InsightFace is using GPU!")
        else:
            print("\nWARNING: InsightFace is NOT using GPU. It fell back to CPU.")
            
    except Exception as e:
        print(f"\nError occurred: {e}")

if __name__ == "__main__":
    check_gpu()
