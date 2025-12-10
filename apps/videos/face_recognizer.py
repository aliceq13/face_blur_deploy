import os
import logging
import numpy as np
import cv2
import torch
from django.conf import settings
# from insightface.app import FaceAnalysis  <-- Removed top-level import
from .adaface_wrapper import AdaFaceWrapper

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """
    Unified Face Recognizer that supports both ArcFace (buffalo_l) and AdaFace.
    Controlled by settings.FACE_RECOGNITION_MODEL.
    """
    def __init__(self, model_name='arcface', device='auto'):
        self.model_name = model_name
        self.device = device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        logger.info(f"Initializing FaceRecognizer with model: {self.model_name}, device: {self.device}")
        
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            if self.model_name == 'arcface':
                # Lazy import for InsightFace
                from insightface.app import FaceAnalysis
                
                # 1. 공통: InsightFace Detection 모델 로드 (Alignment용)
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if self.device == 'cpu':
                    providers = ['CPUExecutionProvider']
                
                # Detection만 사용하더라도 'recognition' 모듈이 없으면 get()에서 에러가 날 수 있으므로 둘 다 로드
                self.detector = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'], providers=providers)
                ctx_id = 0 if self.device == 'cuda' else -1
                # det_thresh를 0.3으로 낮춰서 옆모습 검출 확률 높임
                self.detector.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.3)
                logger.info("Face Detector (InsightFace) initialized for alignment")

                # ArcFace는 self.detector를 그대로 사용
                self.model = self.detector
                logger.info("ArcFace (buffalo_l) initialized successfully")
                
            elif self.model_name == 'adaface':
                # AdaFace 초기화
                # NOTE: InsightFace alignment은 제거 (속도 향상)
                # Initialize AdaFace model
                base_dir = os.path.dirname(os.path.abspath(__file__))
                # Try ViT-12M first (best performance)
                weight_path = os.path.join(base_dir, 'weights', 'adaface_vit_base_kprpe_webface12m.pt')

                if not os.path.exists(weight_path):
                    # Fallback to ViT-4M
                    weight_path_4m = os.path.join(base_dir, 'weights', 'adaface_vit_base_kprpe_webface4m.pt')
                    if os.path.exists(weight_path_4m):
                        weight_path = weight_path_4m
                        logger.info("Using ViT-4M model (12M not found)")
                    else:
                        # Try .ckpt extension
                        ckpt_path = os.path.join(base_dir, 'weights', 'adaface_vit_base_kprpe_webface12m.ckpt')
                        if os.path.exists(ckpt_path):
                            weight_path = ckpt_path
                        else:
                            logger.warning(f"ViT weights not found, falling back to IR-50")
                            weight_path = os.path.join(base_dir, 'weights', 'adaface_ir50_ms1mv2.ckpt')

                self.model = AdaFaceWrapper(weight_path, device=self.device)
                logger.info(f"AdaFace initialized from {weight_path}")
                
            else:
                raise ValueError(f"Unknown model_name: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize FaceRecognizer ({self.model_name}): {e}")
            self.model = None
            raise e

    def get_embedding(self, face_img):
        """
        Extract embedding from face image.
        Args:
            face_img: BGR image (numpy array)
        Returns:
            embedding: 512-d (ArcFace) or 512/768-d (AdaFace) numpy array
        """
        if self.model is None:
            logger.error("Model is not initialized")
            return None
            
        if face_img is None or face_img.size == 0:
            return None

        try:
            if self.model_name == 'arcface':
                # ArcFace inference
                # 1. Try standard detection & alignment
                faces = self.detector.get(face_img)
                if faces:
                    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                    return faces[0].embedding
                
                # 2. Fallback: Direct extraction without detection
                return self._get_fallback_embedding_arcface(face_img)
                    
            elif self.model_name == 'adaface':
                # AdaFace inference
                # NOTE: InsightFace alignment은 이미 YOLO로 크롭된 작은 이미지에서
                # 얼굴을 다시 찾으려고 하므로 실패율이 높고 느립니다.
                # 따라서 AdaFace에 직접 전달하여 속도를 우선시합니다.
                return self.model.get_embedding(face_img, skip_alignment=False)
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None

    def _get_fallback_embedding_arcface(self, face_img):
        # ArcFace Fallback Logic
        try:
            if face_img.size > 0:
                blob = cv2.resize(face_img, (112, 112))
                blob = blob.astype(np.float32)
                blob = (blob - 127.5) / 128.0
                blob = np.transpose(blob, (2, 0, 1))
                blob = np.expand_dims(blob, axis=0)
                
                rec_model = self.detector.models.get('recognition')
                if rec_model:
                    input_name = rec_model.session.get_inputs()[0].name
                    output_name = rec_model.session.get_outputs()[0].name
                    embedding = rec_model.session.run([output_name], {input_name: blob})[0][0]
                    
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding /= norm
                    return embedding
        except Exception:
            pass
        return None
