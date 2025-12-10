"""
AdaFace Wrapper with KPRPE Support
===================================

CVLface HuggingFace KPRPE 모델을 사용한 얼굴 임베딩 추출.
KPRPE (KeyPoint Relative Position Encoding)를 통해 affine transform에 강건한 임베딩을 생성합니다.
"""

import torch
import cv2
import numpy as np
import os
import sys
import logging
import inspect
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

logger = logging.getLogger(__name__)

# CVLface 경로 추가
CVLFACE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'CVLface')
if CVLFACE_PATH not in sys.path:
    sys.path.insert(0, CVLFACE_PATH)


class AdaFaceWrapper:
    """
    AdaFace Wrapper with KPRPE (KeyPoint Relative Position Encoding) Support
    
    KPRPE 모델을 사용하면 얼굴의 회전, 스케일, 이동에 더 강건한 임베딩을 추출할 수 있습니다.
    """
    
    def __init__(self, model_path=None, device='cuda', architecture='vit', use_alignment=True, use_kprpe=True):
        """
        Initialize AdaFace Wrapper
        
        Args:
            model_path: 로컬 모델 경로 (KPRPE 모드에서는 fallback용)
            device: 'cuda' or 'cpu'
            architecture: 'vit' or 'ir50'
            use_alignment: FaceAligner 사용 여부
            use_kprpe: True이면 CVLface HuggingFace KPRPE 모델 사용
        """
        # GPU fallback
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = 'cpu'
        self.device = device
        self.use_kprpe = use_kprpe
        self.use_alignment = use_alignment
        self.model_path = model_path
        self.supports_keypoints = False
        
        # HuggingFace 토큰 (환경변수에서 읽기, 없어도 됨)
        self.hf_token = os.environ.get('HF_TOKEN', None)
        
        # Face Aligner (landmarks 추출용)
        self.aligner = None
        if self.use_alignment:
            try:
                from .face_aligner import FaceAligner
                self.aligner = FaceAligner(device=self.device)
                logger.info("✅ FaceAligner initialized successfully - Face alignment is ENABLED")
                print("✅ FaceAligner initialized successfully - Face alignment is ENABLED")
            except Exception as e:
                logger.warning(f"❌ FaceAligner initialization failed: {e}")
                print(f"❌ FaceAligner initialization failed: {e}")
                self.aligner = None
        
        # 모델 로드
        if self.use_kprpe:
            self._load_kprpe_model()
        else:
            self._load_legacy_model(model_path, architecture)
    
    def _load_kprpe_model(self):
        """CVLface HuggingFace KPRPE 모델 로드"""
        try:
            from cvlface.general_utils.huggingface_model_utils import load_model_by_repo_id
            
            # KPRPE 모델 로드 (WebFace4M)
            repo_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface12m'
            cache_path = os.path.expanduser(f'~/.cvlface_cache/{repo_id}')
            
            logger.info(f"Loading CVLface KPRPE model from {repo_id}...")
            print(f"Loading CVLface KPRPE model from {repo_id}...")
            
            self.model = load_model_by_repo_id(
                repo_id=repo_id,
                save_path=cache_path,
                HF_TOKEN=self.hf_token
            ).to(self.device)
            self.model.eval()
            
            # KPRPE 지원 여부 확인
            try:
                input_signature = inspect.signature(self.model.model.net.forward)
                self.supports_keypoints = input_signature.parameters.get('keypoints') is not None
            except Exception:
                self.supports_keypoints = True  # KPRPE 모델은 기본 지원
            
            logger.info(f"✅ CVLface KPRPE model loaded (keypoints support: {self.supports_keypoints})")
            print(f"✅ CVLface KPRPE model loaded on {self.device} (keypoints support: {self.supports_keypoints})")
            
        except Exception as e:
            logger.error(f"KPRPE model loading failed: {e}, falling back to legacy mode")
            print(f"⚠️ KPRPE model loading failed: {e}")
            print("⚠️ Falling back to legacy model...")
            self.use_kprpe = False
            self._load_legacy_model(self.model_path, 'vit')
    
    def _load_legacy_model(self, model_path, architecture):
        """기존 로컬 모델 로드 (fallback)"""
        from .adaface.iresnet import iresnet50
        from .adaface.vit import vit_base
        
        self.supports_keypoints = False
        
        if model_path is None:
            model_path = '/app/apps/videos/weights/adaface_vit_base_kprpe_webface4m.pt'
        
        if architecture == 'vit' or (model_path and 'vit' in os.path.basename(model_path)):
            self.model = vit_base(num_classes=0)
            print("AdaFaceWrapper: Using ViT-Base architecture (Legacy)")
        else:
            self.model = iresnet50(num_features=512)
            print("AdaFaceWrapper: Using IR-50 architecture (Legacy)")
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                
                new_state_dict = {}
                for k, v in state_dict.items():
                    key = k[7:] if k.startswith('module.') else k
                    new_state_dict[key] = v
                
                self.model.load_state_dict(new_state_dict, strict=False)
                print(f"Legacy AdaFace model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading legacy model: {e}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _numpy_to_tensor(self, face_img_rgb):
        """numpy RGB 이미지를 모델 입력 텐서로 변환"""
        pil_img = Image.fromarray(face_img_rgb)
        trans = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return trans(pil_img).unsqueeze(0).to(self.device)
    
    def get_embedding(self, face_img, skip_alignment=False, keypoints=None):
        """
        Extract embedding from face image.
        
        Args:
            face_img: BGR image (numpy array)
            skip_alignment: If True, assume face_img is already aligned 112x112
            keypoints: Optional pre-computed keypoints tensor (1, 5, 2)
        
        Returns:
            embedding: 512-d numpy array (normalized)
        """
        if face_img is None or face_img.size == 0:
            return None
        
        try:
            aligned_ldmks = keypoints
            
            if not skip_alignment and self.aligner is not None:
                # DFA alignment + landmarks 추출
                aligned_rgb, aligned_ldmks = self.aligner.align_face_with_landmarks(face_img)
                face_tensor = self._numpy_to_tensor(aligned_rgb)
            else:
                # Already aligned
                if face_img.shape[:2] != (112, 112):
                    face_img = cv2.resize(face_img, (112, 112))
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_tensor = self._numpy_to_tensor(face_img_rgb)
            
            # Inference
            with torch.no_grad():
                if self.use_kprpe and self.supports_keypoints and aligned_ldmks is not None:
                    # KPRPE: keypoints 전달
                    if isinstance(aligned_ldmks, np.ndarray):
                        aligned_ldmks = torch.from_numpy(aligned_ldmks).float().to(self.device)
                    embedding = self.model(face_tensor, aligned_ldmks)
                else:
                    # Standard
                    embedding = self.model(face_tensor)
            
            # L2 정규화
            embedding = embedding.cpu().numpy()[0]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"AdaFace inference failed: {e}")
            print(f"AdaFace inference failed: {e}")
            return None
