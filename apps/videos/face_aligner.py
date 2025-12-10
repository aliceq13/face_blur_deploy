"""
Face Alignment Module using CVLface DFA (Deep Face Alignment)

This module provides landmark-based face alignment to normalize face orientation
before embedding extraction, significantly improving recognition accuracy.
"""

import os
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import logging

logger = logging.getLogger(__name__)


class FaceAligner:
    """
    Face alignment using CVLface DFA MobileNet model.
    Aligns faces to canonical 112x112 position using facial landmarks.
    """
    
    def __init__(self, device='auto', model_id='minchul/cvlface_DFA_mobilenet'):
        """
        Initialize Face Aligner.
        
        Args:
            device: 'cuda', 'cpu', or 'auto'
            model_id: HuggingFace model ID for alignment model
        """
        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model_id = model_id
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load alignment model from HuggingFace using CVLface utils"""
        try:
            import sys
            from pathlib import Path

            # Add CVLface to Python path
            project_root = Path(__file__).resolve().parents[2]
            cvlface_path = project_root / 'CVLface' / 'cvlface'
            run_v1_path = cvlface_path / 'research' / 'recognition' / 'code' / 'run_v1'

            if not cvlface_path.exists():
                logger.warning(f"CVLface directory not found at {cvlface_path}")
                logger.warning("Falling back to simple resize alignment")
                self.model = None
                return

            # Add both CVLface and run_v1 to Python path (for aligners module)
            sys.path.insert(0, str(cvlface_path.parent))
            if run_v1_path.exists():
                sys.path.insert(0, str(run_v1_path))

            # Import CVLface utilities
            try:
                from cvlface.general_utils.huggingface_model_utils import load_model_by_repo_id

                # Download and load model
                cache_dir = os.path.expanduser(f'~/.cvlface_cache/{self.model_id}')
                hf_token = os.environ.get('HF_TOKEN', None)

                logger.info(f"Loading CVLface DFA model from {self.model_id}...")
                self.model = load_model_by_repo_id(
                    repo_id=self.model_id,
                    save_path=cache_dir,
                    HF_TOKEN=hf_token
                )

                self.model.to(self.device)
                self.model.eval()
                logger.info(f"✅ FaceAligner loaded successfully on {self.device}")

            except ImportError as e:
                logger.warning(f"Could not import CVLface utilities: {e}")
                logger.warning("Falling back to simple resize alignment")
                self.model = None
            finally:
                # Remove CVLface paths from sys.path
                if str(run_v1_path) in sys.path:
                    sys.path.remove(str(run_v1_path))
                if str(cvlface_path.parent) in sys.path:
                    sys.path.remove(str(cvlface_path.parent))

        except Exception as e:
            logger.warning(f"Failed to load alignment model: {e}")
            logger.warning("Falling back to simple resize alignment")
            self.model = None
    
    def _pil_to_input(self, pil_image):
        """Convert PIL image to model input tensor"""
        trans = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return trans(pil_image).unsqueeze(0).to(self.device)
    
    def _visualize(self, tensor):
        """Convert tensor back to PIL image"""
        # Denormalize
        tensor = tensor.clone()
        tensor = tensor * 0.5 + 0.5  # Reverse normalization
        tensor = tensor.clamp(0, 1)
        
        # To PIL
        img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)
    
    def align_face_with_landmarks(self, face_img_bgr):
        """
        Align face and return landmarks for KPRPE model.
        
        DFA_mobilenet은 원본 크기 이미지를 입력으로 받아 내부적으로 처리합니다.
        강제 리사이즈 없이 원본 크기를 유지하여 landmarks 예측 정확도를 높입니다.
        
        Args:
            face_img_bgr: Face image in BGR (충분한 마진 포함, 원본 크기 유지)
        
        Returns:
            tuple: (aligned_rgb, aligned_ldmks)
                - aligned_rgb: 정렬된 112x112 RGB 이미지 (numpy array)
                - aligned_ldmks: 정렬된 이미지에서의 5-point landmarks (tensor, shape: 1,5,2)
                  None if fallback is used
        """
        if self.model is None:
            return self._fallback_align(face_img_bgr), None
        
        try:
            # Convert BGR to RGB
            face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
            
            # ⭐ 핵심 변경: 강제 리사이즈 제거
            # DFA_mobilenet은 내부 preprocessor가 적절히 처리함
            # 원본 크기를 유지하여 landmarks 예측 정확도 향상
            
            # Convert to PIL (원본 크기 유지)
            pil_img = Image.fromarray(face_img_rgb)
            
            # Prepare input
            input_tensor = self._pil_to_input(pil_img)
            
            # Align
            with torch.no_grad():
                aligned_tensor, orig_pred_ldmks, aligned_ldmks, score, thetas, normalized_bbox = self.model(input_tensor)
            
            # Convert back to numpy (RGB, 112x112)
            aligned_pil = self._visualize(aligned_tensor)
            aligned_rgb = np.array(aligned_pil)
            
            return aligned_rgb, aligned_ldmks
            
        except Exception as e:
            logger.warning(f"Alignment failed: {e}, falling back to resize")
            return self._fallback_align(face_img_bgr), None
    
    def align_face(self, face_img_bgr):
        """
        Align face to canonical position (호환성 유지용).
        
        Args:
            face_img_bgr: Face image in BGR format (numpy array)
            
        Returns:
            aligned_face_rgb: Aligned face in RGB format (112x112 numpy array)
        """
        aligned_rgb, _ = self.align_face_with_landmarks(face_img_bgr)
        return aligned_rgb
    
    def _fallback_align(self, face_img_bgr):
        """Fallback: simple resize to 112x112"""
        face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        if face_img_rgb.shape[:2] != (112, 112):
            face_img_rgb = cv2.resize(face_img_rgb, (112, 112))
        return face_img_rgb


# ============================================================================
# Global aligner instance and convenience function
# ============================================================================

_global_aligner = None


def get_aligner():
    """Get or create global FaceAligner instance (singleton pattern)"""
    global _global_aligner
    if _global_aligner is None:
        _global_aligner = FaceAligner(device='auto')
    return _global_aligner


def align_face(face_img_bgr):
    """
    Convenience function for face alignment.

    Args:
        face_img_bgr: Face image in BGR format (numpy array)

    Returns:
        aligned_face_rgb: Aligned face in RGB format (112x112 numpy array)

    Note:
        Uses a singleton FaceAligner instance for efficiency.
    """
    aligner = get_aligner()
    return aligner.align_face(face_img_bgr)


def align_face_with_landmarks(face_img_bgr):
    """
    Convenience function for face alignment with landmarks.

    Args:
        face_img_bgr: Face image in BGR format (numpy array, 충분한 마진 포함)

    Returns:
        tuple: (aligned_rgb, aligned_ldmks)
            - aligned_rgb: Aligned face in RGB format (112x112 numpy array)
            - aligned_ldmks: 5-point landmarks in aligned image (tensor) or None

    Note:
        Uses a singleton FaceAligner instance for efficiency.
    """
    aligner = get_aligner()
    return aligner.align_face_with_landmarks(face_img_bgr)
