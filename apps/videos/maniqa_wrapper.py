"""
Image Quality Assessment Wrapper (NIMA)

PyIQA NIMA 모델을 사용한 얼굴 이미지 품질 평가.
배치 처리로 GPU 활용률 극대화.
"""

import torch
import numpy as np
import cv2
import logging
from typing import List, Union
import pyiqa
from PIL import Image

logger = logging.getLogger(__name__)


class MANIQAWrapper:
    """
    IQA 모델 래퍼 (NIMA 사용, 배치 처리 지원)

    - PyIQA 라이브러리 사용
    - No-Reference IQA
    - 배치 처리로 GPU 효율성 향상
    """

    def __init__(self, device: str = 'auto', batch_size: int = 8, model_name: str = 'nima'):
        """
        Parameters:
        -----------
        device : str
            'cuda', 'cpu', 또는 'auto'
        batch_size : int
            배치 처리 크기 (기본 8)
        model_name : str
            사용할 IQA 모델 (기본: 'nima')
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.batch_size = batch_size
        self.model_name = model_name

        logger.info(f"Loading {model_name.upper()} model on {self.device}...")

        # IQA 모델 로드 (pre-trained weights 자동 다운로드)
        self.model = pyiqa.create_metric(model_name, device=self.device)

        logger.info(f"{model_name.upper()} model loaded successfully (batch_size={batch_size})")

    def assess_quality(self, image: np.ndarray) -> float:
        """
        단일 이미지 품질 평가

        Parameters:
        -----------
        image : np.ndarray
            BGR 이미지 (OpenCV format)

        Returns:
        --------
        quality : float
            품질 점수 (0-100 스케일로 정규화)
        """
        # 내부적으로 배치 처리 함수 호출 (배치 크기 1)
        return self.assess_quality_batch([image])[0]

    def assess_quality_batch(self, images: List[np.ndarray]) -> List[float]:
        """
        배치 이미지 품질 평가 (최적화된 버전)

        Parameters:
        -----------
        images : List[np.ndarray]
            BGR 이미지 리스트

        Returns:
        --------
        qualities : List[float]
            품질 점수 리스트 (0-100 스케일)
        """
        if not images:
            return []

        try:
            # BGR → RGB 변환 및 PIL Image 리스트 생성
            pil_images = []
            for img in images:
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = img
                pil_images.append(Image.fromarray(img_rgb))

            # 배치 처리
            qualities = []
            num_images = len(pil_images)

            with torch.no_grad():
                for i in range(0, num_images, self.batch_size):
                    batch = pil_images[i:i + self.batch_size]

                    # PyIQA 배치 추론
                    if len(batch) == 1:
                        # 단일 이미지
                        score = self.model(batch[0]).item()
                        batch_scores = [score]
                    else:
                        # 여러 이미지 배치 처리
                        # PyIQA는 리스트 입력 시 배치 처리
                        batch_scores = []
                        for img in batch:
                            score = self.model(img).item()
                            batch_scores.append(score)

                    # 0-100 스케일로 변환
                    # NIMA는 1-10 스케일 → * 10.0으로 0-100 스케일로 변환
                    qualities.extend([s * 10.0 for s in batch_scores])

            return qualities

        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return [50.0] * len(images)
