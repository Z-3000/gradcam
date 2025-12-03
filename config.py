"""
설정 모듈 (Configuration)
=========================

프로젝트 전역 설정을 중앙 관리.
모델 추가/수정 시 이 파일만 변경하면 됨.

확장 예시:
- 새 모델 추가: MODEL_CONFIG에 항목 추가
- threshold 조정: BEST_THRESHOLDS 수정
- 새 클래스 추가: CLASS_NAMES 수정
"""

from typing import Dict, Any
import torch
from torchvision import models

# ============================================================
# 디바이스 설정
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 클래스 정의
# ============================================================
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Pneumonia"]

# ============================================================
# 모델 설정
# ============================================================
# 새 모델 추가 시 이 딕셔너리에 항목 추가
# - builder: torchvision 모델 빌더 함수
# - checkpoint: 체크포인트 파일 경로
# - target_layer: Grad-CAM 타겟 레이어 이름
# - input_size: 입력 이미지 크기

MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "ResNet18": {
        "builder": models.resnet18,
        "checkpoint": "checkpoints/251115_resnet18_NP.pth",
        "target_layer": "layer4",
        "input_size": 224,
    },
    "EfficientNet-B0": {
        "builder": models.efficientnet_b0,
        "checkpoint": "checkpoints/251115_efficientnet_NP.pth",
        "target_layer": "features",
        "input_size": 224,
    },
    "DenseNet121": {
        "builder": models.densenet121,
        "checkpoint": "checkpoints/251115_densenet121_NP.pth",
        "target_layer": "features",
        "input_size": 224,
    },
    # 확장 예시:
    # "VGG16": {
    #     "builder": models.vgg16,
    #     "checkpoint": "checkpoints/vgg16_NP.pth",
    #     "target_layer": "features",
    #     "input_size": 224,
    # },
}

# ============================================================
# 모델별 최적 임계값
# ============================================================
# 학습 시 ROC 분석 기반 최적값
# 의료 진단 특성상 Recall 우선 → 낮은 threshold 사용

BEST_THRESHOLDS: Dict[str, float] = {
    "ResNet18": 0.25,
    "EfficientNet-B0": 0.15,
    "DenseNet121": 0.18,
}

DEFAULT_THRESHOLD = 0.5  # 미등록 모델용 기본값


def get_best_threshold(model_name: str) -> float:
    """
    모델별 최적 threshold 반환

    Args:
        model_name: 모델 이름

    Returns:
        해당 모델의 최적 threshold (미등록 모델은 DEFAULT_THRESHOLD)
    """
    return BEST_THRESHOLDS.get(model_name, DEFAULT_THRESHOLD)


# ============================================================
# DICOM 전처리 설정
# ============================================================
# 학습 시 사용한 파라미터와 동일하게 유지해야 성능 보존

DICOM_WINDOW_CENTER = 40.0   # 폐 조직 대표 HU 값
DICOM_WINDOW_WIDTH = 800.0   # 관찰 범위 (-360 ~ 440 HU)

# ============================================================
# 이미지 정규화 설정 (ImageNet 표준)
# ============================================================
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
