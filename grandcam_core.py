"""
X-ray Grad-CAM 백엔드 모듈
==========================

Streamlit 앱(main.py)과 분리된 비즈니스 로직 모듈.
FastAPI, CLI 등 다른 프레임워크에서도 재사용 가능.

주요 기능:
- 3개 모델(ResNet18 / EfficientNet-B0 / DenseNet121) 로드 및 캐싱
- DICOM/일반 이미지 → PIL → Tensor 변환
- Grad-CAM heatmap + overlay 생성

설계 원칙:
- UI와 로직 분리 (관심사 분리)
- 설정은 config.py에서 중앙 관리
- 모듈 간 의존성 최소화

Author: JH3907
License: MIT
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, List, Optional

import io
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import numpy as np
import cv2
from PIL import Image

# 설정 모듈에서 import
from config import (
    DEVICE,
    MODEL_CONFIG,
    CLASS_NAMES,
    get_best_threshold,
    DICOM_WINDOW_CENTER,
    DICOM_WINDOW_WIDTH,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)

# 선택적 의존성: pydicom (DICOM 지원)
try:
    import pydicom  # type: ignore
except ImportError:
    pydicom = None  # DICOM 없는 환경에서도 동작 가능


# ============================================================
# 1. 이미지 로딩 유틸리티
# ============================================================

def dicom_to_pil(file_bytes: bytes) -> Tuple[Image.Image, Any]:
    """
    DICOM 바이트를 PIL Image로 변환

    처리 과정:
    1. HU 변환: RescaleSlope/Intercept 적용 → 물리적 단위(CT 밀도)로 변환
    2. Lung window 적용: config.py의 설정값 사용
    3. 정규화: 0~255 스케일로 변환

    Args:
        file_bytes: DICOM 파일 바이트

    Returns:
        (PIL Image RGB, DICOM Dataset) 튜플

    Note:
        학습 시 사용한 전처리와 동일한 파라미터 사용 (성능 보존)
    """
    if pydicom is None:
        raise ImportError("pydicom이 설치되어 있지 않습니다. pip install pydicom")

    dcm = pydicom.dcmread(io.BytesIO(file_bytes))
    img = dcm.pixel_array.astype(np.float32)

    # HU 변환
    if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
        slope = float(dcm.RescaleSlope)
        intercept = float(dcm.RescaleIntercept)
        img = img * slope + intercept

    # Windowing (config.py 설정값 사용)
    min_val = DICOM_WINDOW_CENTER - DICOM_WINDOW_WIDTH / 2.0
    max_val = DICOM_WINDOW_CENTER + DICOM_WINDOW_WIDTH / 2.0

    img = (img - min_val) / (max_val - min_val)
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)

    pil_img = Image.fromarray(img).convert("RGB")
    return pil_img, dcm


def image_bytes_to_pil(file_bytes: bytes) -> Image.Image:
    """
    PNG/JPEG 이미지를 PIL Image(RGB)로 변환

    Args:
        file_bytes: 이미지 파일 바이트

    Returns:
        PIL Image (RGB 모드)
    """
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ============================================================
# 2. Grad-CAM 구현
# ============================================================

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) 구현

    동작 원리:
    1. Forward pass: target layer의 activation 저장
    2. Backward pass: target class에 대한 gradient 저장
    3. CAM 계산: gradient의 global average → 채널별 가중치
                 가중치 * activation → 합산 → ReLU

    사용 예시:
        gradcam = GradCAM(model, target_layer)
        cam, target_class, target_prob, logits = gradcam(x, threshold=0.5)

    Reference:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization", ICCV 2017
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Forward hook: activation 저장
        def forward_hook(module, inputs, output):
            self.activations = output.detach()
            output.register_hook(self._save_gradients)

        self.target_layer.register_forward_hook(forward_hook)

    def _save_gradients(self, grad: torch.Tensor):
        """Backward hook: gradient 저장"""
        self.gradients = grad.detach()

    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """
        Grad-CAM 계산 수행

        Args:
            x: 입력 텐서 (1, C, H, W)
            target_class: 타겟 클래스 (None이면 예측 클래스 사용)
            threshold: 폐렴 분류 임계값

        Returns:
            cam: Grad-CAM 히트맵 (H, W), 0~1 정규화
            target_class: 예측 클래스 (0=Normal, 1=Pneumonia)
            target_prob: 타겟 클래스 확률
            logits: 모델 출력 로짓
        """
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        logits = self.model(x)

        # 이진 분류 모델: 출력이 1개 (Pneumonia에 대한 로짓)
        if logits.shape[1] == 1:
            prob_pos = torch.sigmoid(logits)[0, 0]

            if target_class is None:
                target_class = int((prob_pos >= threshold).item())

            score = logits[0, 0]
            probs_vec = torch.stack([1 - prob_pos, prob_pos], dim=0)

        else:
            # 다중 클래스 모델 (확장 대비)
            probs_vec = F.softmax(logits, dim=1)[0]
            if target_class is None:
                target_class = int(torch.argmax(probs_vec).item())
            score = logits[0, target_class]

        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook이 제대로 작동하지 않았습니다.")

        act = self.activations
        grad = self.gradients

        # Global average pooling → 채널별 가중치
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # 정규화
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.squeeze().cpu().numpy()
        target_prob = float(probs_vec[target_class].item())
        logits_np = logits.detach().cpu().numpy()[0]

        return cam_np, target_class, target_prob, logits_np


# ============================================================
# 3. 모델 로딩 / 전처리
# ============================================================

def _build_model(model_name: str) -> nn.Module:
    """
    MODEL_CONFIG에 맞춰 모델 아키텍처 생성

    Args:
        model_name: 모델 이름

    Returns:
        출력 레이어가 1개 뉴런으로 수정된 모델
    """
    cfg = MODEL_CONFIG[model_name]
    builder = cfg["builder"]

    model = builder(weights=None)

    # 출력 레이어를 1개 뉴런으로 수정 (이진 분류)
    if isinstance(model, models.ResNet):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    elif isinstance(model, models.EfficientNet):
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
    elif isinstance(model, models.DenseNet):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {type(model)}")

    return model


def _load_state_dict(model: nn.Module, ckpt_path: str) -> None:
    """
    체크포인트에서 가중치 로드

    Args:
        model: 로드할 모델
        ckpt_path: 체크포인트 파일 경로
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")

    # PyTorch 버전 호환성 처리
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # 다양한 체크포인트 포맷 대응
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)


# 모델 캐싱
_LOADED_MODELS: Dict[str, Dict[str, Any]] = {}


def get_model_bundle(model_name: str) -> Dict[str, Any]:
    """
    모델 번들(model, target_layer, transform) 로드 및 캐싱

    Args:
        model_name: 모델 이름

    Returns:
        {"model": model, "target_layer": layer, "transform": transform}
    """
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    if model_name not in MODEL_CONFIG:
        raise KeyError(f"지원하지 않는 모델: {model_name}")

    cfg = MODEL_CONFIG[model_name]
    ckpt_path = cfg["checkpoint"]
    target_layer_name = cfg["target_layer"]
    input_size = cfg["input_size"]

    model = _build_model(model_name)
    _load_state_dict(model, ckpt_path)
    model.to(DEVICE)
    model.eval()

    # Target layer 추출
    named_modules = dict(model.named_modules())
    if target_layer_name not in named_modules:
        raise ValueError(f"target_layer '{target_layer_name}'를 찾을 수 없습니다.")
    target_layer = named_modules[target_layer_name]

    # Transform 생성 (config.py 설정값 사용)
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    bundle = {
        "model": model,
        "target_layer": target_layer,
        "transform": transform,
    }
    _LOADED_MODELS[model_name] = bundle
    return bundle


def preprocess_pil(img: Image.Image, transform: transforms.Compose) -> torch.Tensor:
    """
    PIL Image를 모델 입력 텐서로 변환

    Args:
        img: PIL Image
        transform: 전처리 transform

    Returns:
        (1, C, H, W) 텐서
    """
    x = transform(img)
    x = x.unsqueeze(0)
    return x.to(DEVICE)


def apply_colormap_on_image(
    gray_cam: np.ndarray,
    rgb_img: np.ndarray,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grad-CAM 히트맵을 원본 이미지에 오버레이

    Args:
        gray_cam: Grad-CAM 히트맵 (H, W), 0~1 정규화
        rgb_img: 원본 RGB 이미지 (H, W, 3), uint8
        alpha: 히트맵 투명도

    Returns:
        (heatmap_rgb, overlay_rgb) 튜플
    """
    h, w, _ = rgb_img.shape
    cam_resized = cv2.resize(gray_cam, (w, h))

    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (heatmap_rgb * alpha + rgb_img * (1 - alpha)).astype(np.uint8)
    return heatmap_rgb, overlay


# ============================================================
# 4. 외부 API 함수
# ============================================================

def run_gradcam_on_pil(
    model_name: str,
    pil_img: Image.Image,
    alpha: float = 0.4,
    target_class: Optional[int] = None,
) -> Dict[str, Any]:
    """
    단일 모델에 대해 Grad-CAM 실행

    Args:
        model_name: 모델 이름
        pil_img: 입력 이미지
        alpha: 히트맵 투명도
        target_class: 타겟 클래스 (None이면 자동 선택)

    Returns:
        결과 딕셔너리
    """
    bundle = get_model_bundle(model_name)
    model = bundle["model"]
    target_layer = bundle["target_layer"]
    transform = bundle["transform"]

    gradcam = GradCAM(model, target_layer)
    x = preprocess_pil(pil_img, transform)

    best_th = get_best_threshold(model_name)
    cam, t_cls, t_prob, logits = gradcam(x, target_class=target_class, threshold=best_th)

    rgb = np.array(pil_img)
    heatmap_rgb, overlay_rgb = apply_colormap_on_image(cam, rgb, alpha=alpha)

    return {
        "model_name": model_name,
        "cam": cam,
        "heatmap": heatmap_rgb,
        "overlay": overlay_rgb,
        "target_class": t_cls,
        "target_prob": t_prob,
        "logits": logits,
    }


def run_gradcam_on_file_bytes(
    model_name: str,
    file_bytes: bytes,
    alpha: float = 0.4,
    is_dicom: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    파일 바이트에 대해 Grad-CAM 실행

    Args:
        model_name: 모델 이름
        file_bytes: 파일 바이트
        alpha: 히트맵 투명도
        is_dicom: DICOM 여부

    Returns:
        결과 딕셔너리 (pil_image, dicom_meta 포함)
    """
    if is_dicom:
        pil_img, dcm = dicom_to_pil(file_bytes)
    else:
        pil_img = image_bytes_to_pil(file_bytes)
        dcm = None

    result = run_gradcam_on_pil(model_name, pil_img, alpha=alpha)
    result["pil_image"] = pil_img
    result["dicom_meta"] = dcm
    return result


def run_gradcam_all_models(
    pil_img: Image.Image,
    alpha: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    모든 모델에 대해 Grad-CAM 실행

    Args:
        pil_img: 입력 이미지
        alpha: 히트맵 투명도

    Returns:
        각 모델별 결과 딕셔너리 리스트
    """
    results: List[Dict[str, Any]] = []
    for model_name in MODEL_CONFIG.keys():
        results.append(run_gradcam_on_pil(model_name, pil_img, alpha=alpha))
    return results


def get_available_models() -> List[str]:
    """
    사용 가능한 모델 목록 반환

    Returns:
        모델 이름 리스트
    """
    return list(MODEL_CONFIG.keys())
