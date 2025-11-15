# gradcam_core.py
"""
X-ray Grad-CAM 백엔드 모듈

- 3개 모델(ResNet18 / EfficientNet-B0 / DenseNet121) 로드
- DICOM/일반 이미지 → PIL → Tensor 변환
- Grad-CAM heatmap + overlay 생성
- 추후 Streamlit / FastAPI 등에서 재사용 가능

주의:
- .pth 체크포인트의 구조(특히 출력 차원)를 존중하여 성능에 영향을 주지 않도록 설계
- 현재 가정: 학습 당시 최종 출력이 1차원 (단일 로짓, Pneumonia 점수)
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

try:
    # 기존 DICOM 전처리 모듈이 있으면 재사용
    from preprocess_core import dicom_to_pil as _dicom_to_pil_existing
    _HAS_PREPROCESS_CORE = True
except ImportError:
    _HAS_PREPROCESS_CORE = False

try:
    import pydicom  # type: ignore
except ImportError:
    pydicom = None  # DICOM 없는 환경에서도 import 에러 안 나게 처리

# ============================================================
# 0. 전역 설정
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이진 분류 가정: [Normal, Pneumonia]
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Pneumonia"]

# 실제 .pth 위치에 맞게 checkpoint 경로만 맞추면 됨
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
}

# ============================================================
# 1. DICOM / 일반 이미지 로딩
# ============================================================

def dicom_to_pil(file_bytes: bytes) -> Tuple[Image.Image, Any]:
    """
    DICOM 바이트 → (PIL Image RGB, DICOM Dataset)
    """
    if _HAS_PREPROCESS_CORE:
        pil_img, dcm = _dicom_to_pil_existing(file_bytes)
        return pil_img, dcm

    if pydicom is None:
        raise ImportError("pydicom 이 설치되어 있지 않습니다. pip install pydicom 필요.")

    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    pixel_array = ds.pixel_array.astype(np.float32)

    # min-max normalization → 0~255
    pixel_array -= pixel_array.min()
    max_val = pixel_array.max()
    if max_val > 0:
        pixel_array /= max_val
    pixel_array = (pixel_array * 255.0).clip(0, 255).astype(np.uint8)

    # Grayscale → RGB
    pil_img = Image.fromarray(pixel_array).convert("RGB")
    return pil_img, ds


def image_bytes_to_pil(file_bytes: bytes) -> Image.Image:
    """
    PNG/JPEG 등 일반 이미지를 PIL Image(RGB)로 변환
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
    기본 Grad-CAM 구현

    사용:
        gradcam = GradCAM(model, target_layer)
        cam, target_class, target_prob, logits = gradcam(x)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # forward hook
        def forward_hook(module, inputs, output):
            self.activations = output.detach()
            output.register_hook(self._save_gradients)

        self.target_layer.register_forward_hook(forward_hook)

    def _save_gradients(self, grad: torch.Tensor):
        self.gradients = grad.detach()

    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """
        x: (1, C, H, W) on DEVICE

        return:
            cam          : (H, W) [0~1] numpy
            target_class : 0 or 1 (Normal / Pneumonia)
            target_prob  : 해당 클래스 확률
            logits       : numpy array, 원본 모델 출력 (shape (1,) 또는 (C,))
        """
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        logits = self.model(x)  # (1, C)

        # 🔹 출력이 하나인 이진 모델 (logit for "Pneumonia")
        if logits.shape[1] == 1:
            prob_pos = torch.sigmoid(logits)[0, 0]  # P(Pneumonia)

            if target_class is None:
                # 임계값 0.5 기준 (best threshold와 다를 수 있음 → 추측입니다)
                target_class = int((prob_pos >= 0.5).item())

            # Grad-CAM은 Pneumonia logit 기준으로 계산
            score = logits[0, 0]
            probs_vec = torch.stack([1 - prob_pos, prob_pos], dim=0)  # [P(Normal), P(Pneumonia)]

        else:
            # 다중 클래스 모델일 경우
            probs_vec = F.softmax(logits, dim=1)[0]
            if target_class is None:
                target_class = int(torch.argmax(probs_vec).item())
            score = logits[0, target_class]

        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook이 제대로 작동하지 않았습니다.")

        act = self.activations
        grad = self.gradients

        # global average pooling → 채널별 weight
        weights = grad.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

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
    MODEL_CONFIG에 맞춰 모델 생성.
    체크포인트 구조를 그대로 재현하기 위해 out_features=1 로 설정.
    """
    cfg = MODEL_CONFIG[model_name]
    builder = cfg["builder"]

    model = builder(weights=None)

    # 학습 당시 출력이 1개였음 (에러 메세지로 확인됨) → 성능 보존 위해 동일하게 1로 설정
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
        raise ValueError(f"지원하지 않는 모델 타입입니다: {type(model)}")

    return model


def _load_state_dict(model: nn.Module, ckpt_path: str) -> None:
    """
    .pth 체크포인트를 그대로 로딩 (fc 포함) → 성능 보존.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt  # 순수 state_dict 가정

    # 성능 보존을 위해 strict=True 로 전체 로딩
    model.load_state_dict(state_dict)

# 간단 캐싱
_LOADED_MODELS: Dict[str, Dict[str, Any]] = {}


def get_model_bundle(model_name: str) -> Dict[str, Any]:
    """
    모델 / target_layer / transform 을 로드해서 캐싱 후 반환
    """
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    if model_name not in MODEL_CONFIG:
        raise KeyError(f"지원하지 않는 모델 이름입니다: {model_name}")

    cfg = MODEL_CONFIG[model_name]
    ckpt_path = cfg["checkpoint"]
    target_layer_name = cfg["target_layer"]
    input_size = cfg["input_size"]

    model = _build_model(model_name)
    _load_state_dict(model, ckpt_path)
    model.to(DEVICE)
    model.eval()

    named_modules = dict(model.named_modules())
    if target_layer_name not in named_modules:
        raise ValueError(
            f"target_layer '{target_layer_name}' 를 model.named_modules() 내에서 찾을 수 없습니다."
        )
    target_layer = named_modules[target_layer_name]

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
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
    PIL → (1, C, H, W) Tensor on DEVICE
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
    Grad-CAM 결과(gray) + 원본 RGB → heatmap / overlay
    """
    h, w, _ = rgb_img.shape
    cam_resized = cv2.resize(gray_cam, (w, h))

    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (heatmap_rgb * alpha + rgb_img * (1 - alpha)).astype(np.uint8)
    return heatmap_rgb, overlay

# ============================================================
# 4. 외부에서 호출할 함수
# ============================================================

def run_gradcam_on_pil(
    model_name: str,
    pil_img: Image.Image,
    alpha: float = 0.4,
    target_class: Optional[int] = None,
) -> Dict[str, Any]:
    """
    단일 모델에 대해 Grad-CAM 실행
    """
    bundle = get_model_bundle(model_name)
    model = bundle["model"]
    target_layer = bundle["target_layer"]
    transform = bundle["transform"]

    gradcam = GradCAM(model, target_layer)
    x = preprocess_pil(pil_img, transform)
    cam, t_cls, t_prob, logits = gradcam(x)

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
    파일 바이트(DICOM or 일반 이미지)에 대해 Grad-CAM 실행
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


def run_gradcam_all_models_on_pil(
    pil_img: Image.Image,
    alpha: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    3개 모델(ResNet18 / EfficientNet-B0 / DenseNet121)에 대해 한 번에 Grad-CAM 실행
    """
    results: List[Dict[str, Any]] = []
    for m in MODEL_CONFIG.keys():
        results.append(run_gradcam_on_pil(m, pil_img, alpha=alpha))
    return results
