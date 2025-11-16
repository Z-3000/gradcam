# main.py
import io
import os
from typing import Tuple, Dict, List

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import numpy as np
import cv2
from PIL import Image
import pydicom

# ============================================================
# 0. 기본 설정
# ============================================================
st.set_page_config(
    page_title="X-ray Grad-CAM Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이진 분류 가정: 0 = Normal, 1 = Pneumonia
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Pneumonia"]

# 체크포인트 경로 (실제 파일명 기준)
MODEL_CONFIG: Dict[str, Dict] = {
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

# ✅ Phase0 training_results 기준 best_threshold
BEST_THRESHOLDS: Dict[str, float] = {
    "ResNet18": 0.25,
    "EfficientNet-B0": 0.15,
    "DenseNet121": 0.18,
}


def get_best_threshold(model_name: str) -> float:
    """모델별 best_threshold 반환, 없으면 0.5 사용"""
    return BEST_THRESHOLDS.get(model_name, 0.5)

# ============================================================
# 1. 다크 + 그레이스케일(X-ray 느낌) 커스텀 CSS
# ============================================================
DARK_CSS = """
<style>
:root {
    --bg-main: #050608;
    --bg-panel: #14161A;
    --bg-panel-soft: #1E2228;
    --accent: #4FD1C5;
    --accent-soft: #2C7A7B;
    --text-main: #E5E7EB;
    --text-muted: #9CA3AF;
    --border-subtle: #2D333B;
}

/* 전체 앱 배경 */
.stApp {
    background: radial-gradient(circle at top, #1C1F26 0, #050608 55%) !important;
    color: var(--text-main) !important;
}

/* 기본 텍스트 색 */
body, p, span, li {
    color: var(--text-main) !important;
}

/* 헤더/타이틀 */
h1, h2, h3 {
    color: #F9FAFB !important;
    font-weight: 600 !important;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background-color: #050608 !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--text-main) !important;
}

/* 사이드바 안의 카드/컨테이너 느낌 */
.block-container {
    padding-top: 1.5rem !important;
}

/* 파일 업로더 */
section[data-testid="stSidebar"] .stFileUploader > div:first-child {
    background-color: var(--bg-panel) !important;
    border-radius: 10px !important;
    border: 1px dashed var(--border-subtle) !important;
    padding: 10px !important;
}

section[data-testid="stSidebar"] .stFileUploader * {
    background-color: transparent !important;
    color: var(--text-main) !important;
}

/* 버튼 */
.stButton > button {
    background-color: var(--accent-soft) !important;
    color: #F9FAFB !important;
    border-radius: 999px !important;
    border: 1px solid var(--accent) !important;
    padding: 0.4rem 1.4rem !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    background-color: var(--accent) !important;
}

/* 라디오/셀렉트박스 라벨 색 */
.stRadio label, .stSelectbox label {
    color: var(--text-muted) !important;
    font-size: 0.9rem !important;
}

/* 슬라이더 */
[data-testid="stSlider"] > div {
    padding-top: 5px !important;
}
[data-baseweb="slider"] > div > div {
    background-color: #374151 !important;
}
[data-baseweb="slider"] > div > div > div {
    background-color: var(--accent) !important;
}
[data-baseweb="slider"] [role="slider"] {
    background-color: var(--accent) !important;
    border: 2px solid #F9FAFB !important;
}

/* 탭 컨테이너 – 뒤에 깔린 회색 알약 배경 제거 */
.stTabs [data-baseweb="tab-list"] {
    border-radius: 0 !important;
    background-color: transparent !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* 탭 버튼만 pill 형태 유지 */
.stTabs [data-baseweb="tab-list"] button {
    border-radius: 999px !important;
}

/* 활성 탭 */
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: rgba(79, 209, 197, 0.16) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}

/* 비활성 탭 */
.stTabs [data-baseweb="tab-list"] button[aria-selected="false"] {
    color: var(--text-muted) !important;
}

/* 이미지 주변 패널 – 배경/테두리 제거해서 빈 박스 없애기 */
.xray-panel {
    padding: 0;
    margin: 0 0 6px 0;
    border: none;
    background: transparent;
}

/* 패널 제목만 심플하게 남기기 */
.xray-panel-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

/* 하단 설명 카드 */
.info-card {
    background-color: var(--bg-panel-soft);
    border-radius: 12px;
    border: 1px solid var(--border-subtle);
    padding: 10px 12px;
}

/* 데이터프레임/테이블 */
.dataframe {
    background-color: var(--bg-panel) !important;
}

/* 알림/경고 박스 */
.stAlert {
    background-color: rgba(55, 65, 81, 0.6) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border-subtle) !important;
}

/* 작은 뱃지 스타일 */
.badge-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    border: 1px solid var(--border-subtle);
    color: var(--text-muted);
}

/* Grad-CAM 트리거 카드 */
.gradcam-trigger-box {
    margin-top: 0rem;
    margin-bottom: 0.75rem;
    padding: 10px 12px;
    border-radius: 12px;
    border: 1px solid var(--border-subtle);
    background-color: #111827;
}
.gradcam-trigger-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-main);
    margin-bottom: 4px;
}
.gradcam-trigger-desc {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 0;
}

/* 확률 바 */
.prob-bar-bg {
    background-color: #111827;
    border-radius: 999px;
    height: 8px;
    width: 100%;
}
.prob-bar-fill {
    background: linear-gradient(90deg, var(--accent), #FBBF24);
    border-radius: 999px;
    height: 8px;
}

/* 제목 섹션 여백 및 크기 조정 */
h3 {
    font-size: 1.35rem !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.6rem !important;
}

/* 페이지 상단 전체 패딩 */
.block-container {
    padding-top: 2.5rem !important;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ============================================================
# 2. DICOM / 이미지 로딩 유틸
# ============================================================

def load_dicom_as_pil(file_bytes: bytes) -> Tuple[Image.Image, pydicom.Dataset]:
    """
    DICOM 바이트 → (PIL Image, DICOM Dataset)
    학습 시 사용한 dicom_to_png와 동일한 방식:
    - HU 변환 (RescaleSlope/Intercept)
    - Lung window (center=40, width=800)
    """
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    img = ds.pixel_array.astype(np.float32)

    # HU 변환
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        img = img * slope + intercept

    # Lung window 적용
    window_center = 40.0
    window_width = 800.0
    min_val = window_center - window_width / 2.0
    max_val = window_center + window_width / 2.0

    img = (img - min_val) / (max_val - min_val)
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)

    pil_img = Image.fromarray(img).convert("RGB")
    return pil_img, ds


def load_generic_image_as_pil(file_bytes: bytes) -> Image.Image:
    """PNG/JPEG → PIL(RGB)"""
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# ============================================================
# 3. Grad-CAM 핵심 클래스
# ============================================================

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()
            out.register_hook(self._save_gradients)

        self.target_layer.register_forward_hook(forward_hook)

    def _save_gradients(self, grad):
        self.gradients = grad.detach()

    def __call__(
        self,
        x: torch.Tensor,
        target_class: int = None,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """
        x: (1, C, H, W)
        threshold:
            P(Pneumonia) ≥ threshold → Pneumonia(1), else Normal(0)

        return:
            cam: (H, W) 0~1 float
            target_class: 0/1
            target_prob: float (선택된 클래스의 확률)
            logits: numpy array
        """
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        logits = self.model(x)  # (1, C)

        # 이진 모델 (logit for Pneumonia)
        if logits.shape[1] == 1:
            prob_pos = torch.sigmoid(logits)[0, 0]  # P(Pneumonia)

            if target_class is None:
                target_class = int((prob_pos >= threshold).item())

            score = logits[0, 0]
            probs_vec = torch.stack([1 - prob_pos, prob_pos], dim=0)

        else:
            # 다중 클래스 대비
            probs_vec = F.softmax(logits, dim=1)[0]
            if target_class is None:
                target_class = int(torch.argmax(probs_vec).item())
            score = logits[0, target_class]

        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook이 제대로 등록되지 않았습니다.")

        act = self.activations
        grad = self.gradients

        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.squeeze().cpu().numpy()
        target_prob = float(probs_vec[target_class].item())
        logits_np = logits.detach().cpu().numpy()[0]

        return cam_np, target_class, target_prob, logits_np

# ============================================================
# 4. 모델 로딩/전처리
# ============================================================

@st.cache_resource
def load_model(model_name: str) -> Tuple[nn.Module, nn.Module, transforms.Compose]:
    """
    모델 로드 + target_layer 찾기 + transform 반환
    - 성능 보존을 위해 checkpoint 구조(out_features=1)에 맞춰 모델 생성
    """
    cfg = MODEL_CONFIG[model_name]
    builder = cfg["builder"]
    target_layer_name = cfg["target_layer"]
    input_size = cfg["input_size"]

    model = builder(weights=None)

    # 학습 당시 fc 출력 차원 = 1 → 동일하게 맞춤
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

    ckpt_path = cfg["checkpoint"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {ckpt_path}\n"
            f"MODEL_CONFIG에서 경로를 실제 .pth 파일 위치로 수정하세요."
        )

    # PyTorch 2.6: weights_only 기본값 True → 예전 체크포인트 대응
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # 성능 보존을 위해 strict 로 전체 로딩
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    named_modules = dict(model.named_modules())
    if target_layer_name not in named_modules:
        raise ValueError(
            f"target_layer '{target_layer_name}' 를 model.named_modules()에서 찾을 수 없습니다."
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

    return model, target_layer, transform


def preprocess_for_model(img: Image.Image, transform: transforms.Compose) -> torch.Tensor:
    x = transform(img)
    x = x.unsqueeze(0)
    return x.to(DEVICE)


def apply_colormap_on_image(
    gray_cam: np.ndarray,
    rgb_img: np.ndarray,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    gray_cam: (H, W) 0~1
    rgb_img: (H, W, 3) uint8
    """
    h, w, _ = rgb_img.shape
    cam_resized = cv2.resize(gray_cam, (w, h))

    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (heatmap_rgb * alpha + rgb_img * (1 - alpha)).astype(np.uint8)
    return heatmap_rgb, overlay

# ============================================================
# 5. 상단 설명
# ============================================================

st.markdown(
    """
    ### X-ray Grad-CAM Explorer
    
    딥러닝 모델(ResNet18 / EfficientNet-B0 / DenseNet121)의 **폐렴 분류 결과와 Grad-CAM 시각화**를 한 화면에서 확인하는 도구입니다.  
    - DICOM 또는 PNG/JPEG X-ray 입력  
    - 개별 모델 또는 3개 모델 비교  
    - 추후 유사 증례 검색(Embedding 기반) 확장 가능 구조
    """,
    unsafe_allow_html=False,
)

st.markdown("<div class='badge-pill'>Experimental · RSNA Pneumonia (Assumed)</div>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# 6. 사이드바 (업로드 + 트리거 + 옵션)
# ============================================================

st.sidebar.header("입력 이미지 업로드")

uploaded_file = st.sidebar.file_uploader(
    "",
    type=["dcm", "png", "jpg", "jpeg"],
)

# 세션 상태로 "한 번만 눌러도 이후 자동" 플래그 관리
if "gradcam_auto" not in st.session_state:
    st.session_state["gradcam_auto"] = False

generate_btn = st.sidebar.button("Grad-CAM 생성", use_container_width=True)

if generate_btn:
    st.session_state["gradcam_auto"] = True

st.sidebar.markdown("---")
st.sidebar.header("Grad-CAM Options")

view_mode = st.sidebar.radio(
    "View Mode",
    ["단일 모델", "3개 모델 비교"],
)

model_name = st.sidebar.selectbox(
    "Model Selection",
    list(MODEL_CONFIG.keys()),
    index=0,
)

alpha = st.sidebar.slider(
    "Heatmap Overlay Alpha",
    min_value=0.1, max_value=0.9, value=0.45, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.header("유사 증례(Prototype)")

show_similar = st.sidebar.checkbox(
    "유사 증례 패널 표시 (구조만, 인덱스는 추후 구현)", value=False
)

# ============================================================
# 7. Grad-CAM 실행 함수
# ============================================================

def run_gradcam_for_model(
    model_label: str,
    base_img: Image.Image,
    alpha: float,
) -> Dict:
    model, target_layer, transform = load_model(model_label)
    gradcam = GradCAM(model, target_layer)

    x = preprocess_for_model(base_img, transform)

    # ✅ 모델별 best_threshold 적용
    best_th = get_best_threshold(model_label)
    cam, target_class, target_prob, logits = gradcam(
        x,
        target_class=None,
        threshold=best_th,
    )

    rgb = np.array(base_img)
    heatmap_rgb, overlay_rgb = apply_colormap_on_image(cam, rgb, alpha=alpha)

    return {
        "model_name": model_label,
        "cam": cam,
        "heatmap": heatmap_rgb,
        "overlay": overlay_rgb,
        "target_class": target_class,
        "target_prob": target_prob,
        "logits": logits,
    }

# ============================================================
# 8. 메인 로직
# ============================================================

if uploaded_file is None:
    st.info("좌측에서 DICOM 또는 X-ray 이미지를 업로드한 뒤, Grad-CAM 생성을 수행하세요.")
else:
    # 버튼이 한 번이라도 눌렸는지 여부
    triggered = st.session_state.get("gradcam_auto", False)

    # 파일 로드
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    if filename.endswith(".dcm"):
        base_img, dcm_ds = load_dicom_as_pil(file_bytes)
        dicom_mode = True
    else:
        base_img = load_generic_image_as_pil(file_bytes)
        dcm_ds = None
        dicom_mode = False

    # 탭 구성
    tab_viz, tab_pred, tab_meta = st.tabs(
        ["🖼 Grad-CAM 시각화", "📊 예측 결과", "ℹ 메타데이터 / 유사 증례"]
    )

    # -----------------------------
    # 탭 1: Grad-CAM 시각화
    # -----------------------------
    with tab_viz:
        if not triggered:
            st.warning("좌측의 **Grad-CAM 생성** 버튼을 한 번 눌러 시각화를 시작하세요.")
        else:
            if view_mode == "단일 모델":
                with st.spinner(f"{model_name} Grad-CAM 계산 중..."):
                    result = run_gradcam_for_model(model_name, base_img, alpha)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("<div class='xray-panel'>", unsafe_allow_html=True)
                    st.markdown("<div class='xray-panel-title'>Original</div>", unsafe_allow_html=True)
                    st.image(base_img, use_container_width=True, clamp=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown("<div class='xray-panel'>", unsafe_allow_html=True)
                    st.markdown("<div class='xray-panel-title'>Grad-CAM Heatmap</div>", unsafe_allow_html=True)
                    st.image(result["heatmap"], use_container_width=True, clamp=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col3:
                    st.markdown("<div class='xray-panel'>", unsafe_allow_html=True)
                    st.markdown("<div class='xray-panel-title'>Overlay</div>", unsafe_allow_html=True)
                    st.image(result["overlay"], use_container_width=True, clamp=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(
                    f"<div class='info-card'>"
                    f"<b>{model_name}</b> · Predicted: <b>{CLASS_NAMES[result['target_class']]}</b> "
                    f"(p = {result['target_prob']:.3f})"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            else:  # 3개 모델 비교
                cols = st.columns(3)
                results: List[Dict] = []

                with st.spinner("3개 모델에 대해 Grad-CAM 계산 중..."):
                    for m in MODEL_CONFIG.keys():
                        results.append(run_gradcam_for_model(m, base_img, alpha))

                for col, res in zip(cols, results):
                    with col:
                        st.markdown("<div class='xray-panel'>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='xray-panel-title'>{res['model_name']} · Overlay</div>",
                            unsafe_allow_html=True,
                        )
                        st.image(res["overlay"], use_container_width=True, clamp=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                        st.markdown(
                            f"<div class='info-card'><b>{res['model_name']}</b><br>"
                            f"Pred: <b>{CLASS_NAMES[res['target_class']]}</b> "
                            f"(p = {res['target_prob']:.3f})"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    # -----------------------------
    # 탭 2: 예측 결과
    # -----------------------------
    with tab_pred:
        if not triggered:
            st.info("좌측에서 한 번 Grad-CAM을 생성한 후, 예측 결과가 표시됩니다.")
        else:
            if view_mode == "단일 모델":
                res = run_gradcam_for_model(model_name, base_img, alpha)
                st.subheader(f"{model_name} · 클래스 확률")

                logits = torch.from_numpy(res["logits"])
                if logits.numel() == 1:
                    prob_pos = torch.sigmoid(logits)[0].item()
                    probs = np.array([1 - prob_pos, prob_pos])  # [Normal, Pneumonia]
                else:
                    probs = F.softmax(logits, dim=0).numpy()

                for idx, cls in enumerate(CLASS_NAMES):
                    with st.container():
                        st.write(f"{cls}: {probs[idx]:.3f}")
                        st.markdown(
                            "<div class='prob-bar-bg'>"
                            f"<div class='prob-bar-fill' style='width: {probs[idx]*100:.1f}%;'></div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.subheader("3개 모델 클래스 확률 비교")
                for m in MODEL_CONFIG.keys():
                    res = run_gradcam_for_model(m, base_img, alpha)
                    st.markdown(f"#### {m}")

                    logits = torch.from_numpy(res["logits"])
                    if logits.numel() == 1:
                        prob_pos = torch.sigmoid(logits)[0].item()
                        probs = np.array([1 - prob_pos, prob_pos])
                    else:
                        probs = F.softmax(logits, dim=0).numpy()

                    for idx, cls in enumerate(CLASS_NAMES):
                        with st.container():
                            st.write(f"{cls}: {probs[idx]:.3f}")
                            st.markdown(
                                "<div class='prob-bar-bg'>"
                                f"<div class='prob-bar-fill' style='width: {probs[idx]*100:.1f}%;'></div>"
                                "</div>",
                                unsafe_allow_html=True,
                            )
                    st.markdown("---")

    # -----------------------------
    # 탭 3: 메타데이터 / 유사 증례
    # -----------------------------
    with tab_meta:
        st.subheader("입력 이미지 정보")

        cols_info = st.columns(2)
        with cols_info[0]:
            st.write(f"파일명: `{uploaded_file.name}`")
            st.write(f"형식: {'DICOM' if dicom_mode else '일반 이미지'}")
            st.write(f"원본 크기: {base_img.size[0]} x {base_img.size[1]}")

        if dicom_mode and dcm_ds is not None:
            with cols_info[1]:
                wc = dcm_ds.get("WindowCenter", "N/A")
                ww = dcm_ds.get("WindowWidth", "N/A")
                st.write(f"Window Center: {wc}")
                st.write(f"Window Width: {ww}")
                st.write(f"Modality: {dcm_ds.get('Modality', 'N/A')}")
        else:
            with cols_info[1]:
                st.write("DICOM 메타데이터는 제공되지 않습니다.")

        st.markdown("---")
        st.subheader("유사 증례(Prototype)")

        if not show_similar:
            st.info(
                "유사 증례 검색은 아직 인덱스가 구현되지 않은 프로토타입입니다.\n"
                "- 각 모델의 마지막 feature 벡터를 추출\n"
                "- 사전에 구축한 embedding index(예: FAISS, Annoy)에 대해 k-NN 검색\n"
                "- 상위 k개 사례의 X-ray와 레이블을 함께 표시\n\n"
                "향후 RSNA 또는 병원 PACS에서 추출한 feature index를 연결하여 구현할 수 있습니다."
            )
        else:
            st.warning(
                "⚠ 현재 코드에는 실제 유사 증례 인덱스가 연결되어 있지 않습니다.\n"
                "feature 추출 함수와 k-NN 검색 함수를 추가 구현해야 합니다."
            )

            st.code(
                '''
# 개략적 프로토타입 (추가 구현 필요)

def extract_feature_vector(model, x):
    # 분류기 직전 feature를 꺼내도록 forward hook 또는 모델 수정 필요
    with torch.no_grad():
        feats = model.forward_features(x)  # 모델에 따라 이름 다름 (직접 구현)
    return feats.cpu().numpy()[0]

def find_similar_cases(feature_vec, index, k=5):
    # 예: FAISS index.search(feature_vec[None, :], k)
    distances, indices = index.search(feature_vec[None, :], k)
    return distances[0], indices[0]
                ''',
                language="python",
            )

            st.markdown(
                """
                - 위 형태로 feature extractor + 인덱스를 구현하면,  
                  이 탭에서 상위 k개 증례의 썸네일과 레이블을 Grid로 보여줄 수 있음.
                """
            )
