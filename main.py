"""
X-ray Grad-CAM Explorer - Streamlit 웹 앱
==========================================

폐렴/정상 이진분류 딥러닝 모델의 Grad-CAM 시각화 데모 앱

주요 기능:
- DICOM/PNG/JPEG 흉부 X-ray 이미지 업로드
- ResNet18 / EfficientNet-B0 / DenseNet121 모델 선택
- Grad-CAM 히트맵 오버레이 시각화
- 단일 모델 / 3개 모델 비교 모드

아키텍처:
- main.py: UI 전담 (Streamlit)
- grandcam_core.py: 비즈니스 로직 (Grad-CAM 계산)
- config.py: 설정 중앙 관리
- styles.py: CSS 스타일 관리

Author: JH3907
License: MIT
"""

from typing import Dict, List

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np

# 내부 모듈 import
from config import CLASS_NAMES, MODEL_CONFIG, get_best_threshold
from styles import get_css
from grandcam_core import (
    dicom_to_pil,
    image_bytes_to_pil,
    run_gradcam_on_pil,
    run_gradcam_all_models,
    get_available_models,
)


# ============================================================
# 0. 페이지 설정
# ============================================================
st.set_page_config(
    page_title="X-ray Grad-CAM Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 적용
st.markdown(get_css("dark"), unsafe_allow_html=True)


# ============================================================
# 1. 메인 UI - 상단 설명
# ============================================================
st.markdown(
    """
    ### X-ray Grad-CAM Explorer

    딥러닝 모델(ResNet18 / EfficientNet-B0 / DenseNet121)의 **폐렴 분류 결과와 Grad-CAM 시각화**를 한 화면에서 확인하는 도구입니다.

    * **ResNet18 / EfficientNet-B0 / DenseNet121** : ImageNet으로 사전학습된 CNN 모델들로, 의료 영상 분류에 전이학습(Transfer Learning)을 적용하여 폐렴 진단에 활용
    * **Grad-CAM** : 딥러닝 모델이 예측 시 주목한 영역을 히트맵으로 시각화하는 기법으로, 모델의 판단 근거를 해석 가능하게 만듦
    """,
    unsafe_allow_html=False,
)

st.markdown("<div class='badge-pill'>Experimental · RSNA Pneumonia (Assumed)</div>", unsafe_allow_html=True)
st.markdown("---")


# ============================================================
# 2. 사이드바 - 이미지 업로드 및 옵션
# ============================================================
st.sidebar.header("입력 이미지 업로드")

uploaded_file = st.sidebar.file_uploader(
    "",
    type=["dcm", "png", "jpg", "jpeg"],
)

st.sidebar.markdown("---")
st.sidebar.header("Grad-CAM Options")

view_mode = st.sidebar.radio(
    "View Mode",
    ["단일 모델", "3개 모델 비교"],
)

model_name = st.sidebar.selectbox(
    "Model Selection",
    get_available_models(),
    index=0,
)

alpha = st.sidebar.slider(
    "Heatmap Overlay Alpha",
    min_value=0.1, max_value=0.9, value=0.45, step=0.05
)


# ============================================================
# 3. 메인 로직 - 이미지 처리 및 결과 시각화
# ============================================================
if uploaded_file is None:
    st.info("좌측에서 이미지를 업로드하면 Grad-CAM이 생성됩니다.")
    st.info("📌 **드래그 앤 드롭이 안 될 경우**: 업로드 영역을 **클릭**해서 파일을 선택하거나, 페이지를 새로고침(F5)해보세요.")
else:
    # 파일 로드
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    if filename.endswith(".dcm"):
        base_img, dcm_ds = dicom_to_pil(file_bytes)
        dicom_mode = True
    else:
        base_img = image_bytes_to_pil(file_bytes)
        dcm_ds = None
        dicom_mode = False

    # 탭 구성
    tab_viz, tab_pred, tab_meta = st.tabs(
        ["🖼 Grad-CAM 시각화", "📊 예측 결과", "ℹ 이미지정보"]
    )

    # -----------------------------
    # 탭 1: Grad-CAM 시각화
    # -----------------------------
    with tab_viz:
        if view_mode == "단일 모델":
            with st.spinner(f"{model_name} Grad-CAM 계산 중..."):
                result = run_gradcam_on_pil(model_name, base_img, alpha)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='xray-panel'>", unsafe_allow_html=True)
                st.markdown("<div class='xray-panel-title'>Original</div>", unsafe_allow_html=True)
                st.image(base_img, clamp=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='xray-panel'>", unsafe_allow_html=True)
                st.markdown("<div class='xray-panel-title'>Overlay</div>", unsafe_allow_html=True)
                st.image(result["overlay"], clamp=True)
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

            with st.spinner("3개 모델에 대해 Grad-CAM 계산 중..."):
                results = run_gradcam_all_models(base_img, alpha)

            for col, res in zip(cols, results):
                with col:
                    st.markdown("<div class='xray-panel'>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='xray-panel-title'>{res['model_name']} · Overlay</div>",
                        unsafe_allow_html=True,
                    )
                    st.image(res["overlay"], clamp=True)
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
        if view_mode == "단일 모델":
            result = run_gradcam_on_pil(model_name, base_img, alpha)
            st.subheader(f"{model_name} · 클래스 확률")

            logits = torch.from_numpy(result["logits"])
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
        else:
            st.subheader("3개 모델 클래스 확률 비교")
            for m in get_available_models():
                result = run_gradcam_on_pil(m, base_img, alpha)
                st.markdown(f"#### {m}")

                logits = torch.from_numpy(result["logits"])
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
    # 탭 3: 메타데이터
    # -----------------------------
    with tab_meta:
        st.subheader("이미지 정보")

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
