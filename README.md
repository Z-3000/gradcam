---
title: X-ray Grad-CAM Explorer
emoji: 🩻
colorFrom: gray
colorTo: blue
sdk: streamlit
sdk_version: "1.51.0"
app_file: main.py
pinned: false
---

# X-ray Grad-CAM Explorer

딥러닝 모델의 폐렴 분류 결과와 **판단 근거를 시각화**하는 의료 영상 분석 도구입니다.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)

## 📋 프로젝트 개요

X-ray 이미지에서 폐렴을 진단하는 AI 모델이 **어느 부분을 보고 판단했는지** Grad-CAM 기법으로 시각화합니다.
의료진이 AI의 판단 근거를 검증하고, 모델의 신뢰성을 평가할 수 있도록 지원합니다.

### 주요 기능

- ✅ **3개 딥러닝 모델 지원**: ResNet18, EfficientNet-B0, DenseNet121
- 🔍 **Grad-CAM 시각화**: 모델이 주목한 영역을 히트맵으로 표시
- 🏥 **DICOM 포맷 지원**: 의료 표준 포맷 직접 처리
- 📊 **다중 모델 비교**: 3개 모델의 예측을 동시에 확인
- 🎨 **직관적인 UI**: Streamlit 기반 웹 인터페이스

## 🏗️ 프로젝트 구조

```
.
├── main.py                 # Streamlit 웹 앱 (사용자 인터페이스)
├── gradcam_core.py         # Grad-CAM 백엔드 모듈
├── checkpoints/            # 학습된 모델 체크포인트
│   ├── 251115_resnet18_NP.pth
│   ├── 251115_efficientnet_NP.pth
│   └── 251115_densenet121_NP.pth
└── README.md
```

## 🚀 시작하기

### 필수 요구사항

```bash
Python >= 3.8
CUDA (GPU 사용 시, 선택사항)
```

### 설치

1. **저장소 클론**
```bash
git clone <repository-url>
cd xray-gradcam-explorer
```

2. **의존성 설치**
```bash
pip install torch torchvision
pip install streamlit
pip install pydicom
pip install opencv-python
pip install pillow numpy
```

3. **체크포인트 준비**
   - `checkpoints/` 폴더에 학습된 `.pth` 파일 배치
   - 또는 `gradcam_core.py`의 `MODEL_CONFIG`에서 경로 수정

### 실행

```bash
streamlit run main.py
```

브라우저에서 `http://localhost:8501` 자동 실행

## 💡 사용 방법

### 1️⃣ 이미지 업로드
- 좌측 사이드바에서 X-ray 이미지 업로드
- 지원 포맷: `.dcm` (DICOM), `.png`, `.jpg`, `.jpeg`

### 2️⃣ 모드 선택
- **단일 모델**: 한 모델의 상세 분석
- **3개 모델 비교**: 여러 모델의 예측 비교

### 3️⃣ 결과 확인
- **탭1 - Grad-CAM 시각화**: 원본 vs 히트맵 오버레이
- **탭2 - 예측 결과**: 클래스별 확률 그래프
- **탭3 - 메타데이터**: 이미지 정보 및 DICOM 메타데이터

## 🔬 기술 상세

### Grad-CAM (Gradient-weighted Class Activation Mapping)

모델의 마지막 합성곱 레이어에서:
1. 특정 클래스에 대한 그래디언트 계산
2. 채널별 가중치 평균 산출
3. 활성화 맵과 가중치를 곱해 히트맵 생성
4. 원본 이미지에 오버레이

**결과**: 빨간색 영역 = 모델이 중요하게 본 부분

### DICOM 전처리

```python
1. HU (Hounsfield Unit) 변환
   → RescaleSlope, RescaleIntercept 적용

2. Lung Windowing
   → Center: 40, Width: 800
   → 폐 영역 강조

3. 정규화 및 RGB 변환
   → 모델 입력 형식에 맞게 변환
```

### 모델별 최적 임계값

각 모델의 학습 결과를 기반으로 최적화된 임계값 사용:

| 모델 | Threshold |
|------|-----------|
| ResNet18 | 0.25 |
| EfficientNet-B0 | 0.15 |
| DenseNet121 | 0.18 |

## 📁 핵심 파일 설명

### `gradcam_core.py`
- **역할**: Grad-CAM 계산 및 이미지 전처리 백엔드
- **주요 클래스/함수**:
  - `GradCAM`: Grad-CAM 구현 클래스
  - `dicom_to_pil()`: DICOM → PIL 이미지 변환
  - `run_gradcam_on_pil()`: Grad-CAM 실행 및 결과 반환
  - `get_best_threshold()`: 모델별 최적 임계값 제공

### `main.py`
- **역할**: Streamlit 기반 웹 인터페이스
- **주요 기능**:
  - 이미지 업로드 및 전처리
  - Grad-CAM 결과 시각화
  - 다크 테마 UI (의료 영상 분석에 최적화)
  - 3개 탭 구성 (시각화/예측/메타데이터)

## 🎯 활용 사례

1. **의료진 의사결정 지원**
   - AI 판단 근거를 시각적으로 확인
   - 오진 가능성 사전 검토

2. **모델 성능 평가**
   - 여러 모델의 예측 비교
   - 히트맵을 통한 모델 신뢰도 평가

3. **교육 및 연구**
   - AI 모델의 작동 원리 이해
   - Grad-CAM 기법 학습 자료

## ⚠️ 주의사항

- 본 도구는 **연구 및 교육 목적**으로 제작되었습니다
- 실제 의료 진단에 사용하기 전 반드시 전문가 검증 필요
- DICOM 파일 처리 시 개인정보 보호에 유의하세요

## 🔧 트러블슈팅

### 체크포인트 로드 오류
```python
# gradcam_core.py 또는 main.py의 MODEL_CONFIG 경로 확인
MODEL_CONFIG = {
    "ResNet18": {
        "checkpoint": "checkpoints/251115_resnet18_NP.pth",  # 실제 경로로 수정
        ...
    }
}
```

### CUDA 메모리 부족
```python
# CPU 모드로 강제 전환
DEVICE = torch.device("cpu")
```

### DICOM 파일 인식 실패
```bash
pip install pydicom --upgrade
```

## 📚 참고 자료

- [Grad-CAM 논문](https://arxiv.org/abs/1610.02391)
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [DICOM 표준](https://www.dicomstandard.org/)

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 👤 작성자

AI 데이터 분석 포트폴리오 프로젝트

---

**📧 문의사항이나 개선 제안은 이슈로 등록해주세요!**
