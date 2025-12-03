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

딥러닝 폐렴 분류 모델의 예측 결과를 **Grad-CAM**으로 시각화하는 웹 데모입니다.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/JH3907/xraygradcam)

## 주요 기능

- **DICOM/PNG/JPEG** 흉부 X-ray 이미지 업로드
- **3개 모델** 지원: ResNet18, EfficientNet-B0, DenseNet121
- **Grad-CAM 시각화**: 모델이 주목한 영역을 히트맵으로 표시
- **단일 모델 / 3개 모델 비교** 모드

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    HuggingFace Spaces                       │
│                                                             │
│  [이미지 업로드] → [main.py] → [Grad-CAM 시각화]            │
│   DICOM/PNG/JPG    Streamlit     히트맵 오버레이            │
│                        │                                    │
│                        ▼                                    │
│               [grandcam_core.py]                            │
│                        │                                    │
│        ┌───────────────┼───────────────┐                   │
│        ▼               ▼               ▼                   │
│   [ResNet18]    [EfficientNet]   [DenseNet121]             │
│   (th=0.25)      (th=0.15)        (th=0.18)               │
└─────────────────────────────────────────────────────────────┘
```

## 기술 스택

| 분류 | 기술 |
|------|------|
| 웹 프레임워크 | Streamlit 1.51.0 |
| 딥러닝 | PyTorch, torchvision |
| 영상처리 | OpenCV, Pillow |
| 의료영상 | pydicom (DICOM 파싱) |
| 배포 | HuggingFace Spaces |
| CI/CD | GitHub Actions |

## 프로젝트 구조 (모듈 분리 설계)

```
xraygradcam/
├── main.py              # UI 전담 (Streamlit)
├── grandcam_core.py     # 비즈니스 로직 (Grad-CAM 계산)
├── config.py            # 설정 중앙 관리 (모델, threshold, DICOM)
├── styles.py            # CSS 스타일 관리 (테마)
├── checkpoints/         # 모델 가중치 (Git LFS)
│   ├── 251115_resnet18_NP.pth
│   ├── 251115_efficientnet_NP.pth
│   └── 251115_densenet121_NP.pth
├── requirements.txt     # 의존성
└── .github/workflows/   # HuggingFace 자동 동기화
```

### 모듈 역할

| 모듈 | 역할 | 확장 포인트 |
|------|------|-------------|
| `config.py` | 설정 중앙 관리 | 모델 추가 시 `MODEL_CONFIG`에 항목 추가 |
| `styles.py` | CSS 테마 관리 | 새 테마 추가 가능 |
| `grandcam_core.py` | Grad-CAM 로직 | FastAPI 등 다른 프레임워크에서 재사용 |
| `main.py` | Streamlit UI | UI 변경 시 이 파일만 수정 |

## 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 앱 실행
streamlit run main.py
```

## 모델 설정

| 모델 | Target Layer | Threshold | 비고 |
|------|-------------|-----------|------|
| ResNet18 | layer4 | 0.25 | 표준 CNN, 빠른 추론 |
| EfficientNet-B0 | features | 0.15 | 가장 민감 |
| DenseNet121 | features | 0.18 | Dense 연결 |

- **Threshold**: 학습 시 ROC 분석 기반 최적값
- 의료 진단 특성상 Recall 우선 → 낮은 threshold 사용

## DICOM 전처리

```python
# Lung window 파라미터 (학습 시 동일)
window_center = 40   # 폐 조직 대표 HU 값
window_width = 800   # 관찰 범위 (-360 ~ 440 HU)
```

## 배포 파이프라인

```
GitHub (main) → GitHub Actions → HuggingFace Spaces
```

- `main` 브랜치 push 시 자동 동기화
- `.github/workflows/sync-to-huggingface.yml` 참조

## 라이선스

MIT License

## 작성자

- **GitHub**: [Z-3000](https://github.com/Z-3000)
- **HuggingFace**: [JH3907](https://huggingface.co/JH3907)
