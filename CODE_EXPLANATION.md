# X-ray Grad-CAM Explorer 코드 완전 가이드

> 비전공자를 위한 코드 동작 원리 및 파라미터 설정 근거 설명서

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [핵심 개념 이해하기](#2-핵심-개념-이해하기)
3. [전체 코드 흐름도](#3-전체-코드-흐름도)
4. [gradcam_core.py 상세 설명](#4-gradcam_corepy-상세-설명)
5. [main.py 상세 설명](#5-mainpy-상세-설명)
6. [파라미터 설정 근거](#6-파라미터-설정-근거)
7. [자주 묻는 질문 (FAQ)](#7-자주-묻는-질문-faq)

---

## 1. 프로젝트 개요

### 1.1 이 프로젝트가 하는 일

```
흉부 X-ray 영상 → 딥러닝 모델 → 폐렴/정상 분류 + 판단 근거 시각화
```

**한 줄 요약**: 흉부 X-ray 이미지를 업로드하면, AI가 폐렴인지 정상인지 판단하고, **왜 그렇게 판단했는지 히트맵으로 보여주는** 웹 애플리케이션입니다.

### 1.2 왜 이런 프로젝트가 필요한가?

| 문제점 | 해결책 |
|--------|--------|
| AI가 "폐렴입니다"라고만 하면 의사가 믿기 어려움 | Grad-CAM으로 **어느 부위를 보고** 판단했는지 시각화 |
| 딥러닝은 "블랙박스"라서 설명이 안 됨 | 히트맵으로 모델의 **관심 영역** 표시 |
| 모델마다 다른 특성이 있음 | 3개 모델 비교로 **교차 검증** 가능 |

### 1.3 파일 구성

```
open_cv_grand/
├── main.py              # 웹 UI (Streamlit 앱) - 사용자가 직접 보는 화면
├── gradcam_core.py      # 핵심 엔진 - 실제 AI 분석 수행
├── checkpoints/         # 학습된 모델 파일들 (.pth)
│   ├── 251115_resnet18_NP.pth
│   ├── 251115_efficientnet_NP.pth
│   └── 251115_densenet121_NP.pth
└── requirements.txt     # 필요한 라이브러리 목록
```

---

## 2. 핵심 개념 이해하기

### 2.1 DICOM이란?

**정의**: 의료 영상을 저장하는 국제 표준 포맷입니다.

**일반 이미지(JPG/PNG)와 다른 점**:

| 항목 | JPG/PNG | DICOM |
|------|---------|-------|
| 픽셀값 | 0~255 (밝기) | 원시값 (장비 고유) |
| 메타데이터 | 거의 없음 | 환자정보, 촬영조건 등 풍부 |
| 변환 필요 | 없음 | HU 변환 필요 |

**왜 변환이 필요한가?**

DICOM은 저장 용량을 줄이기 위해 압축된 형태로 저장됩니다. 실제 분석에는 표준화된 값이 필요하므로 **HU(Hounsfield Unit) 변환**이 필수입니다.

```python
# DICOM 변환 공식
실제값(HU) = 저장된_픽셀값 × RescaleSlope + RescaleIntercept
```

### 2.2 Window/Level (윈도우 레벨링)

**정의**: CT/X-ray 이미지에서 특정 조직을 더 잘 보이게 하는 기법입니다.

**왜 필요한가?**
- HU값 범위: -1000 ~ +3000 (매우 넓음)
- 모니터 표현: 0 ~ 255 (좁음)
- 전체 범위를 다 표현하면 **대비가 약해서 병변이 안 보임**

**비유로 이해하기**:
> 카메라로 밝은 하늘과 어두운 실내를 동시에 찍으면 둘 다 잘 안 나오는 것과 같습니다.
> 하늘에 맞추면 실내가 까맣고, 실내에 맞추면 하늘이 하얗게 날아갑니다.
> Window/Level은 "어디에 초점을 맞출지" 정하는 것입니다.

**본 프로젝트의 설정값**:
```python
window_center = 40    # 중심값: 연조직(폐) 기준
window_width = 800    # 범위: 넓게 설정하여 다양한 조직 표현
```

### 2.3 딥러닝 이미지 분류 모델

본 프로젝트는 3가지 사전학습된 모델을 사용합니다:

#### ResNet18 (Residual Network)
```
특징: "잔차 학습" - 입력을 출력에 더해주는 지름길(skip connection) 사용
장점: 깊은 네트워크에서도 학습이 잘 됨
구조: 18개 레이어로 구성 (비교적 가벼움)
```

#### EfficientNet-B0
```
특징: 네트워크의 깊이/너비/해상도를 균형있게 확장
장점: 적은 파라미터로 높은 성능
구조: 자동 탐색된 최적 구조
```

#### DenseNet121 (Dense Network)
```
특징: 모든 이전 레이어의 출력을 현재 레이어 입력으로 연결
장점: 특징 재사용으로 효율적
구조: 121개 레이어, Dense Block 사용
```

**왜 3개 모델을 사용하나?**

각 모델은 이미지의 다른 특징에 집중합니다:
- ResNet: 전체적인 패턴
- EfficientNet: 효율적인 멀티스케일 특징
- DenseNet: 세밀한 국소 특징

→ **3개 모델이 같은 영역을 지목하면 신뢰도가 높아집니다**

### 2.4 Grad-CAM이란?

**정의**: Gradient-weighted Class Activation Mapping의 약자로, **AI가 어디를 보고 판단했는지 시각화**하는 기법입니다.

**작동 원리 (비유)**:
> 학생이 시험 문제를 풀 때, 문제의 어느 부분에 밑줄을 긋고 집중했는지 보여주는 것과 같습니다.
> AI가 "폐렴"이라고 답했을 때, 이미지의 어느 부분에 "밑줄"을 그었는지 히트맵으로 보여줍니다.

**기술적 원리**:
```
1. 이미지를 모델에 입력
2. 예측 결과에 대한 기울기(gradient) 계산
3. 마지막 컨볼루션 레이어의 특징맵과 기울기를 결합
4. 중요도 맵(CAM) 생성 → 히트맵으로 시각화
```

**결과 해석**:
- 🔴 빨간색: 모델이 **가장 주목**한 영역 (판단에 큰 영향)
- 🟡 노란색: 중간 정도 주목
- 🔵 파란색: 거의 주목하지 않은 영역

---

## 3. 전체 코드 흐름도

### 3.1 사용자 관점의 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                        사용자 화면 (main.py)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ① 이미지 업로드        ② 모델 선택         ③ 결과 확인         │
│   ┌──────────┐         ┌──────────┐        ┌──────────────┐    │
│   │  DICOM   │         │ ResNet18 │        │ 원본 | 히트맵 │    │
│   │   또는    │   →     │ Efficient│   →    │              │    │
│   │ JPG/PNG  │         │ DenseNet │        │ 폐렴: 85%    │    │
│   └──────────┘         └──────────┘        └──────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓ 내부 처리
┌─────────────────────────────────────────────────────────────────┐
│                   백엔드 엔진 (gradcam_core.py)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   A. 이미지 로딩        B. 전처리           C. Grad-CAM 실행      │
│   ┌──────────┐        ┌──────────┐        ┌──────────────┐     │
│   │ DICOM→HU │   →    │ 리사이즈  │   →    │ 모델 추론     │     │
│   │ →RGB변환  │        │ 정규화    │        │ 기울기 계산   │     │
│   └──────────┘        │ 텐서변환  │        │ 히트맵 생성   │     │
│                       └──────────┘        └──────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 데이터 처리 파이프라인

```
원본 파일
    │
    ▼
┌───────────────────┐
│ 1. 파일 형식 판별  │  .dcm → DICOM 처리
│    (DICOM/일반)   │  .jpg/.png → 일반 이미지 처리
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 2. DICOM 전처리   │  (DICOM인 경우만)
│  - HU 변환        │  pixel × slope + intercept
│  - 윈도우 적용    │  center=40, width=800
│  - 0~255 스케일링 │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 3. PIL 이미지 변환 │  RGB 3채널로 통일
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 4. 모델 입력 전처리│
│  - 224×224 리사이즈│  모델 입력 크기 통일
│  - ToTensor       │  [0,1] 범위로 변환
│  - Normalize      │  ImageNet 평균/표준편차 적용
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 5. Grad-CAM 실행  │
│  - Forward pass   │  예측값 계산
│  - Backward pass  │  기울기 계산
│  - CAM 생성       │  가중치 × 활성화맵
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 6. 시각화         │
│  - 히트맵 생성    │  JET 컬러맵 적용
│  - 오버레이 합성  │  원본 + 히트맵 블렌딩
└───────────────────┘
```

---

## 4. gradcam_core.py 상세 설명

### 4.1 전역 설정 (라인 42-79)

```python
# GPU 사용 가능하면 GPU, 아니면 CPU 사용
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 분류 클래스 정의
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Pneumonia"]  # 0: 정상, 1: 폐렴
```

**왜 이렇게 설정했나?**
- 이진 분류(정상 vs 폐렴) 문제이므로 클래스는 2개
- GPU가 있으면 계산이 10~100배 빨라짐

```python
# 모델별 설정
MODEL_CONFIG = {
    "ResNet18": {
        "builder": models.resnet18,           # 모델 생성 함수
        "checkpoint": "checkpoints/...",       # 학습된 가중치 파일
        "target_layer": "layer4",              # Grad-CAM을 적용할 레이어
        "input_size": 224,                     # 입력 이미지 크기
    },
    # ... 다른 모델들도 동일한 구조
}
```

**target_layer 설정 근거**:

| 모델 | target_layer | 이유 |
|------|--------------|------|
| ResNet18 | layer4 | 마지막 잔차 블록, 가장 추상화된 특징 포함 |
| EfficientNet-B0 | features | 전체 특징 추출부의 마지막 |
| DenseNet121 | features | Dense Block들의 최종 출력 |

> **왜 마지막 레이어를 사용하나?**
>
> CNN은 앞쪽 레이어에서 선/엣지 같은 단순 특징을, 뒤쪽 레이어에서 "폐렴 패턴" 같은 고수준 특징을 학습합니다.
> 분류 판단에 가장 직접적인 영향을 주는 것은 마지막 특징맵이므로, 여기에 Grad-CAM을 적용합니다.

```python
# 모델별 최적 임계값 (학습 결과에서 도출)
BEST_THRESHOLDS = {
    "ResNet18": 0.25,
    "EfficientNet-B0": 0.15,
    "DenseNet121": 0.18,
}
```

**임계값(threshold)이란?**

모델의 출력은 0~1 사이의 확률값입니다:
- 출력값 ≥ 임계값 → "폐렴"으로 분류
- 출력값 < 임계값 → "정상"으로 분류

**왜 0.5가 아닌 다른 값을 사용하나?**

학습 데이터의 특성과 모델 성향에 따라 최적 임계값이 다릅니다:
- 0.25, 0.15, 0.18은 **학습 과정에서 F1-score나 Youden's J를 최대화**하는 값으로 도출됨
- 의료 진단에서는 민감도(Sensitivity)가 중요하므로 임계값을 낮춰 "놓치지 않는 것"을 우선시

---

### 4.2 DICOM 로딩 함수 (라인 93-124)

```python
def dicom_to_pil(file_bytes: bytes) -> Tuple[Image.Image, Any]:
    """
    DICOM 바이트 → (PIL Image RGB, DICOM Dataset)
    """
```

**단계별 설명**:

#### 단계 1: DICOM 파일 읽기
```python
dcm = pydicom.dcmread(io.BytesIO(file_bytes))
img = dcm.pixel_array.astype(np.float32)
```
- `pydicom`: DICOM 파일을 읽는 전문 라이브러리
- `pixel_array`: DICOM의 원시 픽셀 데이터 추출
- `float32`: 정밀한 계산을 위해 실수형으로 변환

#### 단계 2: HU 변환
```python
if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
    slope = float(dcm.RescaleSlope)
    intercept = float(dcm.RescaleIntercept)
    img = img * slope + intercept
```

**왜 필요한가?**

DICOM 파일은 저장 공간 효율을 위해 변환된 값을 저장합니다:
```
저장값 = (실제값 - Intercept) / Slope
```
따라서 분석 시에는 역변환이 필요합니다:
```
실제값(HU) = 저장값 × Slope + Intercept
```

**HU(Hounsfield Unit) 참고값**:
| 조직 | HU 값 |
|------|-------|
| 공기 | -1000 |
| 지방 | -100 ~ -50 |
| 물 | 0 |
| 근육 | +40 |
| 뼈 | +400 ~ +1000 |

#### 단계 3: Window/Level 적용
```python
window_center = 40.0    # 중심값
window_width = 800.0    # 범위

min_val = window_center - window_width / 2.0  # = -360
max_val = window_center + window_width / 2.0  # = +440

img = (img - min_val) / (max_val - min_val)   # 0~1로 정규화
img = np.clip(img, 0.0, 1.0)                  # 범위 제한
img = (img * 255.0).astype(np.uint8)          # 0~255로 스케일링
```

**파라미터 설정 근거**:

| 파라미터 | 값 | 근거 |
|----------|-----|------|
| window_center | 40 | 폐/연조직의 중심 HU값 |
| window_width | 800 | 흉부 X-ray에서 폐, 심장, 뼈를 모두 볼 수 있는 넓은 범위 |

> **주의**: 학습 시 사용한 전처리와 **동일한 설정**을 사용해야 합니다!
> 다른 설정을 사용하면 모델 성능이 크게 저하됩니다.

#### 단계 4: RGB 변환
```python
pil_img = Image.fromarray(img).convert("RGB")
```

**왜 RGB로 변환하나?**

- X-ray는 원래 **흑백(1채널)**이지만
- 사전학습된 모델들은 **컬러 이미지(3채널)**로 학습됨
- 따라서 흑백을 RGB 3채널로 복제 (R=G=B)

---

### 4.3 Grad-CAM 클래스 (라인 140-229)

```python
class GradCAM:
    """
    기본 Grad-CAM 구현
    """
```

#### 초기화 (__init__)

```python
def __init__(self, model: nn.Module, target_layer: nn.Module):
    self.model = model
    self.target_layer = target_layer
    self.activations = None   # 순전파 시 저장될 활성화값
    self.gradients = None     # 역전파 시 저장될 기울기
```

**Hook 등록**:
```python
def forward_hook(module, inputs, output):
    self.activations = output.detach()      # 활성화값 저장
    output.register_hook(self._save_gradients)  # 기울기 저장 hook 추가

self.target_layer.register_forward_hook(forward_hook)
```

**Hook이란?**

PyTorch에서 특정 레이어의 입출력을 "가로채서" 저장하는 기능입니다.
- `forward_hook`: 순전파(입력→출력) 시 실행
- `register_hook`: 역전파(기울기 계산) 시 실행

**비유**: CCTV처럼 특정 지점을 모니터링하면서 데이터를 기록합니다.

#### 호출 (__call__)

```python
def __call__(self, x, target_class=None, threshold=0.5):
```

**전체 흐름**:

```
① Forward Pass (순전파)
   ┌─────────┐      ┌──────────────┐      ┌─────────┐
   │  입력   │  →   │  target_layer │  →   │  출력   │
   │ 이미지  │      │  (활성화 저장)│      │ (logit) │
   └─────────┘      └──────────────┘      └─────────┘
                          ↓
                    activations 저장

② Backward Pass (역전파)
   ┌─────────┐      ┌──────────────┐      ┌─────────┐
   │ 기울기  │  ←   │  target_layer │  ←   │  loss   │
   │         │      │  (기울기 저장)│      │(score)  │
   └─────────┘      └──────────────┘      └─────────┘
                          ↓
                    gradients 저장

③ CAM 계산
   weights = 기울기의 공간 평균 (어떤 채널이 중요한지)
   CAM = Σ(weights × activations)  (가중 합)
   CAM = ReLU(CAM)  (음수 제거 - 긍정적 영향만)
```

**코드 상세**:

```python
# 1. Forward Pass
logits = self.model(x)  # 모델 예측 실행, shape: (1, 1)

# 2. 확률 계산 (이진 분류)
if logits.shape[1] == 1:
    prob_pos = torch.sigmoid(logits)[0, 0]  # 폐렴 확률 (0~1)

    # 임계값 기반 분류
    if target_class is None:
        target_class = int((prob_pos >= threshold).item())
        # prob_pos=0.3, threshold=0.25 → 0.3≥0.25 → True → 1(폐렴)
```

**Sigmoid 함수**:
```
              1
sigmoid(x) = ───────
             1 + e^(-x)

x = -2 → 0.12 (낮은 확률)
x = 0  → 0.50 (중간)
x = 2  → 0.88 (높은 확률)
```

```python
# 3. Backward Pass (기울기 계산)
score = logits[0, 0]
score.backward(retain_graph=True)
```

**왜 score에 대해 backward를 하나?**

"폐렴 점수를 높이려면 어떤 픽셀이 중요한가?"에 대한 답을 구하는 것입니다.
→ 이 기울기가 큰 영역 = 폐렴 판단에 중요한 영역

```python
# 4. CAM 계산
act = self.activations    # [1, C, H, W] - C개 채널의 특징맵
grad = self.gradients     # [1, C, H, W] - 각 채널의 기울기

# Global Average Pooling으로 채널별 가중치 계산
weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
# dim=(2,3): 높이(H)와 너비(W)에 대해 평균 → 채널별 중요도

# 가중합으로 CAM 생성
cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, H, W]
# 각 채널을 중요도에 따라 가중합

cam = F.relu(cam)  # 음수 제거 (양의 기여만 시각화)
```

**직관적 설명**:

1. **활성화맵(act)**: 각 채널이 "어디에서" 반응했는지 (위치 정보)
2. **기울기(grad)**: 각 채널이 "얼마나 중요한지" (가중치)
3. **가중합**: 중요한 채널의 활성화만 합산 → 판단 근거 영역

```python
# 5. 정규화 (0~1 범위로)
cam = cam - cam.min()
if cam.max() > 0:
    cam = cam / cam.max()

cam_np = cam.squeeze().cpu().numpy()  # 텐서 → NumPy 배열
```

---

### 4.4 모델 빌드 함수 (라인 235-258)

```python
def _build_model(model_name: str) -> nn.Module:
```

**핵심 과정**:

```python
# 1. 기본 모델 생성 (사전학습 가중치 없이)
model = builder(weights=None)

# 2. 마지막 레이어를 이진분류용으로 교체
if isinstance(model, models.ResNet):
    in_features = model.fc.in_features      # 원래 입력 차원 (512)
    model.fc = nn.Linear(in_features, 1)    # 출력을 1개로 변경
```

**왜 출력을 1개로 하나?**

| 방식 | 출력 개수 | 설명 |
|------|----------|------|
| 다중 클래스 | 2개 (정상, 폐렴) | softmax로 확률 계산 |
| **이진 분류** | **1개** (폐렴 점수) | sigmoid로 확률 계산 |

본 프로젝트는 이진 분류 방식을 사용:
- 출력값 = "폐렴일 확률"
- 정상 확률 = 1 - 폐렴 확률

**각 모델의 마지막 레이어 위치**:

| 모델 | 원래 구조 | 수정 방법 |
|------|----------|----------|
| ResNet18 | `model.fc` | `nn.Linear(512, 1)` |
| EfficientNet-B0 | `model.classifier[1]` | `nn.Linear(1280, 1)` |
| DenseNet121 | `model.classifier` | `nn.Linear(1024, 1)` |

---

### 4.5 체크포인트 로딩 (라인 261-281)

```python
def _load_state_dict(model: nn.Module, ckpt_path: str) -> None:
```

```python
# 체크포인트 로드 (GPU/CPU 호환)
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

# 체크포인트 형식에 따라 state_dict 추출
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt  # 순수 state_dict

# 모델에 가중치 적용
model.load_state_dict(state_dict)  # strict=True (기본값)
```

**strict=True의 의미**:
- 모델 구조와 체크포인트가 **완전히 일치**해야 함
- 하나라도 다르면 에러 발생
- **성능 보존**을 위해 반드시 True 사용

---

### 4.6 이미지 전처리 Transform (라인 314-321)

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),           # 크기 통일
    transforms.ToTensor(),                    # [0,255] → [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],           # ImageNet 평균
        std=[0.229, 0.224, 0.225],            # ImageNet 표준편차
    ),
])
```

**각 단계 설명**:

| 단계 | 입력 | 출력 | 목적 |
|------|------|------|------|
| Resize | 다양한 크기 | 224×224 | 모델 입력 크기 통일 |
| ToTensor | PIL (0~255) | Tensor (0~1) | PyTorch 연산 가능하게 |
| Normalize | (0~1) | (약 -2~2) | 사전학습 분포에 맞춤 |

**Normalize 값의 근거**:

```
ImageNet 데이터셋 120만장의 통계:
- R채널 평균: 0.485
- G채널 평균: 0.456
- B채널 평균: 0.406
- R채널 표준편차: 0.229
- G채널 표준편차: 0.224
- B채널 표준편차: 0.225

정규화 공식: (픽셀값 - 평균) / 표준편차
```

**왜 ImageNet 값을 사용하나?**

사전학습된 모델들(ResNet, EfficientNet, DenseNet)은 ImageNet으로 학습되었습니다.
같은 정규화를 적용해야 학습된 특징을 제대로 활용할 수 있습니다.

---

### 4.7 히트맵 생성 함수 (라인 341-357)

```python
def apply_colormap_on_image(gray_cam, rgb_img, alpha=0.4):
```

**단계별 처리**:

```python
# 1. CAM을 원본 이미지 크기로 리사이즈
h, w, _ = rgb_img.shape
cam_resized = cv2.resize(gray_cam, (w, h))

# 2. 컬러맵 적용 (JET: 파랑→초록→노랑→빨강)
heatmap_uint8 = np.uint8(255 * cam_resized)  # 0~1 → 0~255
heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

# 3. 원본과 블렌딩 (오버레이)
overlay = (heatmap_rgb * alpha + rgb_img * (1 - alpha)).astype(np.uint8)
```

**alpha 파라미터**:
- 0.0: 원본만 보임
- 0.5: 반반 합성
- 1.0: 히트맵만 보임
- **기본값 0.4**: 원본 윤곽을 유지하면서 히트맵 강조

**JET 컬러맵**:
```
값:    0.0 ──── 0.25 ──── 0.5 ──── 0.75 ──── 1.0
색상:  🔵 ──── 🟢 ──── 🟡 ──── 🟠 ──── 🔴
의미:  낮은 주목 ─────────────────────── 높은 주목
```

---

## 5. main.py 상세 설명

### 5.1 Streamlit 앱 설정 (라인 19-23)

```python
st.set_page_config(
    page_title="X-ray Grad-CAM Explorer",   # 브라우저 탭 제목
    layout="wide",                           # 넓은 레이아웃
    initial_sidebar_state="expanded"         # 사이드바 기본 펼침
)
```

### 5.2 다크 테마 CSS (라인 68-266)

```python
DARK_CSS = """
<style>
:root {
    --bg-main: #050608;        /* 메인 배경: 거의 검정 */
    --accent: #4FD1C5;         /* 강조색: 청록색 */
    --text-main: #E5E7EB;      /* 텍스트: 밝은 회색 */
}
...
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)
```

**왜 다크 테마를 사용하나?**
- X-ray는 어두운 배경에서 더 잘 보임
- 의료 영상 뷰어의 관례
- 장시간 사용 시 눈의 피로 감소

### 5.3 사이드바 UI (라인 511-536)

```python
# 파일 업로더
uploaded_file = st.sidebar.file_uploader(
    "",
    type=["dcm", "png", "jpg", "jpeg"],  # 허용 확장자
)

# 보기 모드 선택
view_mode = st.sidebar.radio(
    "View Mode",
    ["단일 모델", "3개 모델 비교"],
)

# 모델 선택
model_name = st.sidebar.selectbox(
    "Model Selection",
    list(MODEL_CONFIG.keys()),  # ["ResNet18", "EfficientNet-B0", "DenseNet121"]
)

# 오버레이 투명도 조절
alpha = st.sidebar.slider(
    "Heatmap Overlay Alpha",
    min_value=0.1, max_value=0.9, value=0.45, step=0.05
)
```

### 5.4 @st.cache_resource 데코레이터 (라인 395)

```python
@st.cache_resource
def load_model(model_name: str):
```

**cache_resource란?**

모델을 한 번 로드하면 **메모리에 캐싱**하여 재사용합니다.
- 첫 번째 요청: 모델 로드 (느림)
- 이후 요청: 캐시에서 가져옴 (빠름)

**왜 필요한가?**
- 모델 로드는 시간이 오래 걸림 (수 초)
- 이미지 업로드마다 매번 로드하면 매우 느림
- 캐싱으로 첫 로드 후 즉시 응답 가능

### 5.5 탭 구성 (라인 592-593)

```python
tab_viz, tab_pred, tab_meta = st.tabs([
    "🖼 Grad-CAM 시각화",   # 히트맵 결과
    "📊 예측 결과",         # 확률 바 차트
    "ℹ 메타데이터"          # 파일 정보
])
```

### 5.6 3개 모델 비교 모드 (라인 626-650)

```python
# 3개 열로 나누기
cols = st.columns(3)
results = []

# 각 모델에 대해 Grad-CAM 실행
for m in MODEL_CONFIG.keys():
    results.append(run_gradcam_for_model(m, base_img, alpha))

# 결과 표시
for col, res in zip(cols, results):
    with col:
        st.image(res["overlay"])
        st.markdown(f"Pred: {CLASS_NAMES[res['target_class']]}")
```

---

## 6. 파라미터 설정 근거

### 6.1 DICOM 처리 파라미터

| 파라미터 | 값 | 설정 근거 |
|----------|-----|----------|
| `window_center` | 40 | 폐/연조직의 평균 HU값. 흉부 X-ray에서 가장 중요한 폐 영역을 중심으로 설정 |
| `window_width` | 800 | 넓은 범위(-360~440 HU)를 표현하여 폐, 심장, 뼈, 연조직을 모두 시각화 |

**대안 설정과 비교**:

| 용도 | center | width | 범위 |
|------|--------|-------|------|
| **본 프로젝트** | 40 | 800 | -360 ~ 440 |
| 폐 전용 | -600 | 1500 | -1350 ~ 150 |
| 종격동 전용 | 50 | 350 | -125 ~ 225 |

### 6.2 모델 입력 파라미터

| 파라미터 | 값 | 설정 근거 |
|----------|-----|----------|
| `input_size` | 224 | ImageNet 사전학습 표준 크기. 대부분의 CNN 모델이 이 크기로 학습됨 |
| `mean` | [0.485, 0.456, 0.406] | ImageNet 120만장 평균값. Transfer learning 시 필수 |
| `std` | [0.229, 0.224, 0.225] | ImageNet 120만장 표준편차. Transfer learning 시 필수 |

### 6.3 분류 임계값 (Threshold)

| 모델 | 임계값 | 설정 근거 |
|------|--------|----------|
| ResNet18 | 0.25 | 학습 시 ROC 곡선에서 Youden's J index가 최대인 점 |
| EfficientNet-B0 | 0.15 | 동일 방법으로 도출. 모델이 "자신있게" 예측하는 경향 |
| DenseNet121 | 0.18 | 동일 방법으로 도출 |

**Youden's J index란?**
```
J = Sensitivity + Specificity - 1

Sensitivity: 실제 폐렴 환자 중 폐렴으로 맞춘 비율
Specificity: 실제 정상인 중 정상으로 맞춘 비율

J가 최대인 지점 = 두 가지를 균형있게 잘 맞추는 최적 임계값
```

**의료 맥락에서의 고려**:
- 임계값을 낮추면: Sensitivity ↑ (폐렴 놓치지 않음), Specificity ↓ (오진 증가)
- 임계값을 높이면: Sensitivity ↓ (폐렴 놓칠 수 있음), Specificity ↑ (오진 감소)
- 보통 의료에서는 **"놓치지 않는 것"**이 더 중요 → 낮은 임계값 선호

### 6.4 Grad-CAM 파라미터

| 파라미터 | 값 | 설정 근거 |
|----------|-----|----------|
| `alpha` (오버레이) | 0.4~0.45 | 원본 구조를 60% 유지하면서 히트맵 강조 |
| `target_layer` | 각 모델의 마지막 특징 레이어 | 가장 추상화된 (의미있는) 특징 포함 |
| 컬러맵 | JET | 의료 영상 분석에서 가장 널리 사용되는 표준 |

### 6.5 모델 선택 근거

| 모델 | 파라미터 수 | 특징 | 선택 이유 |
|------|------------|------|----------|
| ResNet18 | 11M | 빠름, 안정적 | 기본 베이스라인, 빠른 추론 |
| EfficientNet-B0 | 5M | 효율적 | 적은 파라미터로 좋은 성능 |
| DenseNet121 | 8M | 특징 재사용 | 세밀한 패턴 감지에 강점 |

---

## 7. 자주 묻는 질문 (FAQ)

### Q1. 왜 출력이 2개가 아니라 1개인가요?

**A**: 이진 분류에서 두 가지 방식이 있습니다:

| 방식 | 출력 | 확률 계산 |
|------|------|----------|
| 방식 1 | 2개 (정상, 폐렴) | softmax |
| **방식 2** | **1개 (폐렴 점수)** | **sigmoid** |

본 프로젝트는 방식 2를 사용합니다.
- 폐렴 확률 = sigmoid(출력값)
- 정상 확률 = 1 - 폐렴 확률

방식 2가 이진 분류에서 더 흔하게 사용되며, 학습 시 이 방식으로 훈련되었습니다.

---

### Q2. ImageNet 정규화를 왜 의료 영상에 사용하나요?

**A**: Transfer Learning(전이학습) 때문입니다.

```
사전학습 (ImageNet, 120만장)
         ↓
    일반적인 특징 학습 완료
    (선, 모서리, 패턴 등)
         ↓
미세조정 (X-ray, 수천~수만장)
         ↓
    의료 특화 특징 추가 학습
```

- 사전학습된 가중치를 활용하려면 **동일한 전처리**가 필요
- ImageNet으로 학습된 모델이 "기대하는" 입력 분포에 맞춰야 함
- 다른 정규화 사용 시 성능 급격히 저하

---

### Q3. threshold가 0.5가 아닌 이유는?

**A**: 데이터 불균형과 모델 특성 때문입니다.

실제 의료 데이터는 보통 정상 > 질병입니다:
```
예: 정상 80%, 폐렴 20%
```

이런 불균형에서 모델은 "정상"을 예측하면 정확도가 높아지므로,
폐렴 확률을 낮게 출력하는 경향이 생깁니다.

최적 임계값은 학습 데이터에서 **ROC 곡선 분석**으로 결정:
```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
youden_j = tpr - fpr  # Youden's J index
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]
```

---

### Q4. Grad-CAM 결과가 모델마다 다른 이유는?

**A**: 각 모델의 **학습 방식과 구조**가 다르기 때문입니다.

| 모델 | 특성 | Grad-CAM 경향 |
|------|------|--------------|
| ResNet18 | 잔차 연결 | 넓은 영역 활성화 |
| EfficientNet-B0 | 복합 스케일링 | 다양한 크기 패턴 감지 |
| DenseNet121 | 밀집 연결 | 세밀한 국소 영역 강조 |

**실무 활용**:
- 3개 모델이 **같은 영역**을 지목 → 높은 신뢰도
- 3개 모델이 **다른 영역** 지목 → 추가 검토 필요

---

### Q5. CUDA/GPU가 없어도 작동하나요?

**A**: 네, CPU에서도 작동합니다.

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- GPU 있음: 빠른 추론 (수십 ms)
- GPU 없음: CPU 사용 (수백 ms ~ 수 초)

웹 데모 수준에서는 CPU로도 충분히 사용 가능합니다.

---

### Q6. 왜 `strict=True`로 체크포인트를 로드하나요?

**A**: **성능 보존**을 위해서입니다.

| 옵션 | 동작 | 위험성 |
|------|------|--------|
| `strict=True` | 모든 가중치 완전 일치 필수 | 안전, 성능 보장 |
| `strict=False` | 일치하는 것만 로드 | 누락된 가중치 → 성능 저하 |

학습된 모델의 성능을 그대로 재현하려면 `strict=True`가 필수입니다.

---

### Q7. 히트맵의 빨간 영역이 항상 병변인가요?

**A**: **반드시 그렇지는 않습니다.**

Grad-CAM은 모델의 **"판단 근거"**를 보여주는 것이지, **"정답"**을 보여주는 것이 아닙니다.

**해석 시 주의사항**:
- 빨간 영역 = 모델이 "폐렴"이라고 판단할 때 주목한 영역
- 모델이 잘못 판단했다면, 빨간 영역도 틀릴 수 있음
- 항상 **전문의의 판단**과 함께 참고 자료로만 사용

---

## 부록: 용어 정리

| 용어 | 설명 |
|------|------|
| **DICOM** | 의료 영상 국제 표준 포맷 (Digital Imaging and Communications in Medicine) |
| **HU** | Hounsfield Unit, CT 밀도 측정 단위 (물=0, 공기=-1000) |
| **Window/Level** | 특정 HU 범위만 화면에 표시하는 기법 |
| **CNN** | Convolutional Neural Network, 이미지 분석에 특화된 신경망 |
| **Grad-CAM** | 딥러닝 모델의 판단 근거를 시각화하는 기법 |
| **Transfer Learning** | 다른 데이터로 학습된 모델을 재활용하는 기법 |
| **Sigmoid** | 출력을 0~1 확률로 변환하는 함수 |
| **Threshold** | 확률을 클래스로 변환하는 기준값 |
| **Hook** | PyTorch에서 레이어 입출력을 가로채는 기능 |
| **Sensitivity** | 민감도, 실제 양성 중 맞춘 비율 (= Recall) |
| **Specificity** | 특이도, 실제 음성 중 맞춘 비율 |

---

*문서 작성일: 2025-11-23*
*대상: 비전공자, 의료영상 분석 입문자*
