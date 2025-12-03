# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 기술 스택
- Python 3.x
- Streamlit 1.51.0
- PyTorch, torchvision
- OpenCV-headless, Pillow, pydicom
- 배포: HuggingFace Spaces

## 프로젝트 구조 (모듈 분리 설계)
```
xraygradcam/
├── main.py              # UI 전담 (Streamlit)
├── grandcam_core.py     # 비즈니스 로직 (Grad-CAM 계산)
├── config.py            # 설정 중앙 관리 (모델, threshold, DICOM)
├── styles.py            # CSS 스타일 관리 (테마)
├── checkpoints/         # 모델 가중치 (Git LFS)
├── requirements.txt
└── .github/workflows/   # HuggingFace 자동 동기화
```

## 모듈 역할
| 모듈 | 역할 | 확장 포인트 |
|------|------|-------------|
| config.py | 설정 중앙 관리 | 모델 추가 시 MODEL_CONFIG에 항목 추가 |
| styles.py | CSS 테마 관리 | 새 테마 추가 가능 (LIGHT_CSS 등) |
| grandcam_core.py | Grad-CAM 로직 | FastAPI 등 다른 프레임워크에서 재사용 |
| main.py | Streamlit UI | UI 변경 시 이 파일만 수정 |

## 핵심 명령어
```bash
streamlit run main.py           # 로컬 실행
pip install -r requirements.txt # 의존성 설치
```

## 코딩 스타일
- 파일명/함수명: snake_case
- 클래스명: PascalCase
- 주석: 한국어

## 프로젝트 특이사항
- 폐렴/정상 이진분류 Grad-CAM 시각화 데모
- **확장성 설계**: config.py에서 모델 추가만으로 새 모델 지원
- DICOM 전처리: `config.DICOM_WINDOW_CENTER=40, config.DICOM_WINDOW_WIDTH=800`
- 모델별 threshold: `config.BEST_THRESHOLDS` 참조
- 체크포인트는 Git LFS 관리
- GitHub push 시 Actions로 HuggingFace 자동 동기화 (HF_TOKEN 필요)

## 새 모델 추가 방법
1. `config.py`의 `MODEL_CONFIG`에 항목 추가
2. `config.py`의 `BEST_THRESHOLDS`에 threshold 추가
3. 체크포인트 파일을 `checkpoints/`에 배치
4. 끝 (main.py, grandcam_core.py 수정 불필요)
