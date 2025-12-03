"""
UI 스타일 모듈
==============

Streamlit 앱의 CSS 스타일 정의.
다크 테마 + X-ray 스타일 적용.

확장 예시:
- 새 테마 추가: LIGHT_CSS 등 추가
- 컴포넌트별 스타일 분리
"""

# ============================================================
# 다크 테마 CSS (X-ray 스타일)
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

/* 사이드바 수납(collapse) 버튼 - 항상 표시 */
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
[data-testid="stSidebar"] button[kind="headerNoPadding"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] > div > div > div > button,
section[data-testid="stSidebar"] svg {
    opacity: 1 !important;
    visibility: visible !important;
}

/* 사이드바 헤더 영역 - hover 없이도 버튼 표시 */
[data-testid="stSidebarHeader"],
[data-testid="stSidebarHeader"] * {
    opacity: 1 !important;
    visibility: visible !important;
}

/* collapse 버튼 컨테이너 강제 표시 */
[data-testid="stSidebar"] > div:first-child {
    opacity: 1 !important;
}

[data-testid="stSidebar"] > div:first-child button {
    opacity: 1 !important;
    visibility: visible !important;
    pointer-events: auto !important;
}

/* 사이드바 카드/컨테이너 */
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

/* 라디오/셀렉트박스 */
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

/* 탭 컨테이너 */
.stTabs [data-baseweb="tab-list"] {
    border-radius: 0 !important;
    background-color: transparent !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

.stTabs [data-baseweb="tab-list"] button {
    border-radius: 0 !important;
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0.4rem 0.9rem !important;
}

.stTabs [data-baseweb="tab"] {
    border-bottom: none !important;
    background-color: transparent !important;
}

/* 활성 탭 */
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: transparent !important;
    color: #F9FAFB !important;
    font-weight: 600 !important;
}

/* 비활성 탭 */
.stTabs [data-baseweb="tab-list"] button[aria-selected="false"] {
    background-color: transparent !important;
    color: var(--text-muted) !important;
    font-weight: 400 !important;
}

/* X-ray 패널 */
.xray-panel {
    padding: 0;
    margin: 0 0 6px 0;
    border: none;
    background: transparent;
}

.xray-panel-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

/* 정보 카드 */
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

/* 뱃지 */
.badge-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    border: 1px solid var(--border-subtle);
    color: var(--text-muted);
}

/* Grad-CAM 트리거 카드 (사용 안 해도 스타일 유지) */
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

/* 제목 섹션 */
h3 {
    font-size: 1.35rem !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.6rem !important;
}

/* 페이지 상단 패딩 */
.block-container {
    padding-top: 2.5rem !important;
}
</style>
"""


def get_css(theme: str = "dark") -> str:
    """
    테마별 CSS 반환

    Args:
        theme: 테마 이름 ("dark" 지원, 확장 가능)

    Returns:
        CSS 문자열
    """
    themes = {
        "dark": DARK_CSS,
        # 확장 예시:
        # "light": LIGHT_CSS,
    }
    return themes.get(theme, DARK_CSS)
