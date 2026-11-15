import streamlit as st

from analysis import PROJECT, QUESTION, QUESTIONS, SOURCE, TITLE
from ui import get_data, inject_theme, page_header, section_title

st.set_page_config(page_title=TITLE, page_icon="🦾", layout="wide", initial_sidebar_state="expanded")
inject_theme()

try:
    raw, prepared, clean = get_data()
except (FileNotFoundError, ValueError) as error:
    st.error(f"데이터를 읽을 수 없습니다: {error}")
    st.info("data/ur3_cobotops.csv 파일이 저장소의 data 폴더에 있는지 확인하세요.")
    st.stop()


def home():
    page_header(
        "UR3 COBOTOPS · DATA SCIENCE PROJECT",
        "센서 데이터로 읽는 그리퍼 파지 손실",
        "19개 센서의 관측값을 탐색하고, 로지스틱 회귀와 의사결정트리로 정상·파지 손실 상태를 분류합니다. "
        "변수 간 관계부터 모델의 판단 근거까지 한 화면 흐름으로 분석합니다."
    )

    section_title("데이터 한눈에 보기")
    a, b, c, d = st.columns(4)
    a.metric("원본 관측 행", f"{len(raw):,}")
    b.metric("분석 사용 행", f"{len(clean):,}")
    c.metric("입력 센서", f"19개")
    d.metric("파지 손실 행 비율", f"{clean.grip_lost.mean():.2%}")
    st.caption("한 행은 한 시점의 관측입니다. 연속된 여러 행이 같은 손실 사건을 기록할 수 있습니다.")

    section_title("분석 워크스페이스")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("### 01 · 데이터 탐색")
            st.write("결측값, 극단값 처리, 센서별 분포와 상관관계를 확인합니다.")
            st.page_link("pages/1_탐색.py", label="데이터 탐색 열기", icon="📊", use_container_width=True)
        with st.container(border=True):
            st.markdown("### 02 · 변수 관계 분석")
            st.write("산점도, 히스토그램, 상자그림, 바이올린, 상관관계와 시간 흐름을 비교합니다.")
            st.page_link("pages/3_변수_관계_분석.py", label="시각화 분석 열기", icon="📈", use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown("### 03 · 분류 모델")
            st.write("두 모델을 동일한 시간순 평가 자료에 적용하고 성능과 혼동 행렬을 비교합니다.")
            st.page_link("pages/2_분류_모델.py", label="모델 평가 열기", icon="🌳", use_container_width=True)
        with st.container(border=True):
            st.markdown("### 04 · 분석 질문")
            st.write("센서의 분포 차이와 변수 관계가 실제 분류 결과에서 어떻게 나타나는지 살펴봅니다.")
            for i, question in enumerate(QUESTIONS, 1):
                st.markdown(f"**{i}.** {question}")

    st.info("분석 목표는 센서가 기록된 시점의 grip_lost 상태를 분류하는 것입니다. 미래 손실 예측이나 실제 로봇 제어 성능을 검증한 것은 아닙니다.")
    st.markdown(f"[UCI UR3 CobotOps 원자료 ↗]({SOURCE})")


page = st.navigation([
    st.Page(home, title="프로젝트 개요", icon="🦾", default=True),
    st.Page("pages/1_탐색.py", title="데이터 탐색", icon="📊", url_path="explore"),
    st.Page("pages/3_변수_관계_분석.py", title="변수 관계 분석", icon="📈", url_path="variables"),
    st.Page("pages/2_분류_모델.py", title="분류 모델", icon="🌳", url_path="models"),
])

with st.sidebar:
    st.markdown(
        '<div class="ur3-sidebar-brand"><strong>🦾 UR3 INSIGHT</strong><span>Grip-loss analytics workspace</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("PROJECT DATA")
    st.markdown("**UR3 CobotOps**")
    st.caption("19 sensor features · Binary classification")
    st.divider()
    st.caption("LABEL GUIDE")
    st.markdown("🟢 정상 · `False / 0`")
    st.markdown("🟠 파지 손실 · `True / 1`")
    st.divider()
    st.caption("분석 데이터는 제공된 CSV를 사용합니다. 작업 주기 기준 시간순 80:20 분할을 적용합니다.")

page.run()
