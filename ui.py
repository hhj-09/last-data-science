"""UR3 Streamlit 공통 UI, 데이터 캐시, 차트 스타일."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from analysis import load_raw, prepare_data, complete_rows, evaluate, outlier_processing

COLORS = {"정상": "#2A9D8F", "파지 손실": "#F08A5D"}
BG = "#F5F7FB"
INK = "#182230"
MUTED = "#718096"


def inject_theme():
    """모든 페이지에 적용되는 공통 디자인 시스템."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
        :root { --ur3-ink:#182230; --ur3-muted:#718096; --ur3-line:#E7ECF3; --ur3-bg:#F5F7FB; --ur3-teal:#2A9D8F; --ur3-orange:#F08A5D; }
        html, body, [class*="css"] { font-family:'DM Sans','Noto Sans KR',sans-serif; color:var(--ur3-ink); }
        .stApp { background:linear-gradient(180deg,#F7F9FC 0%,#F4F7FB 42%,#F8FAFC 100%); }
        [data-testid="stHeader"] { background:rgba(247,249,252,.88); }
        [data-testid="stAppViewContainer"] .main .block-container { max-width:1440px; padding:2.1rem 2.4rem 4rem; }
        [data-testid="stSidebar"] { background:#111C2D; border-right:1px solid #263449; }
        [data-testid="stSidebar"] * { color:#E8EEF7; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#AAB8CA; }
        [data-testid="stSidebar"] [data-testid="stPageLink"] a { border-radius:12px; }
        h1,h2,h3 { letter-spacing:-.035em; color:#182230; }
        h1 { font-weight:800 !important; }
        h2,h3 { font-weight:700 !important; }
        [data-testid="stMetric"] { background:#FFFFFF; border:1px solid var(--ur3-line); border-radius:16px; padding:17px 19px; box-shadow:0 5px 18px rgba(24,34,48,.035); }
        [data-testid="stMetricLabel"] { color:#718096; font-weight:600; }
        [data-testid="stMetricValue"] { color:#182230; font-weight:800; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:7px; border-bottom:1px solid var(--ur3-line); }
        [data-testid="stTabs"] button[role="tab"] { border-radius:10px 10px 0 0; padding:11px 15px; }
        [data-testid="stTabs"] button[aria-selected="true"] { color:#167C72; border-bottom:3px solid #2A9D8F; }
        [data-testid="stVerticalBlockBorderWrapper"] { border-radius:17px; border-color:var(--ur3-line); background:#FFFFFF; box-shadow:0 5px 18px rgba(24,34,48,.025); }
        div[data-testid="stDataFrame"] { border:1px solid var(--ur3-line); border-radius:13px; overflow:hidden; }
        .ur3-hero { padding:27px 30px; border:1px solid #E5EAF2; border-radius:22px; background:linear-gradient(115deg,#FFFFFF 0%,#F0F8F7 62%,#FDF5F0 100%); box-shadow:0 10px 30px rgba(24,34,48,.045); margin:0 0 22px 0; }
        .ur3-eyebrow { color:#168579; text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:800; margin-bottom:9px; }
        .ur3-title { color:#152238; font-size:2.05rem; font-weight:800; letter-spacing:-.055em; line-height:1.25; margin:0 0 10px 0; }
        .ur3-subtitle { color:#64748B; font-size:.98rem; line-height:1.75; max-width:980px; margin:0; }
        .ur3-section { display:flex; align-items:center; gap:10px; margin:26px 0 12px; color:#1C2B3D; font-size:1.12rem; font-weight:800; letter-spacing:-.025em; }
        .ur3-section:before { content:''; width:5px; height:20px; border-radius:8px; background:linear-gradient(180deg,#2A9D8F,#F08A5D); }
        .ur3-note { color:#718096; font-size:.84rem; line-height:1.65; }
        .ur3-pill { display:inline-block; padding:5px 10px; border-radius:999px; background:#EAF6F4; color:#167C72; font-size:.75rem; font-weight:700; }
        .ur3-sidebar-brand { padding:16px 8px 22px; border-bottom:1px solid #2B3A50; margin-bottom:18px; }
        .ur3-sidebar-brand strong { display:block; color:#FFFFFF; font-size:1.05rem; letter-spacing:-.03em; }
        .ur3-sidebar-brand span { color:#AAB8CA; font-size:.76rem; }
        .stButton>button, .stDownloadButton>button { border-radius:11px; font-weight:700; }
        .stButton>button[kind="primary"] { background:#167C72; border-color:#167C72; }
        div[data-testid="stAlert"] { border-radius:13px; }
        @media (max-width: 800px) { [data-testid="stAppViewContainer"] .main .block-container { padding:1.2rem 1rem 3rem; } .ur3-title {font-size:1.55rem;} .ur3-hero {padding:21px;} }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="ur3-hero"><div class="ur3-eyebrow">{eyebrow}</div>'
        f'<div class="ur3-title">{title}</div><p class="ur3-subtitle">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section_title(title):
    st.markdown(f'<div class="ur3-section">{title}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data():
    raw = load_raw()
    prepared = prepare_data(raw)
    return raw, prepared, complete_rows(prepared)


@st.cache_data(show_spinner="두 분류 모델을 학습하고 평가하고 있습니다.")
def get_evaluation(data):
    return evaluate(data)


@st.cache_data(show_spinner=False)
def get_outlier_data(data):
    return outlier_processing(data)


def chart(figure, key=None):
    figure.update_layout(
        font={"family": "DM Sans, Noto Sans KR, sans-serif", "size": 12, "color": "#526174"},
        margin={"l": 24, "r": 24, "t": 62, "b": 28},
        legend={"orientation": "h", "y": -0.22, "title_text": "", "font": {"size": 11}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        title={"font": {"size": 16, "color": "#26364A"}, "x": 0.02, "xanchor": "left"},
        xaxis={"gridcolor": "#EEF1F5", "zerolinecolor": "#DCE3EC", "linecolor": "#E4EAF1"},
        yaxis={"gridcolor": "#EEF1F5", "zerolinecolor": "#DCE3EC", "linecolor": "#E4EAF1"},
    )
    st.plotly_chart(figure, width="stretch", key=key, config={"displaylogo": False, "scrollZoom": True})


def csv_bytes(table):
    return table.to_csv(index=False).encode("utf-8-sig")


def decision_tree_figure(model, feature_names, max_nodes=63):
    """학습된 sklearn 결정트리를 Plotly 노드·연결선으로 시각화한다."""
    tree = model.tree_
    positions = {}
    leaves = []

    def collect(node):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:
            leaves.append(node)
            return
        collect(left)
        collect(right)

    collect(0)
    leaf_x = {node: i for i, node in enumerate(leaves)}

    def place(node, depth=0):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:
            x = leaf_x[node]
        else:
            x = (place(left, depth + 1) + place(right, depth + 1)) / 2
        positions[node] = (x, -depth)
        return x

    place(0)
    visible = set(positions)
    if len(visible) > max_nodes:
        # 깊이 우선이 아닌 상위 레벨부터 최대 노드 수를 선택한다.
        visible = set(sorted(positions, key=lambda n: (positions[n][1], n))[:max_nodes])

    edge_x, edge_y = [], []
    for node in visible:
        for child in (tree.children_left[node], tree.children_right[node]):
            if child in visible and child != node:
                x0, y0 = positions[node]
                x1, y1 = positions[child]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

    node_x, node_y, texts, colors, hover = [], [], [], [], []
    for node in visible:
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        left, right = tree.children_left[node], tree.children_right[node]
        is_leaf = left == right
        if is_leaf:
            predicted = bool(np.argmax(tree.value[node][0]))
            label = "파지 손실" if predicted else "정상"
            text = f"{label}<br>표본 {int(tree.n_node_samples[node])}개"
            colors.append("#F08A5D" if predicted else "#2A9D8F")
            hover.append(f"리프 노드<br>예측: {label}<br>표본: {int(tree.n_node_samples[node])}개")
        else:
            feature = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            text = f"{feature}<br>≤ {threshold:.3g}?"
            colors.append("#EAF0F8")
            hover.append(f"분할 질문: {feature} ≤ {threshold:.6g}<br>표본: {int(tree.n_node_samples[node])}개")
        texts.append(text)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"color": "#BCC8D6", "width": 1.5}, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=texts, textposition="middle center",
        textfont={"size": 9, "color": "#26364A"}, marker={"size": 38, "color": colors, "line": {"color": "#FFFFFF", "width": 2}},
        hovertext=hover, hoverinfo="text", showlegend=False
    ))
    fig.update_layout(height=max(430, (max(abs(y) for y in node_y) + 1) * 115), xaxis={"visible": False}, yaxis={"visible": False}, showlegend=False)
    return fig
