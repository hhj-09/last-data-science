"""센서 변수 관계와 실제 모델 예측을 탐색하는 인터랙티브 시각화 페이지."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import FEATURES, LABELS, TARGET, LR, TREE, split_cycles
from ui import COLORS, chart, csv_bytes, get_data, get_evaluation, get_outlier_data, page_header, section_title


# --------------------------------------------------
# 공통 데이터와 페이지 헤더
# --------------------------------------------------
page_header(
    "EXPLORATORY VISUAL ANALYTICS",
    "변수 관계 분석",
    "센서 두 개의 조합, 상태별 분포, 센서 간 상관관계, 시간 변화와 실제 모델 예측 확률을 탐색합니다. "
    "변수를 직접 바꾸며 정상과 파지 손실이 어떻게 나타나는지 비교해 보세요."
)

raw, prepared, complete = get_data()
processed, bounds, audit, changed = get_outlier_data(complete)

with st.container(border=True):
    st.markdown("**분석 데이터 설정**")
    stage = st.radio(
        "그래프에 사용할 센서 데이터",
        ["극단값 처리 전", "극단값 처리 후"],
        horizontal=True,
        index=1,
        help="처리 후는 학습 구간에서 계산한 0.5%·99.5% 분위수 경계를 적용한 값입니다."
    )
    data = complete if stage == "극단값 처리 전" else processed
    st.caption("모델 예측 확률은 기존 모델 파이프라인을 그대로 사용하므로 극단값 처리 전후 선택과 무관하게 학습된 모델의 평가 결과를 표시합니다.")

plot_data = data.copy()
plot_data["상태"] = plot_data[TARGET].map(LABELS)

m1, m2, m3, m4 = st.columns(4)
m1.metric("표시 기준 관측", f"{len(plot_data):,}")
m2.metric("센서 변수", "19개")
m3.metric("정상 관측", f"{(~plot_data[TARGET]).sum():,}")
m4.metric("파지 손실 관측", f"{plot_data[TARGET].sum():,}")


def sample_for_plot(frame, max_points=5000):
    """그래프 표시용 균형 표본. 원본 분석 및 모델 학습 자료는 변경하지 않는다."""
    if len(frame) <= max_points:
        return frame
    parts = []
    counts = frame[TARGET].value_counts()
    for label, count in counts.items():
        n = max(1, round(max_points * count / len(frame)))
        n = min(int(count), n)
        parts.append(frame.loc[frame[TARGET] == label].sample(n=n, random_state=42))
    return pd.concat(parts).sort_index()


section_title("탐색 도구")
tab_scatter, tab_distribution, tab_corr, tab_time, tab_probability, tab_errors = st.tabs([
    "두 변수 산점도", "분포 비교", "상관관계", "시간 변화", "모델 예측 확률", "오분류 분석"
])


# --------------------------------------------------
# 1. 산점도
# --------------------------------------------------
with tab_scatter:
    st.markdown("#### 센서 조합에 따른 상태 분포")
    st.write("각 점은 하나의 관측 행입니다. X축과 Y축 센서를 바꾸어 두 상태가 어떤 영역에 분포하는지 살펴보세요.")
    c1, c2 = st.columns(2)
    with c1:
        x_sensor = st.selectbox("X축 센서", FEATURES, index=FEATURES.index("Current_J0"), key="scatter_x")
    with c2:
        y_default = "Tool_current" if x_sensor != "Tool_current" else "Current_J0"
        y_sensor = st.selectbox("Y축 센서", FEATURES, index=FEATURES.index(y_default), key="scatter_y")
    opacity = st.slider("점 투명도", 0.2, 1.0, 0.68, 0.05, key="scatter_opacity")
    if x_sensor == y_sensor:
        st.warning("서로 다른 센서 두 개를 선택해 주세요.")
    else:
        scatter_data = sample_for_plot(plot_data)
        fig = px.scatter(
            scatter_data, x=x_sensor, y=y_sensor, color="상태", color_discrete_map=COLORS,
            opacity=opacity, custom_data=["Num", "Timestamp", "cycle"],
            labels={x_sensor: x_sensor, y_sensor: y_sensor},
            title=f"{x_sensor} × {y_sensor} · {stage}"
        )
        fig.update_traces(marker={"size": 7}, hovertemplate=(
            "상태: %{fullData.name}<br>" + x_sensor + ": %{x:.5g}<br>" + y_sensor + ": %{y:.5g}<br>"
            "관측 번호: %{customdata[0]}<br>시각: %{customdata[1]}<br>작업 주기: %{customdata[2]}<extra></extra>"
        ))
        fig.update_layout(height=570)
        chart(fig, "sensor_scatter")
        corr_value = plot_data[[x_sensor, y_sensor]].corr().iloc[0, 1]
        st.metric("선택한 두 변수의 피어슨 상관계수", f"{corr_value:.4f}")
        st.caption(f"그래프 표시 점은 {len(scatter_data):,}개입니다. 데이터가 많으면 상태별 비율을 고려해 최대 5,000개를 표시합니다. 상관관계는 인과관계를 뜻하지 않습니다.")


# --------------------------------------------------
# 2. 분포 그래프
# --------------------------------------------------
with tab_distribution:
    st.markdown("#### 정상과 파지 손실의 센서값 분포")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        selected_sensor = st.selectbox("분석할 센서", FEATURES, index=FEATURES.index("Tool_current"), key="dist_sensor")
    with c2:
        graph_type = st.selectbox("그래프 유형", ["히스토그램", "상자그림", "바이올린 그래프"], key="dist_type")
    if graph_type == "히스토그램":
        bins = st.slider("구간 수", 10, 70, 35, 5, key="hist_bins")
        fig = px.histogram(
            plot_data, x=selected_sensor, color="상태", color_discrete_map=COLORS,
            nbins=bins, histnorm="probability density", barmode="overlay", opacity=0.58,
            title=f"{selected_sensor} 상태별 분포", labels={selected_sensor: f"{selected_sensor} 값"}
        )
    elif graph_type == "상자그림":
        fig = px.box(
            plot_data, x="상태", y=selected_sensor, color="상태", color_discrete_map=COLORS,
            points="outliers", title=f"{selected_sensor} 정상·파지 손실 상자그림"
        )
    else:
        fig = px.violin(
            plot_data, x="상태", y=selected_sensor, color="상태", color_discrete_map=COLORS,
            box=True, points="outliers", title=f"{selected_sensor} 정상·파지 손실 바이올린 그래프"
        )
    fig.update_layout(height=520)
    chart(fig, "sensor_distribution")

    summary = plot_data.groupby("상태")[selected_sensor].agg(
        행수="count", 평균="mean", 중앙값="median", 표준편차="std", 최솟값="min", 최댓값="max"
    )
    st.markdown("**상태별 기초 통계**")
    st.dataframe(summary.style.format("{:.5g}"), use_container_width=True)
    st.caption("정상과 파지 손실의 관측 수 차이가 큽니다. 분포의 겹침, 중앙값과 산포를 함께 살펴보세요.")


# --------------------------------------------------
# 3. 상관관계 히트맵
# --------------------------------------------------
with tab_corr:
    st.markdown("#### 19개 센서의 상관관계 지도")
    corr_group = st.selectbox("계산할 데이터", ["전체 상태", "정상만", "파지 손실만"], key="corr_group")
    if corr_group == "정상만":
        corr_source = plot_data.loc[~plot_data[TARGET], FEATURES]
    elif corr_group == "파지 손실만":
        corr_source = plot_data.loc[plot_data[TARGET], FEATURES]
    else:
        corr_source = plot_data[FEATURES]
    corr_matrix = corr_source.corr(method="pearson")
    fig = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title=f"피어슨 상관관계 · {corr_group}",
        labels={"x": "센서 변수", "y": "센서 변수", "color": "상관계수"}
    )
    fig.update_layout(height=760)
    chart(fig, "sensor_correlation_heatmap")
    st.caption(f"계산에 사용한 관측: {len(corr_source):,}행. +1은 양의 선형 관계, -1은 음의 선형 관계, 0은 선형 관계가 약함을 나타냅니다.")

    selected_corr_sensor = st.selectbox("선택 센서와 관련성이 높은 변수", FEATURES, key="related_sensor")
    related = corr_matrix[selected_corr_sensor].drop(selected_corr_sensor).rename("상관계수").to_frame()
    related["절댓값"] = related["상관계수"].abs()
    related = related.sort_values("절댓값", ascending=True).tail(10).reset_index(names="비교 센서")
    fig = px.bar(
        related, x="상관계수", y="비교 센서", orientation="h", color="상관계수",
        color_continuous_scale="RdBu_r", range_color=[-1, 1],
        title=f"{selected_corr_sensor}와 상관관계가 큰 센서 TOP 10"
    )
    fig.update_xaxes(range=[-1, 1])
    fig.update_layout(height=450)
    chart(fig, "selected_sensor_correlations")


# --------------------------------------------------
# 4. 시간 변화
# --------------------------------------------------
with tab_time:
    st.markdown("#### 시간에 따른 센서값 변화")
    time_sensor = st.selectbox("시간 흐름을 볼 센서", FEATURES, index=FEATURES.index("Tool_current"), key="time_sensor")
    time_data = sample_for_plot(plot_data.sort_values("Timestamp"), max_points=5000)
    fig = px.scatter(
        time_data, x="Timestamp", y=time_sensor, color="상태", color_discrete_map=COLORS,
        opacity=0.65, custom_data=["Num", "cycle"],
        title=f"시간에 따른 {time_sensor} 변화", labels={time_sensor: f"{time_sensor} 값", "Timestamp": "측정 시각 (UTC)"}
    )
    fig.update_traces(hovertemplate="시각: %{x}<br>센서값: %{y:.5g}<br>상태: %{fullData.name}<br>관측 번호: %{customdata[0]}<br>작업 주기: %{customdata[1]}<extra></extra>")
    fig.update_layout(height=520)
    chart(fig, "sensor_over_time")
    st.caption("점은 개별 관측 시점입니다. 서로 다른 작업 주기를 연속된 선으로 연결하지 않았습니다.")


# --------------------------------------------------
# 5. 모델 예측 확률
# --------------------------------------------------
with tab_probability:
    st.markdown("#### 모델이 계산한 파지 손실 확률")
    st.write("평가 데이터에 대해 학습된 모델이 계산한 확률입니다. 색은 실제 정답이며, 0.5 점선은 기본 분류 기준입니다.")
    c1, c2 = st.columns(2)
    with c1:
        model_name = st.selectbox("분류 모델", [LR, TREE], key="prob_model")
    with c2:
        probability_sensor = st.selectbox("X축 센서", FEATURES, index=FEATURES.index("Tool_current"), key="prob_sensor")

    scores, matrices, predictions, fitted_models = get_evaluation(complete)
    _, test = split_cycles(complete)
    probability_data = test[[probability_sensor, TARGET, "Num", "Timestamp", "cycle"]].copy()
    probability_data["실제 상태"] = probability_data[TARGET].map(LABELS)
    probability_data["파지 손실 예측 확률"] = predictions[f"{model_name} 손실 확률"].reindex(probability_data.index)
    probability_data["예측 상태"] = np.where(probability_data["파지 손실 예측 확률"] >= 0.5, "파지 손실", "정상")
    probability_data["분류 결과"] = np.where(probability_data[TARGET] == (probability_data["파지 손실 예측 확률"] >= 0.5), "정답", "오분류")

    prob_sample = sample_for_plot(probability_data, 5000)
    fig = px.scatter(
        prob_sample, x=probability_sensor, y="파지 손실 예측 확률", color="실제 상태",
        symbol="분류 결과", color_discrete_map=COLORS, opacity=0.72,
        custom_data=["Num", "Timestamp", "cycle", "예측 상태"],
        title=f"{model_name} · {probability_sensor}와 예측 확률",
        labels={probability_sensor: f"{probability_sensor} 값", "파지 손실 예측 확률": "파지 손실 확률"}
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#526174", annotation_text="분류 기준 0.5")
    fig.update_yaxes(range=[-0.03, 1.03], tickformat=".0%")
    fig.update_traces(hovertemplate="센서값: %{x:.5g}<br>손실 확률: %{y:.1%}<br>실제 상태: %{fullData.name}<br>예측 상태: %{customdata[3]}<br>관측 번호: %{customdata[0]}<br>시각: %{customdata[1]}<br>작업 주기: %{customdata[2]}<extra></extra>")
    fig.update_layout(height=540)
    chart(fig, "sensor_vs_predicted_probability")
    st.dataframe(scores.loc[[model_name]].style.format("{:.4f}"), use_container_width=True)
    st.caption("중요: 모델은 19개 센서를 모두 입력으로 사용합니다. X축은 그중 선택한 센서 하나를 보여주는 시각화일 뿐, 해당 센서 하나만으로 확률을 계산한 것은 아닙니다.")

    with st.expander("예측 확률이 높은 평가 관측 TOP 100"):
        high_risk = probability_data.sort_values("파지 손실 예측 확률", ascending=False).head(100)
        st.dataframe(high_risk[["Num", "Timestamp", "cycle", probability_sensor, "실제 상태", "예측 상태", "파지 손실 예측 확률", "분류 결과"]], hide_index=True, use_container_width=True)
        st.download_button("예측 확률 결과 CSV 다운로드", csv_bytes(high_risk), "sensor_probability_results.csv", "text/csv")


# --------------------------------------------------
# 6. 오분류 분석
# --------------------------------------------------
with tab_errors:
    st.markdown("#### 모델이 틀린 관측은 어디에 모여 있을까?")
    scores, matrices, predictions, fitted_models = get_evaluation(complete)
    model_error = st.selectbox("오분류를 확인할 모델", [LR, TREE], key="error_model")
    _, test = split_cycles(complete)
    err = test[["Num", "Timestamp", "cycle", TARGET] + FEATURES].copy()
    err["실제 상태"] = err[TARGET].map(LABELS)
    err["예측 상태"] = predictions[model_error].map(LABELS)
    err["손실 예측 확률"] = predictions[f"{model_error} 손실 확률"]
    err["분류 결과"] = np.where(predictions[f"{model_error} 정답 여부"], "정답", "오분류")
    errors = err.loc[err["분류 결과"] == "오분류"].copy()
    tp = int(matrices[model_error][1, 1])
    fn = int(matrices[model_error][1, 0])
    fp = int(matrices[model_error][0, 1])
    tn = int(matrices[model_error][0, 0])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("정답 분류", f"{tn + tp:,}")
    k2.metric("전체 오분류", f"{len(errors):,}")
    k3.metric("손실 놓침 (FN)", f"{fn:,}")
    k4.metric("정상 오경보 (FP)", f"{fp:,}")

    c1, c2 = st.columns(2)
    with c1:
        x_err = st.selectbox("오분류 산점도 X축", FEATURES, index=FEATURES.index("Tool_current"), key="err_x")
    with c2:
        y_err = st.selectbox("오분류 산점도 Y축", FEATURES, index=FEATURES.index("Current_J0"), key="err_y")
    if x_err == y_err:
        st.warning("서로 다른 센서를 선택해 주세요.")
    else:
        view = sample_for_plot(err, 5000)
        fig = px.scatter(
            view, x=x_err, y=y_err, color="분류 결과", symbol="실제 상태",
            color_discrete_map={"정답": "#2A9D8F", "오분류": "#E76F51"},
            custom_data=["Num", "실제 상태", "예측 상태", "손실 예측 확률", "cycle"],
            title="정답과 오분류 관측의 센서 공간 분포"
        )
        fig.update_traces(hovertemplate="X: %{x:.5g}<br>Y: %{y:.5g}<br>결과: %{fullData.name}<br>실제: %{customdata[1]}<br>예측: %{customdata[2]}<br>손실 확률: %{customdata[3]:.1%}<br>관측 번호: %{customdata[0]}<br>작업 주기: %{customdata[4]}<extra></extra>")
        fig.update_layout(height=520)
        chart(fig, "misclassification_scatter")
    st.markdown("**오분류 관측 목록**")
    st.dataframe(errors[["Num", "Timestamp", "cycle", "실제 상태", "예측 상태", "손실 예측 확률"]], hide_index=True, use_container_width=True, height=330)
    st.download_button("오분류 관측 CSV 다운로드", csv_bytes(errors), "misclassified_observations.csv", "text/csv")


st.divider()
st.markdown("**해석할 때 주의할 점**")
st.markdown(
    "- 센서값의 차이나 상관관계만으로 파지 손실의 원인을 단정할 수 없습니다.\n"
    "- 파지 손실 행은 정상 행보다 적으므로 점의 밀도와 분포를 함께 확인해야 합니다.\n"
    "- 예측 확률은 19개 센서를 모두 사용한 모델의 결과이며, 선택한 X축 센서 하나의 단독 효과가 아닙니다.\n"
    "- 이 프로젝트는 기록된 시점의 상태를 분류합니다. 미래 손실 예측이나 실제 로봇 제어 성능을 검증한 것은 아닙니다."
)
