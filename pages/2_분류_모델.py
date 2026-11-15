import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.tree import export_text
from analysis import FEATURES, LR, TREE, TARGET, LABELS, split_cycles, split_table
from ui import chart, csv_bytes, get_data, get_evaluation, page_header, section_title, decision_tree_figure

page_header(
    "MODEL TRAINING & EVALUATION",
    "분류 모델과 평가",
    "동일한 시간순 학습·평가 자료로 로지스틱 회귀와 의사결정트리를 학습합니다. "
    "성능 지표뿐 아니라 실제 오분류와 모델이 학습한 판단 근거까지 확인합니다."
)
_, _, data = get_data()
train, test = split_cycles(data)

st.write(
    "이 페이지는 설명만 보여주는 것이 아니라, 제공된 CSV의 19개 센서값으로 "
    "두 분류 모델을 실제 학습한 뒤 평가 구간의 각 행을 정상 또는 파지 손실로 예측합니다."
)
st.info(
    "정답 기준: grip_lost=False(0)는 정상, grip_lost=True(1)는 파지 손실입니다. "
    "파지 손실을 양성(Positive)으로 계산합니다."
)

with st.expander("두 모델의 학습 방식과 설정", expanded=True):
    st.markdown(f"**{LR}**")
    st.write(
        "19개 센서값을 표준화한 뒤 각 센서에 학습된 가중치를 적용해 파지 손실일 확률을 계산합니다. "
        "확률이 기본 기준 0.5 이상이면 파지 손실, 미만이면 정상으로 분류합니다."
    )
    st.caption("QuantileClipper → StandardScaler → LogisticRegression · C=1.0 · solver=lbfgs · max_iter=2000 · class_weight=balanced")
    st.markdown(f"**{TREE}**")
    st.write(
        "센서값에 대해 '이 값이 특정 기준보다 큰가?'와 같은 질문을 차례로 적용해 데이터를 나눕니다. "
        "마지막 그룹에 속한 학습 자료의 다수 상태를 바탕으로 새 행의 상태를 예측합니다."
    )
    st.caption("QuantileClipper → DecisionTreeClassifier · max_depth=5 · min_samples_leaf=10 · random_state=42 · class_weight=balanced")
    st.write(
        "두 모델은 동일한 학습·평가 행과 동일한 19개 입력 센서를 사용합니다. "
        "결측 행은 제외하고, 작업 주기의 시작 시각 순서로 앞 80% 주기를 학습에, 뒤 20% 주기를 평가에 사용합니다. "
        "분위수 경계와 표준화 기준은 학습 자료에서만 계산합니다."
    )

section_title("학습·평가 자료 분할")
st.dataframe(split_table(train, test), hide_index=True, use_container_width=True)
st.caption("같은 작업 주기의 행은 한쪽 집합에만 들어갑니다. Num, Timestamp, cycle, Robot_ProtectiveStop은 모델 입력에서 제외합니다.")

scores, matrices, predictions, fitted_models = get_evaluation(data)

section_title("실제 평가 결과")
st.dataframe(scores.style.format("{:.4f}"), use_container_width=True)
st.download_button("평가 지표 CSV 다운로드", csv_bytes(scores.reset_index()), "model_metrics.csv", "text/csv")
st.caption(
    f"평가 데이터 {len(test):,}행 중 실제 정상 {(~test[TARGET]).sum():,}행, "
    f"실제 파지 손실 {test[TARGET].sum():,}행입니다. 손실 비율이 낮으므로 정확도만으로 모델을 판단하지 않습니다."
)

metric_help = {
    "정확도": "전체 평가 행 중 맞힌 비율",
    "정밀도": "손실이라고 예측한 행 중 실제 손실의 비율",
    "재현율": "실제 손실 행 중 모델이 찾아낸 비율",
    "F1": "정밀도와 재현율을 함께 고려한 점수",
}
cols = st.columns(4)
for col, (label, description), metric_col in zip(
    cols,
    metric_help.items(),
    ["정확도(Accuracy)", "정밀도(Precision)", "재현율(Recall)", "F1 점수"]
):
    col.metric(label, f"{scores[metric_col].max():.3f}", help=f"두 모델 중 높은 값. {description}")

section_title("혼동 행렬 · 실제 상태와 모델 예측")
for name, matrix in matrices.items():
    st.markdown(f"**{name}**")
    left, right = st.columns([1.2, 1])
    with left:
        fig = px.imshow(
            matrix, x=["예측 정상", "예측 파지 손실"],
            y=["실제 정상", "실제 파지 손실"], text_auto=True,
            color_continuous_scale="Teal",
            labels={"x": "모델 예측", "y": "실제 정답", "color": "행 수"},
            title="평가 데이터의 실제값과 예측값"
        )
        fig.update_layout(height=340)
        chart(fig, f"cm_{name}")
    with right:
        st.write(pd.DataFrame(
            matrix,
            index=["실제 정상", "실제 파지 손실"],
            columns=["예측 정상", "예측 파지 손실"]
        ))
        st.write(f"- 실제 손실을 찾아냄(TP): **{matrix[1,1]:,}행**")
        st.write(f"- 손실을 놓침(FN): **{matrix[1,0]:,}행**")
        st.write(f"- 정상에 오경보(FP): **{matrix[0,1]:,}행**")
        st.write(f"- 정상을 맞힘(TN): **{matrix[0,0]:,}행**")

section_title("평가 데이터별 분류 결과")
st.write(
    "아래 그래프는 각 모델이 평가 데이터의 행을 정상/파지 손실로 실제 예측한 개수입니다. "
    "모델이 출력한 예측 결과를 실제 정답과 나란히 비교합니다."
)
comparison_rows = []
for model_name in [LR, TREE]:
    comparison_rows.extend([
        {"모델": model_name, "구분": "실제 정답", "상태": "정상", "행 수": int((test[TARGET] == False).sum())},
        {"모델": model_name, "구분": "실제 정답", "상태": "파지 손실", "행 수": int((test[TARGET] == True).sum())},
        {"모델": model_name, "구분": "모델 예측", "상태": "정상", "행 수": int((predictions[model_name] == False).sum())},
        {"모델": model_name, "구분": "모델 예측", "상태": "파지 손실", "행 수": int((predictions[model_name] == True).sum())},
    ])
compare_df = pd.DataFrame(comparison_rows)
fig = px.bar(
    compare_df, x="구분", y="행 수", color="상태", barmode="group",
    facet_col="모델", text_auto=True,
    color_discrete_map={"정상": "#157F83", "파지 손실": "#DD6B3E"},
    title="모델별 실제 정답 개수와 예측 개수",
    category_orders={"상태": ["정상", "파지 손실"], "구분": ["실제 정답", "모델 예측"]}
)
fig.update_layout(height=430)
chart(fig, "actual_vs_predicted_counts")

with st.expander("평가 데이터의 행별 분류 결과 보기", expanded=True):
    display = predictions.copy()
    display["실제 상태"] = display[TARGET].map(LABELS)
    for model_name in [LR, TREE]:
        display[f"{model_name} 예측 상태"] = display[model_name].map(LABELS)
        display[f"{model_name} 정답 여부"] = display[model_name] == display[TARGET]
    keep = ["Num", "Timestamp", "cycle", "실제 상태",
            f"{LR} 예측 상태", f"{LR} 손실 확률", f"{LR} 정답 여부",
            f"{TREE} 예측 상태", f"{TREE} 손실 확률", f"{TREE} 정답 여부"]
    st.dataframe(display[keep], hide_index=True, use_container_width=True, height=380)
    st.download_button("평가 데이터 전체 분류 결과 다운로드", csv_bytes(display[keep]), "classification_predictions.csv", "text/csv")

section_title("모델의 판단 근거")
with st.expander("로지스틱 회귀: 센서별 학습 가중치"):
    lr_model = fitted_models[LR].named_steps["logisticregression"]
    coef = pd.DataFrame({
        "센서": FEATURES,
        "학습 가중치": lr_model.coef_[0],
        "절댓값": abs(lr_model.coef_[0])
    }).sort_values("절댓값", ascending=False)
    st.write("가중치의 부호는 다른 센서값이 같을 때 손실 확률을 높이는 방향/낮추는 방향을 뜻합니다. 표준화된 입력 기준의 가중치입니다.")
    st.dataframe(coef.drop(columns="절댓값").style.format({"학습 가중치": "{:.4f}"}), hide_index=True, use_container_width=True)
    fig = px.bar(coef.head(12).sort_values("학습 가중치"), x="학습 가중치", y="센서", orientation="h",
                 title="절댓값이 큰 로지스틱 회귀 가중치 상위 12개")
    chart(fig, "lr_coefficients")

with st.expander("의사결정트리: 실제 학습된 질문 규칙"):
    tree_model = fitted_models[TREE].named_steps["decisiontreeclassifier"]
    st.markdown("**학습된 의사결정트리 구조**")
    st.write("각 내부 노드는 센서값에 대한 질문이고, 말단 노드는 최종 예측 상태입니다.")
    tree_fig = decision_tree_figure(tree_model, FEATURES)
    chart(tree_fig, "decision_tree_structure")
    rules = export_text(tree_model, feature_names=FEATURES, decimals=3)
    with st.expander("의사결정트리 규칙을 텍스트로 보기"):
        st.code(rules, language="text")
    importance = pd.DataFrame({"센서": FEATURES, "중요도": tree_model.feature_importances_}).sort_values("중요도", ascending=False)
    st.write("중요도는 학습된 트리가 분할에 사용한 센서의 상대적 기여를 나타냅니다. 0이면 이 트리의 분할에 사용되지 않았습니다.")
    st.dataframe(importance, hide_index=True, use_container_width=True)
    fig = px.bar(importance.head(12).sort_values("중요도"), x="중요도", y="센서", orientation="h",
                 title="의사결정트리 센서 중요도 상위 12개")
    chart(fig, "tree_importance")

with st.expander("평가 지표의 의미"):
    st.markdown("""
- **정확도(Accuracy)**: 전체 평가 행 중 실제 상태를 맞힌 비율
- **정밀도(Precision)**: 파지 손실이라고 예측한 행 중 실제 손실인 비율
- **재현율(Recall)**: 실제 파지 손실 행 중 모델이 찾아낸 비율
- **F1 점수**: 정밀도와 재현율을 함께 고려하는 조화평균
- **혼동 행렬**: 실제 상태와 예측 상태를 교차해 TP·TN·FP·FN을 세는 표
""")

st.warning(
    "이번 모델은 기록된 시점의 센서값으로 같은 시점의 grip_lost 상태를 분류합니다. "
    "미래의 손실을 사전에 예측하거나 실제 로봇 제어에 바로 적용하는 성능을 검증한 것은 아닙니다. "
    "평가 데이터에서 손실 행이 적고 오경보가 많을 수 있으므로 정밀도·재현율과 혼동 행렬을 함께 해석해야 합니다."
)
