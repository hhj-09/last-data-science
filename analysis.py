"""첨부된 UR3 CSV의 전처리, 시간순 분할, 모델 평가를 담당한다."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

TITLE = "UR3 그리퍼 파지 손실 분석"
PROJECT = "UR3 협동로봇팔의 센서 데이터를 활용한 그리퍼 파지 손실 분류 모델 개발"
QUESTION = "로봇팔의 관절별 전류, 온도, 회전 속도 및 그리퍼 전류 데이터를 활용하여 그리퍼의 파지 손실 발생 여부를 분류할 수 있을까?"
QUESTIONS = [
    "결측값과 극단값 후보는 어느 센서에 얼마나 있으며, 전처리로 어떤 값이 달라지는가?",
    "결측 행 제외 전후에 정상과 파지 손실의 개수와 비율은 어떻게 달라지는가?",
    "정상과 손실로 기록된 행에서 그리퍼 전류의 분포와 중앙값은 어떻게 다른가?",
    "각 관절의 전류와 회전 속도 사이의 선형 관계는 관절마다 어떻게 다른가?",
    "같은 평가 구간에서 두 모델은 실제 손실을 몇 행 찾아내며 정상 행을 몇 번 손실로 분류하는가?",
]
SOURCE = "https://archive.ics.uci.edu/dataset/963/ur3+cobotops"
PAPER = "https://doi.org/10.1007/978-3-031-63851-0_6"
ORIGINAL_FILENAME = "dataset_02052023.xlsx - data.csv"
DATA_PATH = Path(__file__).parent / "data" / "ur3_cobotops.csv"
LOWER_QUANTILE, UPPER_QUANTILE = 0.005, 0.995
FEATURES = [f"{kind}_J{i}" for kind in ("Current", "Temperature", "Speed") for i in range(6)] + ["Tool_current"]
TARGET = "grip_lost"
LR = "로지스틱 회귀(확률로 답하는 모델)"
TREE = "의사결정트리(질문으로 답하는 모델)"
LABELS = {False: "정상", True: "파지 손실"}


def load_raw():
    # 업로드한 CSV를 그대로 복사했다. 실행 중 외부 자료를 내려받지 않는다.
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


def prepare_data(raw):
    """이름과 자료형을 정리한다. 학습 통계를 사용하지 않는 처리다."""
    data = raw.copy()
    data.columns = data.columns.str.strip()
    data = data.rename(columns={"Temperature_T0": "Temperature_J0"})
    if data.columns.duplicated().any():
        raise ValueError("컬럼명 정리 후 중복 이름이 생겼습니다.")
    required = FEATURES + [TARGET, "Timestamp", "cycle"]
    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing_columns)}")
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="raise")
    if np.isinf(data[FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("센서에 무한대 값이 있습니다. 원본 파일을 확인하세요.")
    labels = data[TARGET].astype("string").str.strip().str.lower()
    data[TARGET] = labels.map({"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})
    if (data[TARGET].isna() & raw[TARGET].notna()).any():
        raise ValueError("grip_lost에는 True/False 또는 1/0만 사용할 수 있습니다.")
    data["Timestamp"] = pd.to_datetime(data["Timestamp"].astype("string").str.strip().str.strip('"'), utc=True, errors="raise", format="mixed")
    data["cycle"] = pd.to_numeric(data["cycle"], errors="raise")
    if data[["Timestamp", "cycle"]].isna().any().any():
        raise ValueError("시간 또는 작업 주기가 비어 있어 시간순 분할이 어렵습니다.")
    return data


def complete_rows(data):
    clean = data.dropna(subset=FEATURES + [TARGET]).copy()
    clean[TARGET] = clean[TARGET].astype(bool)
    return clean.sort_values("Timestamp", kind="stable")


class QuantileClipper(TransformerMixin, BaseEstimator):
    """fit 자료의 분위수 경계를 저장하고 transform 자료에 그대로 적용한다.

    통계적 극단값의 영향을 제한하는 처리이며 측정 오류 판정은 아니다.
    Pipeline 안에서 학습하므로 평가 자료로 경계를 다시 계산하지 않는다.
    """

    def __init__(self, lower=LOWER_QUANTILE, upper=UPPER_QUANTILE):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("분위수 설정은 0 이상 1 이하이며 하한이 상한보다 작아야 합니다.")
        if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
            raise ValueError("극단값 처리 전에 결측값과 무한대 값을 확인하세요.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.lower_bounds_ = X.quantile(self.lower, interpolation="linear")
        self.upper_bounds_ = X.quantile(self.upper, interpolation="linear")
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("학습 시 입력 센서와 적용 시 입력 센서가 다릅니다.")
        return X.clip(self.lower_bounds_, self.upper_bounds_, axis="columns")


def split_cycles(data):
    """한 주기의 모든 행을 같은 집합에 배치하고 시간 경계를 검증한다."""
    starts = data.groupby("cycle").Timestamp.min().sort_values(kind="stable")
    boundary = int(len(starts) * 0.8)
    if boundary == 0 or boundary == len(starts):
        raise ValueError("학습과 평가에 나눌 작업 주기가 부족합니다.")
    training_cycles = starts.index[:boundary]
    in_training = data.cycle.isin(training_cycles)
    train, test = data.loc[in_training].copy(), data.loc[~in_training].copy()
    if train.Timestamp.max() >= test.Timestamp.min():
        raise ValueError("분할 경계에서 작업 주기의 시간 범위가 겹칩니다. 주기와 시간을 확인하세요.")
    if train[TARGET].nunique() < 2 or test[TARGET].nunique() < 2:
        raise ValueError("학습과 평가 자료에 정상과 손실이 모두 필요합니다.")
    return train, test


def class_table(data):
    counts = data[TARGET].value_counts().reindex([False, True], fill_value=0)
    denominator = len(data) if len(data) else 1
    return pd.DataFrame({"상태": ["정상", "파지 손실"], "행 수": counts.to_numpy(), "비율(%)": counts.to_numpy() / denominator * 100})


def outlier_processing(data):
    """학습 자료의 기준으로 전체 관측을 변환하고 변경 내역을 집계한다."""
    train, test = split_cycles(data)
    clipper = QuantileClipper().fit(train[FEATURES])
    processed = data.copy()
    processed[FEATURES] = clipper.transform(data[FEATURES])
    changed = data[FEATURES].ne(processed[FEATURES])
    bounds = pd.DataFrame({
        "센서": FEATURES,
        "학습 0.5% 경계": clipper.lower_bounds_.to_numpy(),
        "학습 99.5% 경계": clipper.upper_bounds_.to_numpy(),
        "학습 변경 값 수": changed.loc[train.index].sum().to_numpy(),
        "평가 변경 값 수": changed.loc[test.index].sum().to_numpy(),
        "전체 변경 비율(%)": changed.mean().to_numpy() * 100,
    })
    audit = []
    for name, part in [("학습", train), ("평가", test), ("전체", data)]:
        mask = changed.loc[part.index]
        changed_rows = mask.any(axis=1)
        audit.append({
            "구간": name, "전체 행": len(part), "값 변경 행": int(changed_rows.sum()),
            "변경 센서값 수": int(mask.sum().sum()),
            "변경된 정상 행": int((changed_rows & ~part[TARGET]).sum()),
            "변경된 손실 행": int((changed_rows & part[TARGET]).sum()),
            "삭제 행": 0,
        })
    return processed, bounds, pd.DataFrame(audit), changed


def split_table(train, test):
    rows = []
    for label, part in [("학습", train), ("평가", test)]:
        rows.append({"구간": label, "작업 주기 수": part.cycle.nunique(), "전체 행": len(part),
                     "정상 행": int((~part[TARGET]).sum()), "손실 행": int(part[TARGET].sum()),
                     "시작(UTC)": part.Timestamp.min().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                     "끝(UTC)": part.Timestamp.max().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]})
    return pd.DataFrame(rows)


def column_dictionary(raw):
    rows = []
    for column in raw.columns:
        name = column.strip().replace("Temperature_T0", "Temperature_J0")
        role, unit, kind = "보조", "해당 없음", "숫자"
        if name.startswith("Current_J"):
            description, unit, role = f"J{name[-1]} 관절 전류", "A", "독립변수"
        elif name.startswith("Temperature_J"):
            description, unit, role = f"J{name[-1]} 관절 온도", "°C", "독립변수"
        elif name.startswith("Speed_J"):
            description, unit, role = f"J{name[-1]} 관절 회전 속도", "°/s (소개 논문 표기)", "독립변수"
        elif name == "Tool_current":
            description, unit, role = "그리퍼 전류", "A", "독립변수"
        elif name == TARGET:
            description, role, kind = "False 정상, True 파지 손실", "종속변수", "이진"
        elif name == "Robot_ProtectiveStop":
            description, kind, role = "0 보호 정지 없음, 1 보호 정지 발생", "이진", "모델 입력 제외"
        elif name == "Timestamp":
            description, kind, role, unit = "측정 시각", "시각", "시간순 정렬", "UTC"
        elif name == "cycle":
            description, role = "작업 주기 번호", "학습·평가 구분"
        else:
            description, role = "관측 행의 식별 번호", "모델 입력 제외"
        rows.append({"원본 컬럼": column, "분석 컬럼": name, "설명": description, "단위": unit, "자료형": kind, "역할": role})
    return pd.DataFrame(rows)


def evaluate(data):
    train, test = split_cycles(data)
    models = {
        LR: make_pipeline(QuantileClipper(), StandardScaler(), LogisticRegression(class_weight="balanced", C=1.0, solver="lbfgs", max_iter=2000)),
        TREE: make_pipeline(QuantileClipper(), DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=42)),
    }
    scores, matrices, predictions, fitted_models = {}, {}, test[["Num", "Timestamp", "cycle", TARGET]].copy(), {}
    for name, model in models.items():
        model.fit(train[FEATURES], train[TARGET])
        fitted_models[name] = model
        predicted = model.predict(test[FEATURES])
        probabilities = model.predict_proba(test[FEATURES])[:, 1]
        scores[name] = {
            "정확도(Accuracy)": accuracy_score(test[TARGET], predicted),
            "정밀도(Precision)": precision_score(test[TARGET], predicted, zero_division=0),
            "재현율(Recall)": recall_score(test[TARGET], predicted, zero_division=0),
            "F1 점수": f1_score(test[TARGET], predicted, zero_division=0),
        }
        matrices[name] = confusion_matrix(test[TARGET], predicted, labels=[False, True])
        predictions[name] = predicted
        predictions[f"{name} 손실 확률"] = probabilities
        predictions[f"{name} 정답 여부"] = (predicted == test[TARGET])
    return pd.DataFrame.from_dict(scores, orient="index").rename_axis("모델"), matrices, predictions, fitted_models
