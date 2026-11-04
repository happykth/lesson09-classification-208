import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="채점표 · 뇌졸중 예측 실습실", page_icon="📝", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/happykth/data208/main/stroke.csv"

FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

LOGI = "로지스틱 회귀(확률로 답하는 모델)"
TREE = "의사결정트리(질문으로 답하는 모델)"
BASE = "훈련용에서 많은 쪽으로만 답하는 모델"

RANDOM_STATE = 42
MAX_ROWS = 20
CANNOT = "계산할 수 없음"


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

st.title("📝 채점표")
st.caption("'분류 모델' 페이지와 같은 방법으로 읽고, 나누고, 학습한 두 모델을 테스트 데이터로 채점해요. 뇌졸중이 양성이에요.")

# ── 입력 속성 고르기 ('분류 모델' 페이지와 같다) ──────────────────────────
picked = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda f: KOR[f],
)
selected = [f for f in FEATURES if f in picked]

if len(selected) < 2:
    st.info("속성을 두 개 이상 골라 주세요.")
    st.stop()

# ── 훈련용·테스트용 나누기 (번호 순 정렬 → 열 명씩 묶어 앞 세 명이 테스트용) ──
df = raw.sort_values("id").reset_index(drop=True)
position = pd.Series(range(len(df)))
is_test = (position % 10) < 3

X_all = df[selected].copy()
y_all = df["stroke"]
X_train, X_test = X_all[~is_test].copy(), X_all[is_test].copy()
y_train, y_test = y_all[~is_test], y_all[is_test]

# 체질량지수는 고른 경우에만, 훈련용 중앙값으로 빈 값을 채운다.
if "bmi" in selected:
    bmi_median = float(X_train["bmi"].median())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

# ── 모델 학습 ('분류 모델' 페이지와 같다) ────────────────────────────────
scaler = StandardScaler().fit(X_train)
logi = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
tree = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
).fit(X_train, y_train)
majority = int(y_train.mode()[0])

pred_logi = pd.Series(logi.predict(scaler.transform(X_test)), index=X_test.index)
pred_tree = pd.Series(tree.predict(X_test), index=X_test.index)
pred_base = pd.Series(majority, index=X_test.index)

total = len(y_test)
n_stroke = int((y_test == 1).sum())

st.write(f"테스트용 **{total:,}명** 가운데 뇌졸중인 사람은 **{n_stroke:,}명**이에요.")


def confusion(pred):
    """뇌졸중(1)을 양성으로 둔 혼동행렬의 네 칸."""
    actual_pos = y_test == 1
    pred_pos = pred == 1
    return {
        "tp": int((actual_pos & pred_pos).sum()),
        "fn": int((actual_pos & ~pred_pos).sum()),
        "fp": int((~actual_pos & pred_pos).sum()),
        "tn": int((~actual_pos & ~pred_pos).sum()),
    }


def confusion_table(c):
    return pd.DataFrame(
        [
            [f"{c['tp']:,}명 (TP)", f"{c['fn']:,}명 (FN)", f"{c['tp'] + c['fn']:,}명"],
            [f"{c['fp']:,}명 (FP)", f"{c['tn']:,}명 (TN)", f"{c['fp'] + c['tn']:,}명"],
            [f"{c['tp'] + c['fp']:,}명", f"{c['fn'] + c['tn']:,}명", f"{total:,}명"],
        ],
        index=["실제: 뇌졸중", "실제: 아님", "합계"],
        columns=["예측: 뇌졸중", "예측: 아님", "합계"],
    )


conf = {LOGI: confusion(pred_logi), TREE: confusion(pred_tree), BASE: confusion(pred_base)}

# ── 1. 혼동행렬 ──────────────────────────────────────────────────
st.subheader("1. 혼동행렬")
st.caption(
    "행은 실제, 열은 예측이에요. "
    "TP: 뇌졸중을 뇌졸중이라고 맞힘 · FN: 뇌졸중인데 아니라고 놓침 · "
    "FP: 아닌데 뇌졸중이라고 헛짚음 · TN: 아닌 사람을 아니라고 맞힘."
)
left, right = st.columns(2)
for col, name in ((left, LOGI), (right, TREE)):
    col.markdown(f"**{name}**")
    col.table(confusion_table(conf[name]))

# ── 2. 지표 비교표 ───────────────────────────────────────────────
st.subheader("2. 정확도·재현율·정밀도·F1")


def metrics(c):
    """네 칸에서 (정확도, 재현율, 정밀도, F1). 계산할 수 없으면 None."""
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else None        # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) > 0 else None     # 분모: 뇌졸중이라 예측한 사람 수
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1


def fmt(v):
    return CANNOT if v is None else f"{v * 100:.2f}%"


rows = []
for name in (LOGI, TREE, BASE):
    acc, rec, prec, f1 = metrics(conf[name])
    rows.append(
        {"모델": name, "정확도": fmt(acc), "재현율": fmt(rec), "정밀도": fmt(prec), "F1": fmt(f1)}
    )
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.caption(
    "정확도 = (TP + TN) ÷ 전체 사람 수 · "
    "재현율 = TP ÷ (TP + FN), 즉 실제 뇌졸중인 사람 수로 나눠요 · "
    "정밀도 = TP ÷ (TP + FP), 즉 뇌졸중이라 예측한 사람 수로 나눠요 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
    "뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1은 계산할 수 없어요."
)

# ── 3. 놓친 환자와 헛짚은 사람 ───────────────────────────────────
st.subheader("3. 놓친 환자와 헛짚은 사람")
choice = st.radio("어느 모델의 목록을 볼까요?", [LOGI, TREE], horizontal=True)
pred = pred_logi if choice == LOGI else pred_tree

missed_index = y_test.index[(y_test == 1) & (pred == 0)]     # 실제 뇌졸중 → 아니라고 예측 (FN)
wrong_index = y_test.index[(y_test == 0) & (pred == 1)]      # 실제 아님 → 뇌졸중이라고 예측 (FP)


def people_list(index):
    """나이 많은 순으로 정렬한 전체 목록(번호·나이·평균 혈당·고혈압 여부·심장병 여부)."""
    p = df.loc[index, ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]]
    p = p.sort_values(["age", "id"], ascending=[False, True])
    return pd.DataFrame(
        {
            "번호": p["id"],
            "나이": p["age"],
            "평균 혈당": p["avg_glucose_level"],
            "고혈압 여부": p["hypertension"].map({1: "있음", 0: "없음"}),
            "심장병 여부": p["heart_disease"].map({1: "있음", 0: "없음"}),
        }
    )


def show_list(column, heading, index):
    people = people_list(index)
    n = len(people)
    title = f"**{heading} (전체 {n:,}명)**"
    if n > MAX_ROWS:
        title += f" · 나이가 많은 앞 {MAX_ROWS}명만 보여 줘요"
    column.markdown(title)
    if n == 0:
        column.info("해당하는 사람이 없어요.")
    else:
        column.dataframe(people.head(MAX_ROWS), hide_index=True, width="stretch")


left, right = st.columns(2)
show_list(left, "놓친 환자 — 실제 뇌졸중인데 아니라고 예측한 사람", missed_index)
show_list(right, "헛짚은 사람 — 아닌데 뇌졸중이라고 예측한 사람", wrong_index)

c = conf[choice]
st.write(
    f"두 목록의 사람 수(놓친 환자 {len(missed_index):,}명, 헛짚은 사람 {len(wrong_index):,}명)는 "
    f"{choice}의 혼동행렬 FN 칸({c['fn']:,}명)과 FP 칸({c['fp']:,}명)과 같아요."
)
