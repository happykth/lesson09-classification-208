import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="분류 모델 · 뇌졸중 예측 실습실", page_icon="🤖", layout="wide")

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
PART = {"age": "나이가", "avg_glucose_level": "평균 혈당이", "bmi": "체질량지수가"}
BINARY_QUESTION = {"hypertension": "고혈압이 있나요?", "heart_disease": "심장병이 있나요?"}
BINARY = set(BINARY_QUESTION)

LOGI = "로지스틱 회귀(확률로 답하는 모델)"
TREE = "의사결정트리(질문으로 답하는 모델)"
BASE = "입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델"

NO_COLOR = "#4C9F9A"        # 실제 아님 (점)
YES_COLOR = "#D9480F"       # 실제 뇌졸중 (점)
NO_LIGHT = "#DCEFED"        # 트리가 '아님'이라 답하는 칸 (옅은 색)
YES_LIGHT = "#FBDCC8"       # 트리가 '뇌졸중'이라 답하는 칸 (옅은 색)
LINE_COLOR = "#2B2B2B"

RANDOM_STATE = 42


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

st.title("🤖 분류 모델")
st.caption("뇌졸중(stroke가 1)을 양성으로 두고, 두 가지 모델이 얼마나 잘 맞히는지 비교해요.")

# ── 입력 속성 고르기 ─────────────────────────────────────────────
picked = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda f: KOR[f],
)
selected = [f for f in FEATURES if f in picked]  # 항상 같은 순서로 맞춘다

if len(selected) < 2:
    st.info("속성을 두 개 이상 골라 주세요. 두 개 이상이어야 그림을 그릴 수 있어요.")
    st.stop()

# ── 훈련용·테스트용 나누기 (번호 순 정렬 → 열 명씩 묶어 앞 세 명이 테스트용) ──
df = raw.sort_values("id").reset_index(drop=True)
position = pd.Series(range(len(df)))
is_test = (position % 10) < 3

X_all = df[selected].copy()
y_all = df["stroke"]
X_train, X_test = X_all[~is_test].copy(), X_all[is_test].copy()
y_train, y_test = y_all[~is_test], y_all[is_test]

st.write(
    f"전체 {len(df):,}명을 번호 순으로 세워 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 정해 두었어요. "
    f"**훈련용 {len(X_train):,}명, 테스트용 {len(X_test):,}명**이에요."
)

# ── 체질량지수 빈 값: 고른 경우에만 훈련용 중앙값으로 채우기 ─────────────────
if "bmi" in selected:
    bmi_median = float(X_train["bmi"].median())
    n_fill = int(X_train["bmi"].isna().sum() + X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    st.caption(f"체질량지수가 빈 {n_fill:,}명은 훈련용의 중앙값 {bmi_median:.1f}로 채웠어요.")

# ── 모델 학습 ────────────────────────────────────────────────────
# 로지스틱 회귀: 크기 맞추기(표준화)는 훈련용으로만 계산한다.
scaler = StandardScaler().fit(X_train)
logi = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)

# 의사결정트리: 질문 세 번까지, 마지막 마디는 다섯 명 이상, 난수 고정. (원래 값 그대로 사용)
tree = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
).fit(X_train, y_train)

majority = int(y_train.mode()[0])

logi_train = accuracy_score(y_train, logi.predict(scaler.transform(X_train)))
logi_test = accuracy_score(y_test, logi.predict(scaler.transform(X_test)))
tree_train = accuracy_score(y_train, tree.predict(X_train))
tree_test = accuracy_score(y_test, tree.predict(X_test))
base_train = float((y_train == majority).mean())
base_test = float((y_test == majority).mean())

# ── 1. 정확도 카드 ───────────────────────────────────────────────
st.subheader("1. 정확도 비교")

st.markdown(
    """
    <style>
    .acc-card {
        background: #FFF4E6;
        border: 2px solid #F4A261;
        border-radius: 16px;
        padding: 18px 14px 14px 14px;
        text-align: center;
        height: 100%;
    }
    .acc-title { color: #7A4A1E; font-size: 0.95rem; font-weight: 700; min-height: 3em; }
    .acc-big { color: #D9480F; font-size: 2.8rem; font-weight: 800; line-height: 1.25; }
    .acc-label { color: #7A4A1E; font-size: 0.85rem; margin-bottom: 8px; }
    .acc-small { display: flex; justify-content: space-around; color: #6B5B4B;
                 font-size: 0.85rem; border-top: 1px dashed #F4A261; padding-top: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def acc_card(title, train_acc, test_acc):
    return (
        f'<div class="acc-card">'
        f'<div class="acc-title">{title}</div>'
        f'<div class="acc-big">{test_acc * 100:.2f}%</div>'
        f'<div class="acc-label">테스트 데이터 정확도</div>'
        f'<div class="acc-small"><span>훈련 {train_acc * 100:.2f}%</span>'
        f"<span>테스트 {test_acc * 100:.2f}%</span></div>"
        f"</div>"
    )


for col, (title, tr, te) in zip(
    st.columns(3),
    [(LOGI, logi_train, logi_test), (TREE, tree_train, tree_test), (BASE, base_train, base_test)],
):
    col.markdown(acc_card(title, tr, te), unsafe_allow_html=True)

st.caption(
    "로지스틱 회귀는 크기를 맞춘(표준화한) 값으로, 의사결정트리는 원래 값 그대로 학습했어요. "
    "두 모델이 '많은 쪽으로만 답하는 모델'보다 정말 나은지 비교해 보세요."
)

# ── 2. 산점도 + 로지스틱 회귀 0.5 경계선 + 의사결정트리 칸 ─────────────────
st.subheader("2. 테스트 데이터 산점도와 두 모델의 경계")

c1, c2 = st.columns(2)
x_feat = c1.selectbox("가로축", selected, format_func=lambda f: KOR[f], key="x_axis")
y_feat = c2.selectbox(
    "세로축",
    [f for f in selected if f != x_feat],
    format_func=lambda f: KOR[f],
    key="y_axis",
)
others = [f for f in selected if f not in (x_feat, y_feat)]
fixed = {f: float(X_test[f].median()) for f in others}


def axis_range(f):
    if f in BINARY:
        return -0.4, 1.4
    lo, hi = float(X_test[f].min()), float(X_test[f].max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


xlo, xhi = axis_range(x_feat)
ylo, yhi = axis_range(y_feat)

# 의사결정트리가 나눈 칸: 격자 위의 모든 자리에서 트리의 답을 구한다.
n_grid = 150
xs = [xlo + (xhi - xlo) * i / (n_grid - 1) for i in range(n_grid)]
ys = [ylo + (yhi - ylo) * i / (n_grid - 1) for i in range(n_grid)]
grid = pd.DataFrame({x_feat: xs * n_grid, y_feat: [v for v in ys for _ in range(n_grid)]})
for f in others:
    grid[f] = fixed[f]
grid = grid[selected]
grid_pred = tree.predict(grid).tolist()
z = [grid_pred[i * n_grid:(i + 1) * n_grid] for i in range(n_grid)]
tree_says_yes_somewhere = any(v == 1 for v in grid_pred)

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x=xs, y=ys, z=z, zmin=0, zmax=1,
        colorscale=[[0, NO_LIGHT], [1, YES_LIGHT]],
        showscale=False, hoverinfo="skip",
    )
)

# 점: 같은 값이 겹치지 않게, 0/1 축에서만 아주 조금 흩뿌려 그린다(그림에서만).
order_in_test = pd.Series(range(len(X_test)), index=X_test.index)
jit_x = ((order_in_test * 0.6180339887) % 1 - 0.5) * 0.2
jit_y = ((order_in_test * 0.7548776662) % 1 - 0.5) * 0.2
plot_x = X_test[x_feat] + (jit_x if x_feat in BINARY else 0)
plot_y = X_test[y_feat] + (jit_y if y_feat in BINARY else 0)

for cls, name, color, size in [(0, "실제 아님", NO_COLOR, 6), (1, "실제 뇌졸중", YES_COLOR, 10)]:
    m = y_test == cls
    fig.add_trace(
        go.Scatter(
            x=plot_x[m], y=plot_y[m], mode="markers", name=name,
            marker=dict(color=color, size=size, opacity=0.6 if cls == 0 else 0.95,
                        line=dict(color="white", width=1 if cls == 1 else 0)),
            customdata=pd.DataFrame({"x": X_test.loc[m, x_feat], "y": X_test.loc[m, y_feat]}).values,
            hovertemplate=f"{KOR[x_feat]}: %{{customdata[0]:g}}<br>{KOR[y_feat]}: %{{customdata[1]:g}}"
                          f"<extra>{name}</extra>",
        )
    )

# 로지스틱 회귀가 확률 0.5로 가르는 선: c + a_x·x + a_y·y = 0
w = logi.coef_[0]
b = float(logi.intercept_[0])
coef = {f: float(w[i]) / float(scaler.scale_[i]) for i, f in enumerate(selected)}
c0 = b - sum(float(w[i]) * float(scaler.mean_[i]) / float(scaler.scale_[i]) for i in range(len(selected)))
c_eff = c0 + sum(coef[f] * fixed[f] for f in others)
a_x, a_y = coef[x_feat], coef[y_feat]


def line_in_box():
    """0.5 경계선이 그림 상자와 만나는 두 점. 상자 밖이면 None."""
    tol_x, tol_y = 1e-9 * (xhi - xlo), 1e-9 * (yhi - ylo)
    found = []
    if abs(a_y) > 1e-12:
        for xv in (xlo, xhi):
            yv = -(c_eff + a_x * xv) / a_y
            if ylo - tol_y <= yv <= yhi + tol_y:
                found.append((xv, min(max(yv, ylo), yhi)))
    if abs(a_x) > 1e-12:
        for yv in (ylo, yhi):
            xv = -(c_eff + a_y * yv) / a_x
            if xlo - tol_x <= xv <= xhi + tol_x:
                found.append((min(max(xv, xlo), xhi), yv))
    unique = []
    for p in found:
        if all(abs(p[0] - q[0]) > 1e-9 * (xhi - xlo) or abs(p[1] - q[1]) > 1e-9 * (yhi - ylo) for q in unique):
            unique.append(p)
    if len(unique) < 2:
        return None
    best = max(
        ((p, q) for p in unique for q in unique),
        key=lambda pq: ((pq[0][0] - pq[1][0]) / (xhi - xlo)) ** 2 + ((pq[0][1] - pq[1][1]) / (yhi - ylo)) ** 2,
    )
    return best


segment = line_in_box()
if segment is not None:
    (px0, py0), (px1, py1) = segment
    fig.add_trace(
        go.Scatter(
            x=[px0, px1], y=[py0, py1], mode="lines",
            name="로지스틱 회귀 0.5 경계선",
            line=dict(color=LINE_COLOR, width=3, dash="dash"), hoverinfo="skip",
        )
    )

# 칸 색깔 안내(범례용)
fig.add_trace(
    go.Scatter(x=[None], y=[None], mode="markers", name="의사결정트리: 아님이라 답하는 칸",
               marker=dict(symbol="square", size=14, color=NO_LIGHT, line=dict(color="#9AB", width=1)))
)
fig.add_trace(
    go.Scatter(x=[None], y=[None], mode="markers", name="의사결정트리: 뇌졸중이라 답하는 칸",
               marker=dict(symbol="square", size=14, color=YES_LIGHT, line=dict(color="#C98", width=1)))
)

fig.update_layout(
    height=560,
    xaxis=dict(title=KOR[x_feat], range=[xlo, xhi]),
    yaxis=dict(title=KOR[y_feat], range=[ylo, yhi]),
    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0),
    margin=dict(l=10, r=10, t=20, b=10),
)
for axis_name, feat in (("xaxis", x_feat), ("yaxis", y_feat)):
    if feat in BINARY:
        fig.update_layout(**{axis_name: dict(tickvals=[0, 1], ticktext=["없음", "있음"])})

st.plotly_chart(fig)

if any(f in BINARY for f in (x_feat, y_feat)):
    st.caption("고혈압·심장병 축은 점이 겹쳐 보이지 않게 그림에서만 살짝 흩뿌렸어요. 마우스를 올리면 진짜 값이 나와요.")

if others:
    fixed_text = ", ".join(
        f"{KOR[f]} {fixed[f]:g}" if f in BINARY else f"{KOR[f]} {fixed[f]:.1f}" for f in others
    )
    st.write(f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **{fixed_text}**")
else:
    st.write("고른 속성이 두 개뿐이라 따로 세워 둘 속성이 없어요.")

if segment is not None:
    st.write(f"점선은 {LOGI}가 뇌졸중일 확률을 0.5로 가르는 선이에요.")
else:
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    center_value = c_eff + a_x * cx + a_y * cy
    side = "높아서 모두 뇌졸중이라고" if center_value > 0 else "낮아서 모두 아님이라고"
    st.write(
        f"{LOGI}가 확률을 0.5로 가르는 선은 **이 그림 밖에 있어요.** "
        f"그림 안의 어느 자리에서든 이 모델이 계산한 확률이 0.5보다 {side} 답해요."
    )

if not tree_says_yes_somewhere:
    st.write(f"{TREE}는 이 그림 안의 어느 칸에서도 뇌졸중이라고 답하지 않아서, 칸이 모두 '아님' 색이에요.")

# ── 3. 의사결정트리의 질문 그림 ──────────────────────────────────
st.subheader("3. 의사결정트리가 던진 질문")

t = tree.tree_
path = tree.decision_path(X_train)
node_n = [int(v) for v in path.sum(axis=0).tolist()[0]]
node_yes = [int(v) for v in path[(y_train.values == 1)].sum(axis=0).tolist()[0]]


def node_text(i):
    n, k = node_n[i], node_yes[i]
    rate = k / n * 100 if n else 0.0
    return f"훈련용 {n:,}명\\n그중 뇌졸중 {k:,}명 ({rate:.1f}%)"


dot = [
    "digraph T {",
    'graph [rankdir=TB, nodesep=0.3, ranksep=0.55];',
    'node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=12, margin="0.2,0.12"];',
    'edge [fontname="sans-serif", fontsize=12];',
]
asked = set()
leaf_answers = []
edges = []
for i in range(t.node_count):
    left, right = int(t.children_left[i]), int(t.children_right[i])
    if left == -1:  # 더 묻지 않고 답을 내는 마디
        answer = 1 if node_yes[i] * 2 > node_n[i] else 0
        leaf_answers.append(answer)
        fill = YES_LIGHT if answer == 1 else NO_LIGHT
        border = YES_COLOR if answer == 1 else NO_COLOR
        label = f"답: {'뇌졸중' if answer == 1 else '아님'}\\n{node_text(i)}"
        dot.append(f'n{i} [label="{label}", fillcolor="{fill}", color="{border}", penwidth=2];')
    else:
        feat = selected[int(t.feature[i])]
        asked.add(feat)
        if feat in BINARY:
            question = BINARY_QUESTION[feat]
            yes_child, no_child = right, left   # 0.5보다 크면 '있음'
        else:
            question = f"{PART[feat]} {float(t.threshold[i]):.1f} 이하인가요?"
            yes_child, no_child = left, right
        dot.append(
            f'n{i} [label="{question}\\n{node_text(i)}", fillcolor="#FFF4E6", color="#F4A261"];'
        )
        edges.append(f'n{i} -> n{yes_child} [label="예"];')
        edges.append(f'n{i} -> n{no_child} [label="아니요"];')
dot += edges
dot.append("}")

st.graphviz_chart("\n".join(dot))

n_leaves = len(leaf_answers)
n_no = sum(1 for a in leaf_answers if a == 0)
st.write(f"답을 내는 마디는 모두 **{n_leaves}칸**이고, 그중 **{n_no}칸**이 아님이라고 답해요.")

asked_names = [KOR[f] for f in selected if f in asked]
if asked_names:
    st.write(f"고른 속성 가운데 이 나무가 실제로 물은 것은 **{', '.join(asked_names)}**이에요.")
else:
    st.write("이 나무는 아무것도 묻지 않고 바로 답을 냈어요.")
