import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="성능 높이기 · 뇌졸중 예측 실습실", page_icon="⚖️", layout="wide")

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
BINARY = {"hypertension", "heart_disease"}

LOGI = "로지스틱 회귀(확률로 답하는 모델)"
TREE = "의사결정트리(질문으로 답하는 모델)"

NO_COLOR = "#4C9F9A"       # 실제 아님 (점)
YES_COLOR = "#D9480F"      # 실제 뇌졸중 (점)
BEFORE_COLOR = "#2B2B2B"   # 가중치 맞추기 전 경계선·평면
AFTER_COLOR = "#7B2CBF"    # 가중치 맞춘 뒤 경계선·평면
BEFORE_NAME = "가중치 맞추기 전 0.5 경계선"
AFTER_NAME = "가중치 맞춘 뒤 0.5 경계선"

RANDOM_STATE = 42
CANNOT = "계산할 수 없음"


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

st.title("⚖️ 성능 높이기")
st.caption("뇌졸중을 겪은 사람이 아주 드물어서, 양쪽의 가중치를 맞추기 전과 맞춘 뒤의 결과를 나란히 비교해요.")

# ── 입력 속성 고르기 ('분류 모델' 페이지와 같다) ──────────────────────────
picked = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda f: KOR[f],
)
selected = [f for f in FEATURES if f in picked]

if len(selected) < 2:
    st.info("속성을 두 개 이상 골라 주세요. 두 개 이상이어야 그림을 그릴 수 있어요.")
    st.stop()

# ── 훈련용·테스트용 나누기 ('채점표' 페이지와 같다) ──────────────────────────
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

# ── 가중치: 드문 쪽(뇌졸중)의 한 명을 더 무겁게 센다 ─────────────────────────
n_train = len(y_train)
n_pos = int((y_train == 1).sum())
n_neg = n_train - n_pos
w_pos = n_train / (2 * n_pos)
w_neg = n_train / (2 * n_neg)

# ── 모델 학습: 가중치 맞추기 전(None) / 맞춘 뒤("balanced") ─────────────────
scaler = StandardScaler().fit(X_train)
Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)

logi_before = LogisticRegression(max_iter=1000).fit(Xs_train, y_train)
logi_after = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs_train, y_train)
tree_before = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
).fit(X_train, y_train)
tree_after = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE, class_weight="balanced"
).fit(X_train, y_train)

pred = {
    ("logi", "before"): pd.Series(logi_before.predict(Xs_test), index=X_test.index),
    ("logi", "after"): pd.Series(logi_after.predict(Xs_test), index=X_test.index),
    ("tree", "before"): pd.Series(tree_before.predict(X_test), index=X_test.index),
    ("tree", "after"): pd.Series(tree_after.predict(X_test), index=X_test.index),
}

total = len(y_test)
n_stroke = int((y_test == 1).sum())
st.write(f"테스트용 **{total:,}명** 가운데 뇌졸중인 사람은 **{n_stroke:,}명**이에요.")
st.write(
    f"'가중치를 맞춘다'는 건 드문 쪽(뇌졸중)의 한 명을 더 무겁게 세어 주는 거예요. "
    f"훈련용에서 뇌졸중 한 명은 **{w_pos:.2f}**, 아닌 사람 한 명은 **{w_neg:.2f}**로 세어서 학습했어요."
)


# ── 혼동행렬과 지표 ('채점표' 페이지와 같다) ────────────────────────────────
def confusion(p):
    actual_pos = y_test == 1
    pred_pos = p == 1
    return {
        "tp": int((actual_pos & pred_pos).sum()),
        "fn": int((actual_pos & ~pred_pos).sum()),
        "fp": int((~actual_pos & pred_pos).sum()),
        "tn": int((~actual_pos & ~pred_pos).sum()),
    }


def metrics(c):
    """(정확도, 재현율, 정밀도, F1). 계산할 수 없으면 None."""
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else None       # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) > 0 else None    # 분모: 뇌졸중이라 예측한 사람 수
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1


def fmt(v):
    return CANNOT if v is None else f"{v * 100:.2f}%"


def before_after_table(c0, c1):
    m0, m1 = metrics(c0), metrics(c1)
    data = {
        "TP (찾아낸 환자)": [f"{c0['tp']:,}명", f"{c1['tp']:,}명"],
        "FN (놓친 환자)": [f"{c0['fn']:,}명", f"{c1['fn']:,}명"],
        "FP (헛짚은 사람)": [f"{c0['fp']:,}명", f"{c1['fp']:,}명"],
        "TN (아닌 사람을 맞힘)": [f"{c0['tn']:,}명", f"{c1['tn']:,}명"],
    }
    for i, name in enumerate(["정확도", "재현율", "정밀도", "F1"]):
        data[name] = [fmt(m0[i]), fmt(m1[i])]
    return pd.DataFrame(data, index=["가중치 맞추기 전", "가중치 맞춘 뒤"]).T


# ── 1. 전·후 비교표 ──────────────────────────────────────────────
st.subheader("1. 가중치를 맞추기 전과 맞춘 뒤")
st.caption(
    "TP는 뇌졸중을 찾아낸 사람, FN은 놓친 환자, FP는 아닌데 뇌졸중이라고 헛짚은 사람, TN은 아닌 사람을 아니라고 맞힌 사람이에요. "
    "뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1은 계산할 수 없어요."
)
left, right = st.columns(2)
for col, key, name in ((left, "logi", LOGI), (right, "tree", TREE)):
    c0, c1 = confusion(pred[(key, "before")]), confusion(pred[(key, "after")])
    col.markdown(f"**{name}**")
    col.table(before_after_table(c0, c1))
    col.write(
        f"찾아낸 환자 {c0['tp']:,}명에서 {c1['tp']:,}명으로, "
        f"놓친 환자 {c0['fn']:,}명에서 {c1['fn']:,}명으로, "
        f"헛짚은 사람 {c0['fp']:,}명에서 {c1['fp']:,}명으로 바뀌었어요."
    )


# ── 0.5 경계선 계산 도우미 ───────────────────────────────────────
def equation(model):
    """원래 값 기준 식  c0 + Σ a_j·x_j = 0  (확률 0.5인 곳)."""
    w = model.coef_[0]
    b = float(model.intercept_[0])
    coef = {f: float(w[i]) / float(scaler.scale_[i]) for i, f in enumerate(selected)}
    c0 = b - sum(
        float(w[i]) * float(scaler.mean_[i]) / float(scaler.scale_[i]) for i in range(len(selected))
    )
    return coef, c0


EQUATIONS = [
    ("가중치를 맞추기 전의", BEFORE_NAME, BEFORE_COLOR, equation(logi_before)),
    ("가중치를 맞춘 뒤의", AFTER_NAME, AFTER_COLOR, equation(logi_after)),
]


def axis_range(f):
    if f in BINARY:
        return -0.4, 1.4
    lo, hi = float(X_test[f].min()), float(X_test[f].max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


def box_segment(a_x, a_y, c, xlo, xhi, ylo, yhi):
    """직선 c + a_x·x + a_y·y = 0 이 그림 상자와 만나는 두 점. 상자 밖이면 None."""
    tol_x, tol_y = 1e-9 * (xhi - xlo), 1e-9 * (yhi - ylo)
    found = []
    if abs(a_y) > 1e-12:
        for xv in (xlo, xhi):
            yv = -(c + a_x * xv) / a_y
            if ylo - tol_y <= yv <= yhi + tol_y:
                found.append((xv, min(max(yv, ylo), yhi)))
    if abs(a_x) > 1e-12:
        for yv in (ylo, yhi):
            xv = -(c + a_y * yv) / a_x
            if xlo - tol_x <= xv <= xhi + tol_x:
                found.append((min(max(xv, xlo), xhi), yv))
    unique = []
    for p in found:
        if all(
            abs(p[0] - q[0]) > 1e-9 * (xhi - xlo) or abs(p[1] - q[1]) > 1e-9 * (yhi - ylo)
            for q in unique
        ):
            unique.append(p)
    if len(unique) < 2:
        return None
    return max(
        ((p, q) for p in unique for q in unique),
        key=lambda pq: ((pq[0][0] - pq[1][0]) / (xhi - xlo)) ** 2
        + ((pq[0][1] - pq[1][1]) / (yhi - ylo)) ** 2,
    )


def outside_sentence(prefix, kind, value_at_center):
    side = "높아서 모두 뇌졸중이라고" if value_at_center > 0 else "낮아서 모두 아님이라고"
    return (
        f"{prefix} 0.5 {kind} **이 그림 밖에 있어요.** "
        f"그림 안의 어느 자리에서든 확률이 0.5보다 {side} 답해요."
    )


def stand_text(others, fixed):
    if not others:
        return "고른 속성이 그림의 축에 모두 쓰여서 따로 세워 둘 속성이 없어요."
    parts = ", ".join(
        f"{KOR[f]} {fixed[f]:g}" if f in BINARY else f"{KOR[f]} {fixed[f]:.1f}" for f in others
    )
    return f"축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **{parts}**"


def jitter(multiplier):
    order = pd.Series(range(len(X_test)), index=X_test.index)
    return ((order * multiplier) % 1 - 0.5) * 0.2


def tick_labels(feat):
    return dict(tickvals=[0, 1], ticktext=["없음", "있음"]) if feat in BINARY else {}


# ── 2. 2차원 산점도 + 전·후 0.5 경계선 ────────────────────────────────
st.subheader("2. 산점도와 0.5 경계선 (2차원)")
c1, c2 = st.columns(2)
x_feat = c1.selectbox("가로축", selected, format_func=lambda f: KOR[f], key="x2")
y_feat = c2.selectbox(
    "세로축", [f for f in selected if f != x_feat], format_func=lambda f: KOR[f], key="y2"
)
others2 = [f for f in selected if f not in (x_feat, y_feat)]
fixed2 = {f: float(X_test[f].median()) for f in others2}

xlo, xhi = axis_range(x_feat)
ylo, yhi = axis_range(y_feat)

fig2 = go.Figure()
plot_x = X_test[x_feat] + (jitter(0.6180339887) if x_feat in BINARY else 0)
plot_y = X_test[y_feat] + (jitter(0.7548776662) if y_feat in BINARY else 0)
for cls, name, color, size in [(0, "실제 아님", NO_COLOR, 6), (1, "실제 뇌졸중", YES_COLOR, 10)]:
    m = y_test == cls
    fig2.add_trace(
        go.Scatter(
            x=plot_x[m], y=plot_y[m], mode="markers", name=name,
            marker=dict(
                color=color, size=size, opacity=0.6 if cls == 0 else 0.95,
                line=dict(color="white", width=1 if cls == 1 else 0),
            ),
            customdata=pd.DataFrame({"x": X_test.loc[m, x_feat], "y": X_test.loc[m, y_feat]}).values,
            hovertemplate=f"{KOR[x_feat]}: %{{customdata[0]:g}}<br>{KOR[y_feat]}: %{{customdata[1]:g}}"
                          f"<extra>{name}</extra>",
        )
    )

notes2 = []
for prefix, legend_name, color, (coef, c0) in EQUATIONS:
    c_eff = c0 + sum(coef[f] * fixed2[f] for f in others2)
    a_x, a_y = coef[x_feat], coef[y_feat]
    seg = box_segment(a_x, a_y, c_eff, xlo, xhi, ylo, yhi)
    if seg is not None:
        (px0, py0), (px1, py1) = seg
        fig2.add_trace(
            go.Scatter(
                x=[px0, px1], y=[py0, py1], mode="lines", name=legend_name,
                line=dict(color=color, width=3, dash="dash" if color == BEFORE_COLOR else "solid"),
                hoverinfo="skip",
            )
        )
    else:
        center = c_eff + a_x * (xlo + xhi) / 2 + a_y * (ylo + yhi) / 2
        notes2.append(outside_sentence(prefix, "경계선은", center))

fig2.update_layout(
    height=540,
    xaxis=dict(title=KOR[x_feat], range=[xlo, xhi], **tick_labels(x_feat)),
    yaxis=dict(title=KOR[y_feat], range=[ylo, yhi], **tick_labels(y_feat)),
    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0),
    margin=dict(l=10, r=10, t=20, b=10),
)
st.plotly_chart(fig2)

if any(f in BINARY for f in (x_feat, y_feat)):
    st.caption("고혈압·심장병 축은 점이 겹쳐 보이지 않게 그림에서만 살짝 흩뿌렸어요. 마우스를 올리면 진짜 값이 나와요.")
st.write(f"점선은 가중치를 맞추기 전, 보라색 실선은 맞춘 뒤에 {LOGI}가 확률을 0.5로 가르는 선이에요.")
st.write(stand_text(others2, fixed2))
for sentence in notes2:
    st.write(sentence)


# ── 3. 3차원 산점도 + 전·후 0.5 평면 ──────────────────────────────────
def plane_polygon(a, c, lo, hi):
    """평면 c + a·p = 0 이 상자(lo~hi)와 만나는 다각형의 꼭짓점(순서대로). 상자 밖이면 None."""
    corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]

    def f(p):
        return c + a[0] * p[0] + a[1] * p[1] + a[2] * p[2]

    scale = max(h - l for l, h in zip(lo, hi))
    pts = []
    for i, p in enumerate(corners):
        for q in corners[i + 1:]:
            if sum(1 for k in range(3) if p[k] != q[k]) != 1:
                continue  # 상자의 모서리(한 좌표만 다른 두 꼭짓점)만 본다
            fp, fq = f(p), f(q)
            if fp == 0:
                pts.append(p)
            if fq == 0:
                pts.append(q)
            if fp * fq < 0:
                t = fp / (fp - fq)
                pts.append(tuple(p[k] + t * (q[k] - p[k]) for k in range(3)))
    unique = []
    for p in pts:
        if all(max(abs(p[k] - q[k]) for k in range(3)) > 1e-9 * scale for q in unique):
            unique.append(p)
    if len(unique) < 3:
        return None
    # 평면 위에서 중심을 기준으로 각도 순으로 늘어놓는다.
    cen = tuple(sum(p[k] for p in unique) / len(unique) for k in range(3))
    n = a
    helper = (1, 0, 0) if abs(n[0]) < abs(n[1]) and abs(n[0]) < abs(n[2]) else (0, 1, 0)

    def cross(u, v):
        return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])

    def unit(u):
        length = math.sqrt(sum(t * t for t in u))
        return tuple(t / length for t in u)

    u_axis = unit(cross(n, helper))
    v_axis = cross(unit(n), u_axis)

    def angle(p):
        d = tuple(p[k] - cen[k] for k in range(3))
        return math.atan2(sum(d[k] * v_axis[k] for k in range(3)), sum(d[k] * u_axis[k] for k in range(3)))

    return sorted(unique, key=angle)


st.subheader("3. 산점도와 0.5 평면 (3차원)")
if len(selected) < 3:
    st.info("속성을 셋 이상 고르면 세 축으로 돌려 볼 수 있는 3차원 그림이 나와요.")
else:
    d1, d2, d3 = st.columns(3)
    x3 = d1.selectbox("가로축(x)", selected, format_func=lambda f: KOR[f], key="x3")
    y3 = d2.selectbox(
        "세로축(y)", [f for f in selected if f != x3], format_func=lambda f: KOR[f], key="y3"
    )
    z3 = d3.selectbox(
        "높이축(z)", [f for f in selected if f not in (x3, y3)], format_func=lambda f: KOR[f], key="z3"
    )
    axes3 = [x3, y3, z3]
    others3 = [f for f in selected if f not in axes3]
    fixed3 = {f: float(X_test[f].median()) for f in others3}
    lo3 = tuple(axis_range(f)[0] for f in axes3)
    hi3 = tuple(axis_range(f)[1] for f in axes3)

    fig3 = go.Figure()
    coords = [
        X_test[f] + (jitter(mult) if f in BINARY else 0)
        for f, mult in zip(axes3, (0.6180339887, 0.7548776662, 0.5698402910))
    ]
    for cls, name, color, size in [(0, "실제 아님", NO_COLOR, 3), (1, "실제 뇌졸중", YES_COLOR, 5)]:
        m = y_test == cls
        fig3.add_trace(
            go.Scatter3d(
                x=coords[0][m], y=coords[1][m], z=coords[2][m], mode="markers", name=name,
                marker=dict(color=color, size=size, opacity=0.55 if cls == 0 else 0.95),
                customdata=X_test.loc[m, axes3].values,
                hovertemplate=(
                    f"{KOR[x3]}: %{{customdata[0]:g}}<br>{KOR[y3]}: %{{customdata[1]:g}}"
                    f"<br>{KOR[z3]}: %{{customdata[2]:g}}<extra>{name}</extra>"
                ),
            )
        )

    notes3 = []
    for prefix, legend_name, color, (coef, c0) in EQUATIONS:
        c_eff = c0 + sum(coef[f] * fixed3[f] for f in others3)
        a = tuple(coef[f] for f in axes3)
        poly = plane_polygon(a, c_eff, lo3, hi3) if max(abs(t) for t in a) > 1e-12 else None
        if poly is not None:
            m_pts = len(poly)
            fig3.add_trace(
                go.Mesh3d(
                    x=[p[0] for p in poly], y=[p[1] for p in poly], z=[p[2] for p in poly],
                    i=[0] * (m_pts - 2), j=list(range(1, m_pts - 1)), k=list(range(2, m_pts)),
                    color=color, opacity=0.5, name=legend_name.replace("경계선", "평면"),
                    showlegend=True, hoverinfo="skip", flatshading=True,
                )
            )
        else:
            center = c_eff + sum(a[k] * (lo3[k] + hi3[k]) / 2 for k in range(3))
            notes3.append(outside_sentence(prefix, "평면은", center))

    def scene_axis(feat, lo, hi):
        return dict(title=KOR[feat], range=[lo, hi], **tick_labels(feat))

    fig3.update_layout(
        height=680,
        scene=dict(
            xaxis=scene_axis(x3, lo3[0], hi3[0]),
            yaxis=scene_axis(y3, lo3[1], hi3[1]),
            zaxis=scene_axis(z3, lo3[2], hi3[2]),
            aspectmode="cube",
        ),
        legend=dict(orientation="h", yanchor="top", y=0.0, x=0),
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig3)

    st.caption("마우스로 끌면 그림이 돌아가고, 휠로 확대·축소할 수 있어요. 회색 면은 가중치를 맞추기 전, 보라색 면은 맞춘 뒤의 0.5 평면이에요.")
    if any(f in BINARY for f in axes3):
        st.caption("고혈압·심장병 축은 점이 겹쳐 보이지 않게 그림에서만 살짝 흩뿌렸어요. 마우스를 올리면 진짜 값이 나와요.")
    st.write(stand_text(others3, fixed3))
    for sentence in notes3:
        st.write(sentence)
