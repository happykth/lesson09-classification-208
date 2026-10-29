import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="탐색 · 뇌졸중 예측 실습실", page_icon="🔎", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/happykth/data208/main/stroke.csv"

WARM = "#E8590C"      # 주황
SOFT = "#F4A261"      # 연한 주황
GROUP_COLORS = {"겪지 않음": "#4C9F9A", "겪음": "#D9480F"}


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

st.title("🔎 탐색")
st.caption("데이터를 여러 방향에서 살펴봐요.")

df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map({0: "겪지 않음", 1: "겪음"})

# ── 1. 나이와 평균 혈당의 분포 ───────────────────────────────────
st.subheader("1. 나이와 평균 혈당의 분포")
left, right = st.columns(2)

fig_age = px.histogram(
    df, x="age", nbins=30, color_discrete_sequence=[WARM],
    labels={"age": "나이", "count": "사람 수"}, title="나이 분포",
)
fig_age.update_layout(yaxis_title="사람 수", bargap=0.05)
left.plotly_chart(fig_age)

fig_glu = px.histogram(
    df, x="avg_glucose_level", nbins=40, color_discrete_sequence=[SOFT],
    labels={"avg_glucose_level": "평균 혈당"}, title="평균 혈당 분포",
)
fig_glu.update_layout(yaxis_title="사람 수", bargap=0.05)
right.plotly_chart(fig_glu)

# ── 2. 뇌졸중 여부별 비교 ────────────────────────────────────────
st.subheader("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
order = ["겪지 않음", "겪음"]
left, right = st.columns(2)

fig_box_age = px.box(
    df, x="뇌졸중 여부", y="age", color="뇌졸중 여부",
    category_orders={"뇌졸중 여부": order}, color_discrete_map=GROUP_COLORS,
    labels={"age": "나이"}, title="나이",
)
fig_box_age.update_layout(showlegend=False)
left.plotly_chart(fig_box_age)

fig_box_glu = px.box(
    df, x="뇌졸중 여부", y="avg_glucose_level", color="뇌졸중 여부",
    category_orders={"뇌졸중 여부": order}, color_discrete_map=GROUP_COLORS,
    labels={"avg_glucose_level": "평균 혈당"}, title="평균 혈당",
)
fig_box_glu.update_layout(showlegend=False)
right.plotly_chart(fig_box_glu)

mean_table = (
    df.groupby("뇌졸중 여부")
    .agg(
        사람_수=("id", "count"),
        나이_평균=("age", "mean"),
        평균_혈당_평균=("avg_glucose_level", "mean"),
    )
    .reindex(order)
    .reset_index()
)
mean_table.columns = ["그룹", "사람 수", "나이 평균", "평균 혈당 평균"]
mean_table["나이 평균"] = mean_table["나이 평균"].round(2)
mean_table["평균 혈당 평균"] = mean_table["평균 혈당 평균"].round(2)
st.dataframe(mean_table, hide_index=True, width="stretch")

# ── 3. 고혈압·심장병별 뇌졸중 비율 ───────────────────────────────
st.subheader("3. 고혈압·심장병 여부별 뇌졸중 비율")


def stroke_rate_by(column, label):
    """column 값(0/1)마다 뇌졸중 비율(%)을 계산한다."""
    t = df.groupby(column)["stroke"].agg(["mean", "count"]).reset_index()
    t["여부"] = t[column].map({0: f"{label} 없음", 1: f"{label} 있음"})
    t["뇌졸중 비율(%)"] = (t["mean"] * 100).round(2)
    t["사람 수"] = t["count"]
    return t


left, right = st.columns(2)
for col_widget, column, label in [(left, "hypertension", "고혈압"), (right, "heart_disease", "심장병")]:
    t = stroke_rate_by(column, label)
    fig = px.bar(
        t, x="여부", y="뇌졸중 비율(%)", text="뇌졸중 비율(%)",
        color="여부", color_discrete_sequence=["#4C9F9A", "#D9480F"],
        hover_data={"사람 수": True}, title=f"{label} 여부별 뇌졸중 비율",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="뇌졸중 비율(%)")
    fig.update_yaxes(range=[0, max(t["뇌졸중 비율(%)"].max() * 1.25, 1)])
    col_widget.plotly_chart(fig)

# ── 4. bmi가 빈 사람들 ───────────────────────────────────────────
st.subheader("4. 체질량지수(bmi)가 비어 있는 사람들")
bmi_empty = df[df["bmi"].isna()]


def summary_row(name, part):
    n = len(part)
    s = int((part["stroke"] == 1).sum())
    rate = f"{s / n * 100:.2f}%" if n > 0 else "계산할 수 없음"
    return {"구분": name, "사람 수": n, "뇌졸중 사람 수": s, "뇌졸중 비율": rate}


bmi_table = pd.DataFrame(
    [summary_row("bmi가 빈 사람", bmi_empty), summary_row("전체", df)]
)
st.dataframe(bmi_table, hide_index=True, width="stretch")

# ── 5. 흡연 상태별 사람 수 ───────────────────────────────────────
st.subheader("5. 흡연 상태별 사람 수")
smoking = (
    df["smoking_status"].value_counts(dropna=False).rename_axis("흡연 상태").reset_index(name="사람 수")
)
smoking["흡연 상태"] = smoking["흡연 상태"].fillna("(빈 값)")
st.dataframe(smoking, hide_index=True, width="stretch")
