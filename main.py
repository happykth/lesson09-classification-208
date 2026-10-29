import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── 기본 설정: 브라우저 탭 제목과 아이콘 ──────────────────────────
st.set_page_config(page_title="뇌졸중 예측 실습실", page_icon="🧠", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/happykth/data208/main/stroke.csv"
SAVE_DIR = Path(__file__).parent / "내_기록"
SAVE_FILE = SAVE_DIR / "데이터_소개.json"


# ── 기록 불러오기 / 저장하기 ─────────────────────────────────────
def load_notes():
    """내_기록 폴더의 json을 읽어 {'meanings': {...}, 'source': '...'} 형태로 돌려준다."""
    notes = {"meanings": {}, "source": ""}
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            notes["meanings"] = data.get("meanings", {})
            notes["source"] = data.get("source", "")
        except (json.JSONDecodeError, OSError):
            pass  # 파일이 깨졌으면 빈 기록으로 시작한다
    return notes


def save_notes(meanings=None, source=None):
    """바뀐 부분만 갈아 끼워서 json 파일에 저장한다."""
    notes = load_notes()
    if meanings is not None:
        notes["meanings"] = meanings
    if source is not None:
        notes["source"] = source
    SAVE_DIR.mkdir(exist_ok=True)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


# ── 데이터 불러오기 ──────────────────────────────────────────────
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def describe_values(s):
    """열의 값이 어떤 종류인지 한 줄로 설명한다."""
    uniq = s.dropna().unique()
    if s.dtype == "object" or len(uniq) <= 5:
        return ", ".join(sorted(str(v) for v in uniq))
    return f"숫자 {s.min():g} ~ {s.max():g} ({len(uniq)}가지)"


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

notes = load_notes()

# ── 맨 위: 아이콘 + 제목 ─────────────────────────────────────────
st.title("🧠 뇌졸중 예측 실습실")
st.caption("첫 화면 · 이 데이터는 무엇일까요?")

# ── 큰 숫자 카드 네 개 ───────────────────────────────────────────
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100

st.markdown(
    """
    <style>
    .num-card {
        background: #FFF4E6;
        border: 2px solid #F4A261;
        border-radius: 16px;
        padding: 20px 12px;
        text-align: center;
    }
    .num-card .label { color: #7A4A1E; font-size: 1rem; font-weight: 600; }
    .num-card .value { color: #D9480F; font-size: 2.6rem; font-weight: 800; line-height: 1.3; }
    </style>
    """,
    unsafe_allow_html=True,
)

cards = [
    ("전체 사람 수", f"{total:,}명"),
    ("열 개수", f"{n_cols}개"),
    ("stroke가 1인 사람 수", f"{n_stroke:,}명"),
    ("stroke가 1인 비율", f"{ratio:.2f}%"),
]
for col, (label, value) in zip(st.columns(4), cards):
    col.markdown(
        f'<div class="num-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )

# ── 열 설명 표 (우리말 뜻은 직접 입력) ───────────────────────────
st.subheader("열 설명 표")
st.write("**우리말 뜻** 칸을 두 번 눌러 교재를 보고 직접 적어 넣고, 아래 저장 버튼을 눌러 주세요.")

table = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [notes["meanings"].get(c, "") for c in df.columns],
        "값의 종류": [describe_values(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    }
)

with st.form("meaning_form"):
    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "우리말 뜻": st.column_config.TextColumn("우리말 뜻", width="medium"),
        },
        key="meaning_editor",
    )
    meaning_saved = st.form_submit_button("우리말 뜻 저장하기")

if meaning_saved:
    meanings = {
        row["열 이름"]: (row["우리말 뜻"] or "").strip() for _, row in edited.iterrows()
    }
    save_notes(meanings=meanings)
    st.success("우리말 뜻을 저장했어요. 새로고침해도 다시 불러와요.")

# ── 데이터 처음 다섯 줄 ──────────────────────────────────────────
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), width="stretch")

# ── 데이터 출처 ──────────────────────────────────────────────────
st.subheader("데이터 출처")
with st.form("source_form"):
    source_text = st.text_area(
        "교재에 있는 출처 글을 여기에 적어 주세요.",
        value=notes["source"],
        height=120,
    )
    source_saved = st.form_submit_button("출처 저장하기")

if source_saved:
    save_notes(source=source_text)
    st.success("출처를 저장했어요. 새로고침해도 다시 불러와요.")
