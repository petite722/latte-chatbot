"""
라떼는 말이야 ☕ — SNU MBA 후배를 위한 AI 선배 챗봇
Streamlit 웹앱 버전

실행 방법:
  streamlit run latte_app.py

필요한 환경변수 (.env 파일 또는 시스템 환경변수):
  OPENAI_API_KEY=...
  TAVILY_API_KEY=...

필요한 파일 (같은 폴더에):
  halogen-chemist-437121-t6-c46e0c6ddb1e.json  (Google Sheets 서비스 계정 키)
  Latte/ 폴더 (강의계획서 PDF/DOCX + 후기 엑셀)
    ├── syllabus/
    └── data_260624.xlsx
"""

import os
import re
import json
import time
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from collections import Counter

load_dotenv()

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="라떼는 말이야 ☕",
    page_icon="☕",
    layout="wide",
)

st.markdown("""
<style>
/* 전체 배경 */
.stApp { background-color: #FDF6EC; }

/* 사이드바 */
[data-testid="stSidebar"] {
    background-color: #4A2C2A;
    color: #FDF6EC;
}
[data-testid="stSidebar"] * { color: #FDF6EC !important; }

/* 채팅 말풍선 */
.stChatMessage [data-testid="stChatMessageContent"] {
    background-color: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E8D5C0;
}

/* 입력창 */
.stChatInput textarea {
    background-color: #FFFFFF;
    border: 1.5px solid #C8733A;
    border-radius: 10px;
}

/* 헤더 */
h1 { color: #4A2C2A; }
h3 { color: #8B5E3C; }

/* 버튼 */
.stButton button {
    background-color: #C8733A;
    color: white;
    border-radius: 8px;
    border: none;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 데이터 로드 (캐싱 — 최초 1회만 실행)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="☕ 데이터 불러오는 중...")
def load_all():
    """벡터DB, 에이전트, Google Sheets 연결을 한 번만 초기화한다."""
    import pandas as pd
    import hashlib
    import shutil
    from pypdf import PdfReader
    from docx import Document as DocxDocument
    from langchain_core.documents import Document as LCDocument
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain.chat_models import init_chat_model
    from langchain_tavily import TavilySearch
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import gspread
    from google.oauth2.service_account import Credentials

    DATA_DIR = Path("Latte")
    SYLLABUS_DIR = DATA_DIR / "syllabus"
    REVIEW_XLSX = DATA_DIR / "data_260624.xlsx"
    CHROMA_DIR = Path("chroma_db")

    # 강의계획서 로드
    def load_syllabus_file(path):
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        elif path.suffix.lower() == ".docx":
            doc = DocxDocument(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        return ""

    SYLLABUS_FILES = list(SYLLABUS_DIR.glob("*")) if SYLLABUS_DIR.exists() else []
    syllabus_texts = [(f.stem, load_syllabus_file(f)) for f in SYLLABUS_FILES]

    # 후기 로드
    def load_professor_aliases(xlsx_path):
        df = pd.read_excel(xlsx_path, sheet_name="Data DB", header=2, usecols="Q:U")
        df.columns = ["과목명", "교수", "진행언어", "수강시기", "교수.1"]
        aliases = {}
        for _, row in df.iterrows():
            prof = str(row["교수"]).strip()
            alt = str(row["교수.1"]).strip() if pd.notna(row["교수.1"]) else ""
            if prof and prof != "nan":
                aliases[prof] = alt
        return aliases

    professor_aliases = load_professor_aliases(REVIEW_XLSX)

    def build_review_docs(xlsx_path):
        docs = []
        for sheet, lang in [("Data DB", "한국어"), ("Data DB_Eng", "영어")]:
            try:
                raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=2)
            except Exception:
                continue
            cols = list(raw.columns)
            for _, row in raw.iterrows():
                try:
                    course = str(row.iloc[3]).strip()
                    prof = str(row.iloc[4]).strip()
                    term = str(row.iloc[6]).strip()
                    review_text = str(row.iloc[10]).strip()
                    if not course or course == "nan" or not review_text or review_text == "nan":
                        continue
                    alt = professor_aliases.get(prof, "")
                    prof_display = f"{prof} ({alt})" if alt else prof
                    content = (
                        f"[과목명: {course}] [교수: {prof_display}] "
                        f"[언어: {lang}] [수강시기: {term}]\n후기: {review_text}"
                    )
                    docs.append(LCDocument(
                        page_content=content,
                        metadata={"type": "review", "course": course, "professor": prof,
                                  "language": lang, "term": term}
                    ))
                except Exception:
                    continue
        return docs

    review_docs = build_review_docs(REVIEW_XLSX)

    # 강의계획서 청크
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    syllabus_docs = []
    for name, text in syllabus_texts:
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        for chunk in chunks:
            syllabus_docs.append(LCDocument(
                page_content=chunk,
                metadata={"type": "syllabus", "source": name}
            ))

    all_docs = syllabus_docs + review_docs

    # 임베딩 & 벡터DB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if CHROMA_DIR.exists():
        vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(all_docs, embeddings, persist_directory=str(CHROMA_DIR))

    # Google Sheets
    review_sheet = None
    try:
        creds = Credentials.from_service_account_file(
            "halogen-chemist-437121-t6-c46e0c6ddb1e.json",
            scopes=["https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"]
        )
        gc = gspread.authorize(creds)
        review_sheet = gc.open_by_key("1VZ60mjNmSb-rwde7uYvcF21Se39skhbnH2Tlz-mJ4ew").worksheet("reviews")
    except Exception:
        pass  # Sheets 없어도 챗봇 작동

    return {
        "all_docs": all_docs,
        "review_docs": review_docs,
        "vectorstore": vectorstore,
        "professor_aliases": professor_aliases,
        "review_sheet": review_sheet,
    }


# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☕ 라떼는 말이야")
    st.markdown("SNU MBA 후배를 위한 AI 선배 챗봇")
    st.divider()
    st.markdown("**이런 질문을 해봐요:**")
    examples = [
        "박성호 교수님 Marketing Analytics 어때?",
        "이제호 교수님 Strategy 어때?",
        "팀플 적은 수업 추천해줘",
        "KEIT R&D 기획 직무에 도움 되는 수업은?",
        "후기 많은 과목 보여줘",
        "Portfolio Management vs Financial Engineering 비교해줘",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["suggested"] = ex

    st.divider()
    if st.button("🗑️ 대화 초기화"):
        st.session_state["messages"] = []
        st.session_state["conversation_history"] = []
        st.rerun()

    st.markdown("---")
    st.markdown("**수강후기 남기기**")
    st.markdown("채팅창에 `수강후기 남길게요` 또는 `leave a review` 라고 입력해봐요!")


# ──────────────────────────────────────────────
# 메인 화면
# ──────────────────────────────────────────────
st.title("☕ 라떼는 말이야")
st.markdown("**SNU MBA 선배들의 솔직한 수강후기 — AI가 정리해드려요**")
st.markdown("※ 과목명은 한국어/영어 그대로 입력해주시면 더 정확해요")
st.markdown("   Please use the exact course name in Korean or English for best results")

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "is_first_turn" not in st.session_state:
    st.session_state["is_first_turn"] = True

# 데이터 로드
try:
    resources = load_all()
    all_docs = resources["all_docs"]
    vectorstore = resources["vectorstore"]
    professor_aliases = resources["professor_aliases"]
    review_sheet = resources["review_sheet"]
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

# 기존 메시지 표시
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 추천 질문 자동 입력
prompt = st.session_state.pop("suggested", None)

# 채팅 입력
if user_input := (prompt or st.chat_input("궁금한 걸 입력하세요 / Ask anything")):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("☕ 선배들의 후기를 찾아보는 중..."):
            try:
                from langchain.messages import HumanMessage
                from langchain.agents import create_agent
                from langchain.chat_models import init_chat_model
                from langchain_tavily import TavilySearch
                from langchain.tools import tool
                from langchain_core.documents import Document as LCDocument

                # 노트북의 핵심 로직을 여기서 임포트해서 사용
                # (실제 배포 시에는 별도 모듈로 분리)
                answer = f"[데모] '{user_input}'에 대한 답변입니다. 실제 배포 시 노트북의 에이전트가 여기서 실행돼요."
                st.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})
                st.session_state["is_first_turn"] = False

            except Exception as e:
                error_msg = f"오류가 발생했어요: {e}"
                st.error(error_msg)

# ──────────────────────────────────────────────
# 하단 정보
# ──────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    review_count = sum(1 for d in all_docs if d.metadata.get("type") == "review")
    st.metric("📝 총 후기 수", f"{review_count}건")
with col2:
    course_count = len({d.metadata.get("course") for d in all_docs if d.metadata.get("type") == "review"})
    st.metric("📚 과목 수", f"{course_count}개")
with col3:
    sheets_status = "✅ 연결됨" if review_sheet else "⚠️ 미연결"
    st.metric("☁️ Google Sheets", sheets_status)
