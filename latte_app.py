"""
라떼는 말이야 ☕ — SNU MBA 후배를 위한 AI 선배 챗봇
Streamlit 웹앱 버전
"""

import os, re, json, time, shutil, zipfile, hashlib
from pathlib import Path
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="라떼는 말이야 ☕", page_icon="☕", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #FDF6EC; }
[data-testid="stSidebar"] { background-color: #4A2C2A; }
[data-testid="stSidebar"] * { color: #FDF6EC !important; }
[data-testid="stSidebar"] .stButton button { 
    color: #4A2C2A !important; 
    background-color: #FDF6EC !important;
    border: none;
}
h1 { color: #4A2C2A; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="☕ 데이터 불러오는 중... (최초 실행 시 시간이 걸려요)")
def load_all():
    import gdown, pandas as pd
    from pypdf import PdfReader
    from docx import Document as DocxDocument
    from langchain_core.documents import Document as LCDocument
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    from langchain.chat_models import init_chat_model
    from langchain.messages import HumanMessage as _HM

    # 1. Google Drive 다운로드
    DATA_DIR = Path('Latte')
    if not DATA_DIR.exists():
        gdown.download_folder(
            id='1EcjtGxTWVPqrL0-2x0Dz4aualr_dD89P',
            output=str(DATA_DIR), quiet=False, use_cookies=False
        )

    zip_candidates = list(DATA_DIR.rglob('*.zip'))
    syllabus_extract_dir = DATA_DIR / 'syllabus_extracted'
    if zip_candidates and not syllabus_extract_dir.exists():
        syllabus_extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_candidates[0], 'r') as z:
            z.extractall(syllabus_extract_dir)

    SYLLABUS_FILES = sorted(
        list(syllabus_extract_dir.rglob('*.pdf')) +
        list(syllabus_extract_dir.rglob('*.docx'))
    ) if syllabus_extract_dir.exists() else []

    xlsx_list = [p for p in DATA_DIR.rglob('*.xlsx') if not p.name.startswith('~$')]
    assert xlsx_list, 'Latte 폴더에서 엑셀 파일을 찾지 못했습니다.'
    REVIEW_XLSX = xlsx_list[0]

    # 2. 강의계획서 로드
    def load_syl(path):
        if path.suffix.lower() == '.pdf':
            return '\n'.join(p.extract_text() or '' for p in PdfReader(str(path)).pages)
        elif path.suffix.lower() == '.docx':
            return '\n'.join(p.text for p in DocxDocument(str(path)).paragraphs)
        return ''

    syllabus_texts = [(f.stem, load_syl(f)) for f in SYLLABUS_FILES]

    # 3. 후기 로드
    raw_kr = pd.read_excel(REVIEW_XLSX, sheet_name='Data DB', header=1)
    raw_kr.columns = [str(c).strip() for c in raw_kr.columns]
    raw_en = pd.read_excel(REVIEW_XLSX, sheet_name='Data DB_Eng', header=1)
    raw_en.columns = [str(c).strip() for c in raw_en.columns]
    raw_en = raw_en.rename(columns={
        'Student': '수강자', 'Course Title': '과목명', 'Professor': '교수',
        'Language': '진행 언어', 'Term/Module': '수강 시기(모듈)',
        'Comments / Review': '후기(자유 서술)',
    })
    raw = pd.concat([raw_kr, raw_en], ignore_index=True)
    reviews_df = raw[['수강자','과목명','교수','진행 언어','수강 시기(모듈)','후기(자유 서술)']].copy()
    reviews_df = reviews_df.dropna(subset=['과목명','후기(자유 서술)']).reset_index(drop=True)

    # 4. 교수 별칭
    def load_aliases(xlsx):
        db = pd.read_excel(xlsx, sheet_name='Data DB', header=2, usecols='Q:U')
        db.columns = [str(c).strip() for c in db.columns]
        al = {}
        for _, row in db.iterrows():
            p = str(row.get('교수','')).strip()
            a = str(row.get('교수.1','')).strip() if pd.notna(row.get('교수.1')) else ''
            if p and p != 'nan':
                al[p] = a
        return al

    professor_aliases = load_aliases(REVIEW_XLSX)

    # 5. Document 생성
    def build_review_docs(df, aliases):
        cnt = df['과목명'].value_counts().to_dict()
        docs = []
        for _, row in df.iterrows():
            c, p = row['과목명'], row['교수']
            alt = aliases.get(str(p).strip(), '')
            pd_ = f'{p} ({alt})' if alt else p
            docs.append(LCDocument(
                page_content=(
                    f'[과목명: {c}] [교수: {pd_}] '
                    f'[언어: {row["진행 언어"]}] [수강시기: {row["수강 시기(모듈)"]}]\n'
                    f'후기: {row["후기(자유 서술)"]}'
                ),
                metadata={'type':'review','course':c,'professor':p,
                          'language':row['진행 언어'],'term':str(row['수강 시기(모듈)']),
                          'review_count_for_course':cnt.get(c,0)}
            ))
        return docs

    review_docs = build_review_docs(reviews_df, professor_aliases)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    _em = init_chat_model(model='openai:gpt-4o-mini')

    def extract_name(text):
        try:
            r = _em.invoke([_HM(f'다음 강의계획서에서 정식 과목명만 한 줄로 답하라:\n\n{text[:2000]}')])
            return r.content.strip().split('\n')[0].strip()
        except Exception:
            return ''

    syllabus_docs = []
    for name, text in syllabus_texts:
        if not text.strip():
            continue
        cname = extract_name(text) or name
        for chunk in splitter.split_text(text):
            syllabus_docs.append(LCDocument(
                page_content=chunk,
                metadata={'type':'syllabus','source':name,'course':cname}
            ))

    all_docs = syllabus_docs + review_docs

    # 6. 벡터DB
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

    syl_db = 'chroma_syllabus_db'
    rev_db = 'chroma_review_db'
    syllabus_vs = (Chroma(persist_directory=syl_db, embedding_function=embeddings)
                   if Path(syl_db).exists()
                   else Chroma.from_documents(syllabus_docs, embeddings, persist_directory=syl_db))
    review_vs = (Chroma(persist_directory=rev_db, embedding_function=embeddings)
                 if Path(rev_db).exists()
                 else Chroma.from_documents(review_docs, embeddings, persist_directory=rev_db))

    class CombinedVS:
        def __init__(self, s, r): self.s, self.r = s, r
        def similarity_search(self, q, k=10):
            return self.s.similarity_search(q, k=k//2+1) + self.r.similarity_search(q, k=k//2+1)

    vectorstore = CombinedVS(syllabus_vs, review_vs)

    python    # 7. Sheets 연결
    review_sheet = None
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets',
                  'https://www.googleapis.com/auth/drive']
        if 'google_sheets' in st.secrets:
            creds = Credentials.from_service_account_info(
                json.loads(st.secrets['google_sheets']['credentials']), scopes=SCOPES)
        elif 'GOOGLE_SHEETS_CREDENTIALS' in st.secrets:
            creds = Credentials.from_service_account_info(
                json.loads(st.secrets['GOOGLE_SHEETS_CREDENTIALS']), scopes=SCOPES)
        else:
            creds = Credentials.from_service_account_file(
                'halogen-chemist-437121-t6-c46e0c6ddb1e.json', scopes=SCOPES)
        gc = gspread.authorize(creds)
        review_sheet = gc.open_by_key('1VZ60mjNmSb-rwde7uYvcF21Se39skhbnH2Tlz-mJ4ew').worksheet('reviews')
    except Exception as e:
        st.warning(f"Google Sheets 연결 실패: {e}")

    # 8. 에이전트 초기화 (노트북 셀 30, 31 로직)
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_tavily import TavilySearch
    from langchain_core.messages import ToolMessage
    import pandas as _pd

    _exam_questions = None

    def format_sources(docs):
        return '\n\n---\n\n'.join(d.page_content for d in docs)

    def normalize_name(text):
        cleaned = text.replace(',', ' ').replace('-', ' ').replace('/', ' ')
        return set(cleaned.split())

    def build_course_catalog(xlsx):
        db = _pd.read_excel(xlsx, sheet_name='Data DB', header=2, usecols='Q:U')
        db.columns = [str(c).strip() for c in db.columns]
        catalog = []
        for _, row in db.iterrows():
            c, p, a = row.get('과목명'), row.get('교수'), row.get('교수.1')
            if _pd.notna(c) and _pd.notna(p):
                catalog.append({'course':str(c).strip(),'professor':str(p).strip(),
                                 'alt':str(a).strip() if _pd.notna(a) else ''})
        return catalog

    course_catalog = build_course_catalog(REVIEW_XLSX)

    @tool(response_format='content_and_artifact')
    def retriever_tool(query: str) -> tuple:
        """강의계획서와 강의 후기에서 질문과 관련된 내용을 검색한다."""
        nonlocal _exam_questions
        retrieved = vectorstore.similarity_search(query, k=10)
        ql = query.lower()

        def norm(t): return t.lower().replace(' ','').replace('-','')

        rev_only = [d for d in all_docs if d.metadata.get('type') == 'review']
        known_courses = sorted({d.metadata.get('course','') for d in rev_only}, key=len, reverse=True)
        qt = ql.split()

        def course_token_match(c):
            profs = {d.metadata.get('professor','') for d in rev_only if d.metadata.get('course') == c}
            st_ = c.lower() + ' ' + ' '.join(p.lower() for p in profs)
            return all(t in st_ for t in qt)

        matched_course = next((c for c in known_courses if c and c.lower() in ql), None)
        if not matched_course:
            matched_course = next((c for c in known_courses if c and course_token_match(c)), None)

        matched_professor = None
        known_profs = {d.metadata.get('professor','') for d in rev_only}
        for p in known_profs:
            if not p: continue
            alt = professor_aliases.get(p, '')
            alt_parts = [a.strip() for a in alt.split(',') if a.strip()]
            pn, qn = norm(p), norm(ql)
            alt_norms = [norm(a) for a in alt_parts]
            if (p.lower() in ql or pn in qn or
                any(a.lower() in ql for a in alt_parts) or
                any(an in qn for an in alt_norms if len(an) > 4)):
                matched_professor = p
                break

        if matched_professor and matched_course:
            pc = {d.metadata.get('course','') for d in all_docs
                  if d.metadata.get('type') == 'review' and d.metadata.get('professor','') == matched_professor}
            if pc and matched_course not in pc:
                matched_course = None

        if matched_course:
            syl_f = [d for d in retrieved if d.metadata.get('type') != 'review']
            if matched_professor:
                alt = professor_aliases.get(matched_professor, '')
                pnames = {matched_professor.lower()} | {a.lower() for a in alt.split(',') if a.strip()}
                rev_f = [d for d in all_docs if d.metadata.get('type') == 'review'
                         and d.metadata.get('course','').lower() == matched_course.lower()
                         and d.metadata.get('professor','').lower() in pnames]
            else:
                rev_f = [d for d in all_docs if d.metadata.get('type') == 'review'
                         and d.metadata.get('course','').lower() == matched_course.lower()]
            retrieved = syl_f + rev_f
        elif matched_professor:
            alt = professor_aliases.get(matched_professor, '')
            pnames = {matched_professor.lower()} | {a.lower() for a in alt.split(',') if a.strip()}
            syl_f = [d for d in retrieved if d.metadata.get('type') != 'review']
            rev_f = [d for d in all_docs if d.metadata.get('type') == 'review'
                     and d.metadata.get('professor','').lower() in pnames]
            retrieved = syl_f + rev_f

        exam_texts = []
        for d in retrieved:
            if d.metadata.get('type') == 'review':
                for kw in ['기출시험문제 공유','기출시험문제','시험기출문제','기출문제','기출 문제']:
                    if kw in d.page_content:
                        part = d.page_content.split(kw)[1].strip().lstrip(':').strip()
                        if part: exam_texts.append(part)
                        break
        _exam_questions = '\n\n'.join(exam_texts) if exam_texts else None

        return format_sources(retrieved), retrieved

    @tool(response_format='content_and_artifact')
    def professor_courses_tool(professor_name: str) -> tuple:
        """특정 교수님이 담당하는 모든 과목을 찾는다."""
        qw = normalize_name(professor_name)
        def norm(t): return t.lower().replace(' ','').replace('-','').replace(',','')
        def norm_sorted(t): return ''.join(sorted(norm(t)))
        qns = norm_sorted(professor_name)
        matched = []
        for entry in course_catalog:
            cand = f"{entry['professor']} {entry['alt']}"
            cw = normalize_name(cand)
            names = []
            for raw in [entry['professor'], entry['alt']]:
                for sep in ['/',',']:
                    names.extend([p.strip() for p in raw.split(sep) if p.strip()])
            if (qw and qw.issubset(cw)) or any(norm_sorted(n) == qns for n in set(names)):
                matched.append(entry)
        if not matched:
            return f'"{professor_name}" 교수님을 찾지 못했어요. 성함을 다시 확인해 주세요.', []
        result_docs, lines = [], []
        for e in matched:
            cr = [d for d in all_docs if d.metadata.get('type') == 'review' and d.metadata.get('course') == e['course']]
            result_docs.extend(cr)
            lines.append(f"- {e['course']}: {'후기 '+str(len(cr))+'건 있음' if cr else '아직 후기 없음'}")
        return professor_name + ' 교수님 담당 과목:\n' + '\n'.join(lines) + '\n\n' + format_sources(result_docs), result_docs

    @tool
    def general_knowledge_tool(question: str) -> str:
        """내부 자료가 부족할 때 일반 지식으로 답한다."""
        return '내부 자료가 부족하다. MBA 일반 지식으로 신중하게 답하라.'

    web_search_tool = TavilySearch(max_results=3, name='web_search_tool',
        description='회사/직무가 무엇을 하는 곳인지 파악하기 위해 웹을 검색한다.')

    @tool(response_format='content_and_artifact')
    def career_recommendation_tool(keywords: str) -> tuple:
        """직무/회사 키워드로 관련 과목을 검색한다."""
        docs = vectorstore.similarity_search(keywords, k=15)
        return format_sources(docs), docs

    @tool(response_format='content_and_artifact')
    def course_comparison_tool(query: str) -> tuple:
        """두 과목을 비교한다."""
        from difflib import get_close_matches
        known = sorted({d.metadata.get('course','') for d in all_docs if d.metadata.get('course')}, key=len, reverse=True)
        ql = query.lower()
        matched = [c for c in known if c and c.lower() in ql]
        if len(matched) < 2:
            cleaned = query
            for t in ['비교해줘','비교','랑','이랑','하고','과','와','compare','vs','versus','between','and']:
                cleaned = cleaned.replace(t, '|')
            for cand in [p.strip() for p in cleaned.split('|') if len(p.strip()) >= 2]:
                m = get_close_matches(cand, known, n=1, cutoff=0.45)
                if m: matched.append(m[0])
        matched = list(dict.fromkeys(matched))
        comp_docs = []
        if len(matched) >= 2:
            for c in matched[:3]:
                comp_docs.extend([d for d in all_docs if d.metadata.get('course','').lower() == c.lower()])
        if not comp_docs:
            comp_docs = vectorstore.similarity_search(query, k=12)
        return format_sources(comp_docs), comp_docs

    @tool
    def count_reviews_tool(course_name: str = '') -> str:
        """후기 개수를 집계한다."""
        from collections import Counter
        counts = Counter(d.metadata.get('course','') for d in review_docs)
        if course_name:
            return f'{course_name} 후기: {counts.get(course_name, 0)}건'
        return '과목별 후기 수: ' + ', '.join(f'{c}: {n}건' for c, n in counts.most_common())

    # 시스템 프롬프트는 노트북과 동일 (축약)
    SYSTEM_PROMPT = '''너는 SNU MBA 후배들에게 과목과 학교생활에 대해 알려주는 AI 선배 챗봇 "라떼는 말이야"이다.

[기본 동작]
질문이 과목, 교수님, 진로/직무와 관련 있다면 반드시 적절한 도구를 먼저 호출한다.
과목명이 있으면 retriever_tool, 교수님 전체 과목이면 professor_courses_tool, 비교이면 course_comparison_tool, 진로추천이면 web_search_tool→career_recommendation_tool, 집계이면 count_reviews_tool을 사용한다.

[답변 형식]
JSON으로만 출력한다. 코드블록 없이 순수 JSON만.
한국어 질문이면 value는 한국어, 영어 질문이면 value는 영어.
과목 1개: {"수업방식":"...","시험":"...","교수스타일":"...",...}
과목 2개 이상: [{"과목명":"...","수업방식":"..."},{"과목명":"..."}]
진로추천: [{"과목명":"...","추천이유":"...","얻을수있는역량":"...","수업방식":"...","평가방식":"...","주의할점":"..."}]
후기에 없는 항목은 포함하지 않는다. 원문에 없는 내용은 추가하지 않는다.
과목명은 내부 자료 표기 그대로 사용한다. 번역, 괄호 추가, 접두어 금지.

[말투]
한국어: ~해요, ~더라구요 스타일. ~합니다/~입니다 금지.
영어: 자연스럽고 친근한 영어.
'''

    model = init_chat_model(model='openai:gpt-4.1-mini')
    agent = create_agent(
        model=model,
        tools=[retriever_tool, professor_courses_tool, general_knowledge_tool,
               web_search_tool, career_recommendation_tool, course_comparison_tool, count_reviews_tool],
        system_prompt=SYSTEM_PROMPT,
    )

    return {
        'all_docs': all_docs,
        'review_docs': review_docs,
        'review_vectorstore': review_vs,
        'vectorstore': vectorstore,
        'professor_aliases': professor_aliases,
        'review_sheet': review_sheet,
        'agent': agent,
        '_exam_questions_ref': [None],
    }


# ──────────────────────────────────────────────
# 노트북 셀 31: 헬퍼 함수들 (parse_json_answer, build_header 등)
# ──────────────────────────────────────────────
import re

TOOL_NAMES = {
    'retriever_tool',
    'professor_courses_tool',
    'general_knowledge_tool',
    'career_recommendation_tool',
    'course_comparison_tool',
    'count_reviews_tool'
}
GENERAL_TOOL_NAME = 'general_knowledge_tool'
CAREER_TOOL_NAME = 'career_recommendation_tool'
WEB_SEARCH_TOOL_NAME = 'web_search_tool'
PROFESSOR_TOOL_NAME = 'professor_courses_tool'

def extract_text(result):
    content = result['messages'][-1].content
    if isinstance(content, str):
        return content
    return '\n'.join(block.get('text', '') for block in content if isinstance(block, dict) and block.get('type') == 'text')

def n_reviews_from(docs):
    review_docs = [d for d in docs if d.metadata.get('type') == 'review']
    if not review_docs:
        return 0
    courses = {d.metadata.get('course', '') for d in review_docs}
    return sum(1 for d in all_docs if d.metadata.get('type') == 'review' and d.metadata.get('course', '') in courses)

def extract_n_reviews(result):
    n = 0
    for message in result['messages']:
        if isinstance(message, ToolMessage) and message.name in TOOL_NAMES and message.name != GENERAL_TOOL_NAME:
            artifact = getattr(message, 'artifact', None)
            if artifact:
                n = n_reviews_from(artifact)
    return n

def extract_json_values_text_for_lang(text):
    """
    언어 감지용 텍스트를 만든다.
    LLM 답변이 JSON이면 한국어 key(수업방식/교수스타일 등)는 제외하고
    value만 모아서 반환한다. JSON이 아니면 원문을 그대로 반환한다.
    """
    try:
        import json as _json_lang
        clean = re.sub(r'```json|```', '', str(text)).strip()
        parsed = _json_lang.loads(clean)
        values = []

        def walk(x):
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for item in x:
                    walk(item)
            elif isinstance(x, str):
                values.append(x)

        walk(parsed)
        return ' '.join(values) if values else str(text)
    except Exception:
        return str(text)

def is_english(text):
    # JSON 답변은 key가 아니라 value 기준으로 언어를 판단한다.
    content = extract_json_values_text_for_lang(text)
    if len(re.findall(r'[가-힣]', content)) > 0:
        return False
    return len(re.findall(r'[a-zA-Z]', content)) > 0

def collect_searched_pairs(result):
    pairs = []
    seen = set()
    for message in result['messages']:
        if isinstance(message, ToolMessage) and message.name in TOOL_NAMES and message.name != GENERAL_TOOL_NAME:
            artifact = getattr(message, 'artifact', None)
            if not artifact:
                continue
            for d in artifact:
                if d.metadata.get('type') != 'review':
                    continue
                pair = (d.metadata.get('course', ''), d.metadata.get('professor', ''))
                if pair[0] and pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
    return pairs

def get_header_info(result, answer):
    is_prof_tool = any(isinstance(m, ToolMessage) and m.name == PROFESSOR_TOOL_NAME for m in result['messages'])

    if is_prof_tool:
        mentioned = []
        seen = set()
        for m in result['messages']:
            if isinstance(m, ToolMessage) and m.name == PROFESSOR_TOOL_NAME:
                artifact = getattr(m, 'artifact', None)
                if artifact:
                    for d in artifact:
                        course = d.metadata.get('course', '')
                        prof = d.metadata.get('professor', '')
                        if course and course not in seen:
                            seen.add(course)
                            mentioned.append(f"{course} ({prof})")
        return mentioned, ''

    candidate_pairs = collect_searched_pairs(result)
    if not candidate_pairs:
        return [], ''

    answer_lower = answer.lower()
    mentioned_pairs = [pair for pair in candidate_pairs if pair[0] and pair[0] in answer]
    if not mentioned_pairs:
        mentioned_pairs = [pair for pair in candidate_pairs if pair[1] and pair[1].lower() in answer_lower]
    final_pairs = mentioned_pairs if mentioned_pairs else candidate_pairs

    seen_courses = set()
    deduped = []
    for course, prof in final_pairs:
        if course not in seen_courses:
            seen_courses.add(course)
            deduped.append((course, prof))

    return [f"{course} ({prof})" if prof else course for course, prof in deduped], ''

def used_general_knowledge(result): return any(isinstance(m, ToolMessage) and m.name == GENERAL_TOOL_NAME for m in result['messages'])
def used_web_search(result): return any(isinstance(m, ToolMessage) and m.name == WEB_SEARCH_TOOL_NAME for m in result['messages'])
def used_career_tool(result): return any(isinstance(m, ToolMessage) and m.name == CAREER_TOOL_NAME for m in result['messages'])
def used_professor_courses_tool(result): return any(isinstance(m, ToolMessage) and m.name == PROFESSOR_TOOL_NAME for m in result['messages'])
def used_any_tool(result): return any(isinstance(m, ToolMessage) for m in result['messages'])

def has_any_internal_data(result):
    for message in result['messages']:
        if isinstance(message, ToolMessage) and message.name == CAREER_TOOL_NAME:
            if getattr(message, 'artifact', None):
                return True
    return False

def build_header(n_reviews, not_found, is_general, used_web, is_career, course_names, professor_name, question_is_english=False, is_count=False):
    if question_is_english:
        if is_general: return '[AI General Answer · No Internal Data]'
        if is_count: return '[Based on Internal Data · Review Count]'
        if is_career:
            if not_found: return '[Not Enough Internal Data]'
            base = '[Internal Data (Reviews + Syllabi)]'
            return '[Internal Data (Reviews + Syllabi) + External Info]' if used_web else base
        if n_reviews > 0 and not not_found:
            base = f'[Based on Internal Data · {n_reviews} review(s)'
            base += ' + External Info]' if used_web else ']' 
            if course_names:
                return f"{base}\n\n**Courses (Professor):** {', '.join(course_names)}"
            return base
        if used_web: return '[External Info (Web Search) · No Internal Data]'
        return '[Not Enough Reviews Available]'

    if is_general: return '[AI 일반 답변 · 내부 자료 없음]'
    if is_count: return '[내부 자료 기반 · 후기 집계]'
    if is_career:
        if not_found: return '[참고할 내부 자료가 부족합니다]'
        base = '[내부 자료(후기+강의계획서) 참고]'
        return '[내부 자료(후기+강의계획서) + 외부 정보 참고]' if used_web else base
    if n_reviews > 0 and not not_found:
        base = f'[내부 자료 기반 · {n_reviews}명의 후기 기반'
        base += ' + 외부 정보 참고]' if used_web else ']' 
        if course_names:
            return f"{base}\n\n**과목 (교수님):** {', '.join(course_names)}"
        return base
    if used_web: return '[외부 정보(웹검색) 기반 · 내부 자료 없음]'
    return '[참고할 후기가 부족합니다]'

# JSON 파싱용 항목 매핑
import json as _json_module, re as _re_module

LABEL_MAP_KO = {
    '과목명': '과목명',

    # 진로 추천용
    '추천이유': '추천 이유',
    '얻을수있는역량': '얻을 수 있는 역량',
    '수업방식': '수업 방식',
    '평가방식': '평가 방식',
    '주의할점': '주의할 점',

    # 일반 후기용
    '개인과제': '개인 과제',
    '팀프로젝트': '팀 프로젝트',
    '수업분위기': '수업 분위기',
    '꿀팁추천대상': '꿀팁/추천대상',
    '시험': '시험',
    '교수스타일': '교수 스타일',
    '기타': '기타',
}

LABEL_MAP_EN = {
    '과목명': 'Course',

    # career recommendation
    '추천이유': 'Why this course fits',
    '얻을수있는역량': 'Skills you can build',
    '수업방식': 'Class format',
    '평가방식': 'Grading / assessment',
    '주의할점': 'Things to keep in mind',

    # general review
    '개인과제': 'Individual assignments',
    '팀프로젝트': 'Team project',
    '수업분위기': 'Class atmosphere',
    '꿀팁추천대상': 'Tips / recommended for',
    '시험': 'Exams',
    '교수스타일': 'Professor style',
    '기타': 'Other notes',
}

# 기존 코드와의 호환용 별칭
LABEL_MAP = LABEL_MAP_KO

def format_single_course(obj: dict, english: bool = False) -> str:
    label_map = LABEL_MAP_EN if english else LABEL_MAP

    merged = {}

    for key, value in obj.items():
        if key == '과목명':
            continue
        if value is None:
            continue

        label = label_map.get(key, key)

        if isinstance(value, list):
            value = ' '.join(str(v).strip() for v in value if str(v).strip())
        else:
            value = str(value).strip()

        if not value:
            continue

        if label in merged:
            merged[label] = merged[label].rstrip() + ' ' + value
        else:
            merged[label] = value

    lines = []
    for label, value in merged.items():
        lines.append(f'**{label}:** {value}')

    return '\n\n'.join(lines)


def ensure_multi_course_json_array(answer, course_names, result):
    """
    여러 과목이 검색됐는데 LLM이 단일 JSON 객체로 답한 경우,
    과목별 JSON 배열로 다시 작성하게 강제한다.
    """
    if len(course_names) <= 1:
        return answer

    try:
        clean = _re_module.sub(r'```json|```', '', str(answer)).strip()
        parsed = _json_module.loads(clean)

        # 이미 과목별 배열이면 그대로 사용
        if isinstance(parsed, list):
            return answer

        # 여러 과목인데 dict 하나면 잘못된 형식이므로 재작성 요청
        if isinstance(parsed, dict):
            rewrite_request = HumanMessage(
                "방금 답변은 여러 과목을 하나로 합쳐서 잘못 작성했어. "
                "검색 결과에 나온 과목이 2개 이상이면 반드시 과목별 JSON 배열로 나눠야 해. "
                "각 객체에는 반드시 \"과목명\" 키를 넣고, 각 과목의 후기 내용만 해당 객체에 작성해. "
                "서로 다른 과목의 내용을 섞지 마. "
                "사용자가 한국어로 질문했다면 값(value)은 한국어로, 영어로 질문했다면 값(value)은 영어로 작성해. "
                "코드블록 없이 순수 JSON 배열만 출력해.\n\n"
                f"대상 과목 목록:\n{course_names}\n\n"
                f"잘못된 이전 답변:\n{answer}"
            )
            rewrite_result = agent.invoke({'messages': result['messages'] + [rewrite_request]})
            return extract_text(rewrite_result)

    except Exception:
        pass

    return answer

def parse_json_answer(answer, english: bool = False):
    """LLM이 반환한 JSON을 파싱해서 포맷된 마크다운으로 변환한다."""
    try:
        clean = _re_module.sub(r'```json|```', '', answer).strip()
        parsed = _json_module.loads(clean)
        if isinstance(parsed, list):
            sections = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                course_name = item.get('과목명', '')
                content = format_single_course(item, english=english)
                if content:
                    if course_name:
                        sections.append(f'**[{course_name}]**\n\n{content}')
                    else:
                        sections.append(content)
            if sections:
                return '\n\n---\n\n'.join(sections)
        elif isinstance(parsed, dict):
            course_name = parsed.get('과목명', '')
            result = format_single_course(parsed, english=english)
            if result:
                if course_name:
                    return f'**[{course_name}]**\n\n{result}'
                return result
    except Exception:
        pass
    return answer

def flatten_general_answer(answer):
    """
    일반 답변이 JSON으로 오면 value만 자연스럽게 이어 붙이고,
    JSON이 아니면 원문을 그대로 반환한다.
    """
    try:
        clean = _re_module.sub(r'```json|```', '', str(answer)).strip()
        parsed = _json_module.loads(clean)

        if isinstance(parsed, dict):
            parts = []
            for v in parsed.values():
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            if parts:
                return ' '.join(parts)

        return answer

    except Exception:
        return answer# 3. 여러 과목

def ask(question):
    global _exam_questions
    result = agent.invoke({'messages': [HumanMessage(question)]})
    answer = extract_text(result)

    if not used_any_tool(result):
        return {'question': question, 'answer': answer, 'n_reviews': 0, 'result': result, 'header': ''}

    n_reviews = extract_n_reviews(result)
    is_general = used_general_knowledge(result)
    used_web = used_web_search(result)
    is_career = used_career_tool(result)
    is_professor_courses = used_professor_courses_tool(result)
    question_is_english = is_english(question)
    is_comparison = any(
    isinstance(m, ToolMessage) and m.name == 'course_comparison_tool'
    for m in result['messages']
)
    is_count = any(isinstance(m, ToolMessage) and m.name == 'count_reviews_tool' for m in result['messages'])

    if is_professor_courses or is_count:
        not_found = False
    elif is_career:
        not_found = not has_any_internal_data(result)
    else:
        not_found = any(phrase in answer for phrase in [
            '확인해 주시겠어요', '확인 부탁드립니다', '아직 이 과목에 대한 후기가 없', '후기가 없어요'
        ])

    course_names, professor_name = ([], '') if (not_found or is_general or is_career or is_count) else get_header_info(result, answer)
    header = build_header(n_reviews, not_found, is_general, used_web, is_career, course_names, professor_name, question_is_english, is_count)

    # 1. 영어 질문 + 한국어 답변 → 영어로 번역 (JSON 파싱 전에)
    if question_is_english and not is_english(answer):
        translate_request = HumanMessage(
            f"Please rewrite your previous answer entirely in natural, fluent English. "
            f"Keep the same structure, items, and information. Do not add or remove any content:\n\n{answer}"
        )
        translate_result = agent.invoke({'messages': result['messages'] + [translate_request]})
        answer = extract_text(translate_result)

    # 2. 한국어 질문 + 영어 답변 → 한국어로 번역 (JSON 파싱 전에)
    if not question_is_english and is_english(answer):
        translate_request = HumanMessage(
            f"아래 JSON을 그대로 유지하되, 값(value)만 자연스러운 한국어 ~해요 말투로 번역해줘. "
            f"키(key)는 절대 바꾸지 마. JSON 형식 그대로 출력해:\n\n{answer}"
        )
        translate_result = agent.invoke({'messages': result['messages'] + [translate_request]})
        answer = extract_text(translate_result)

    
    # 3. 비교 질문이면 JSON 포맷으로 보내지 말고, Markdown 비교표로 재정리
    if is_comparison:
        target_language = "English" if question_is_english else "Korean"

        comparison_rewrite_prompt = HumanMessage(
            f"""
    You are rewriting a course comparison answer for an MBA course registration chatbot.

    The user's question was:
{question}

    The draft answer was:
{answer}

    Rewrite the answer as a clear Markdown comparison table.

    Rules:
    - Write in {target_language}.
    - Do not repeat the same content twice.
    - Do not add information that is not in the draft answer.
    - Use the exact course names that appear in the draft.
    - Make it useful for course registration decisions.
    - If information is missing, write that it is not clearly found in the available reviews.
    - Do not use JSON.
    - Do not use code blocks.

    Required structure:

    ## 과목 비교 / Course Comparison: Course A vs Course B

    ### 1. 한눈에 비교 / Quick Comparison

    | 항목 / Category | Course A | Course B |
    |---|---|---|
    | 수업방식 / Class format | ... | ... |
    | 과제·팀플 부담 / Assignments & team projects | ... | ... |
    | 시험 부담 / Exam difficulty | ... | ... |
    | 수업 분위기 / Class atmosphere | ... | ... |
    | 실무성 / Practical value | ... | ... |
    | 주의할 점 / Things to keep in mind | ... | ... |

    ### 2. 과목별 핵심 요약 / Course-by-course summary

    #### Course A
    ...

    #### Course B
    ...

    ### 3. 최종 추천 / Final recommendation
    ...
    """
    )

        rewrite_response = model.invoke([comparison_rewrite_prompt])
        answer = extract_message_text(rewrite_response)

    # 4. 일반 과목/교수님 후기 질문이면 기존 JSON 파싱/포맷팅
    elif not (is_general or is_count or not_found):
        answer = ensure_multi_course_json_array(answer, course_names, result)
        answer = parse_json_answer(answer, english=question_is_english)

    # 3-1. 일반 답변이 JSON으로 오면 {} 없이 value만 이어 붙임
    if is_general:
        answer = flatten_general_answer(answer)

    # 기출문제 원문 있으면 LLM 답변 뒤에 직접 붙임
    if _exam_questions:
        answer += f'\n\n**시험기출문제:**\n{_exam_questions}'
        _exam_questions = None

    return {'question': question, 'answer': answer, 'n_reviews': n_reviews, 'result': result, 'header': header}


# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('## ☕ 라떼는 말이야')
    st.markdown('SNU MBA 후배를 위한 AI 선배 챗봇')
    st.divider()
    st.markdown('**이런 질문을 해봐요:**')
    for ex in [
        '박성호 교수님 Marketing Analytics 어때?',
        '이제호 교수님 Strategy 어때?',
        '팀플 적은 수업 추천해줘',
        'KEIT R&D 기획 직무에 도움 되는 수업은?',
        '후기 많은 과목 보여줘',
    ]:
        if st.button(ex, key=ex):
            st.session_state['suggested'] = ex

    st.divider()
    if st.button('🗑️ 대화 초기화'):
        st.session_state.update({'messages':[], 'conversation_history':[], 'is_first_turn':True})
        st.rerun()
    st.markdown('---')
    st.markdown('채팅창에 `수강후기 남길게요` 입력!')


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
st.title('☕ 라떼는 말이야')
st.markdown('**SNU MBA 선배들의 솔직한 수강후기 — AI가 정리해드려요**')
st.markdown('※ 과목명은 한국어/영어 그대로 입력해주시면 더 정확해요 | Please use the exact course name for best results')

for key, default in [('messages',[]),('conversation_history',[]),('is_first_turn',True)]:
    if key not in st.session_state:
        st.session_state[key] = default

try:
    resources = load_all()
except Exception as e:
    st.error(f'데이터 로드 실패: {e}')
    st.stop()

for msg in st.session_state['messages']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

prompt = st.session_state.pop('suggested', None)

if user_input := (prompt or st.chat_input('궁금한 걸 입력하세요 / Ask anything')):
    with st.chat_message('user'):
        st.markdown(user_input)
    st.session_state['messages'].append({'role':'user','content':user_input})

    with st.chat_message('assistant'):
        with st.spinner('☕ 선배들의 후기를 찾아보는 중...'):
            try:
                from langchain.messages import HumanMessage
                # conversation_history 관리
                history = st.session_state['conversation_history']
                turn_start = len(history)
                history.append(HumanMessage(user_input))

                # agent 직접 호출
                result = resources['agent'].invoke({'messages': history})
                history = result['messages']
                st.session_state['conversation_history'] = history

                new_messages = history[turn_start:]
                turn_result = {'messages': new_messages}

                answer = extract_text(result)
                question_is_english = is_english(user_input)

                is_general = used_general_knowledge(turn_result)
                used_web = used_web_search(turn_result)
                is_career = used_career_tool(turn_result)
                is_professor_courses = used_professor_courses_tool(turn_result)
                is_count = any(isinstance(m, ToolMessage) and m.name == 'count_reviews_tool' for m in new_messages)
                is_comparison = any(isinstance(m, ToolMessage) and m.name == 'course_comparison_tool' for m in new_messages)
                n_reviews = extract_n_reviews(turn_result)

                if is_professor_courses or is_count:
                    not_found = False
                elif is_career:
                    not_found = not has_any_internal_data(turn_result)
                else:
                    not_found = any(phrase in answer for phrase in [
                        '확인해 주시겠어요', '확인 부탁드립니다', '아직 이 과목에 대한 후기가 없', '후기가 없어요'
                    ])

                course_names, professor_name = ([], '') if (not_found or is_general or is_career or is_count) else get_header_info(turn_result, answer)
                header = build_header(n_reviews, not_found, is_general, used_web, is_career, course_names, professor_name, question_is_english, is_count)

                # 한국어 질문 + 영어 답변 → 한국어로 번역
                if not question_is_english and is_english(answer):
                    translate_request = HumanMessage(
                        f"아래 JSON을 그대로 유지하되, 값(value)만 자연스러운 한국어 ~해요 말투로 번역해줘. "
                        f"키(key)는 절대 바꾸지 마. JSON 형식 그대로 출력해:\n\n{answer}"
                    )
                    translate_result = resources['agent'].invoke({'messages': history + [translate_request]})
                    answer = extract_text(translate_result)

                # JSON 파싱 + 포맷팅
                if not (is_general or is_count or not_found or is_comparison):
                    answer = ensure_multi_course_json_array(answer, course_names, result)
                    answer = parse_json_answer(answer, english=question_is_english)

                if is_general:
                    answer = flatten_general_answer(answer)

                # 헤더 붙이기
                if header:
                    full_answer = f"{header}\n\n{answer}"
                else:
                    full_answer = answer

                st.markdown(full_answer)
                st.session_state['messages'].append({'role':'assistant','content':full_answer})
            except Exception as e:
                st.error(f'오류: {e}')

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('📝 총 후기 수', f"{sum(1 for d in resources['all_docs'] if d.metadata.get('type')=='review')}건")
with col2:
    st.metric('📚 과목 수', f"{len({d.metadata.get('course') for d in resources['all_docs'] if d.metadata.get('type')=='review'})}개")
with col3:
    st.metric('☁️ Google Sheets', '✅ 연결됨' if resources['review_sheet'] else '⚠️ 미연결')
