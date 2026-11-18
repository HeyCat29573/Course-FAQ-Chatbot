import os, re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from rapidfuzz import fuzz, process

load_dotenv()


# ---- 1. Load data and build vector DB (runs once at startup) ----
df = pd.read_csv("data/courses.csv").fillna("None")
df["code"] = df["code"].str.upper().str.strip()

def format_credits(v) -> str:
    v = str(v).strip()
    if re.fullmatch(r"\d+(\.0+)?", v):          # "3", "3.0" -> "3 credits"
        n = int(float(v))
        return f"{n} credit" + ("" if n == 1 else "s")
    return v                                     # "12 Training Credits" stays as it is

df["credits"] = df["credits"].map(format_credits)

# ---- Acronyms and aliases ----
STOPWORDS = {"and", "of", "the", "for", "to", "in", "a"}

def make_acronym(name: str) -> str:
    words = [w for w in re.split(r"[^A-Za-z0-9]+", name) if w and w.lower() not in STOPWORDS]
    return "".join(w[0] for w in words).upper() if len(words) >= 2 else ""

# Manual aliases (alias -> course codes). Edit to match your own courses.
ALIASES = {
    "AI": ["COMP4431"],
    "HCI": ["COMP3423"],
    "WIE": ["COMP4001"],
}

alias_map = {}
for _, r in df.iterrows():
    ac = make_acronym(r["name"])
    if len(ac) >= 3:  # skip 2-letter ones: "IS", "IT" would match normal English words
        alias_map.setdefault(ac, []).append(r.code)
for alias, codes in ALIASES.items():
    for c in codes:
        if c in set(df["code"]):
            alias_map.setdefault(alias.upper(), [])
            if c not in alias_map[alias.upper()]:
                alias_map[alias.upper()].append(c)

def find_courses(question: str, fuzzy_cutoff: int = 85) -> list[str]:
    # 1. Acronym / alias match
    tokens = set(re.findall(r"[A-Za-z0-9]+", question.upper()))
    hits = []
    for tok in tokens:
        for code in alias_map.get(tok, []):
            if code not in hits:
                hits.append(code)
    if hits:
        return hits

    # 2. Fuzzy match on course names (handles typos / partial names)
    q = question.lower()
    scored = []
    for _, r in df.iterrows():
        name = r["name"].lower()
        if len(name) < 6:
            continue
        score = fuzz.partial_ratio(name, q)
        if score >= fuzzy_cutoff:
            scored.append((score, r.code))
    scored.sort(reverse=True)
    return [code for _, code in scored[:3]]

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"  # supports English + Chinese
)
client_db = chromadb.Client()
collection = client_db.get_or_create_collection("courses", embedding_function=ef)

if collection.count() == 0:
    collection.add(
        ids=df["code"].tolist(),
        documents=[
            f"Course code: {r.code}. Name: {r['name']} (also known as {make_acronym(r['name'])}). "
            f"Credits: {r.credits}. Prerequisites: {r.prerequisites}."
            f"Level: {r.level}. Assessment: {r.assessment}"
            for _, r in df.iterrows()
        ],
    )

# ---- 2. Retrieval: exact code match first, then semantic search ----
CODE_PATTERN = re.compile(r"[A-Za-z]{4}\s?\d{4}")

name_of = dict(zip(df["code"], df["name"]))

def _items(ids, docs):
    return [{"code": i, "name": name_of.get(i, ""), "text": d} for i, d in zip(ids, docs)]

# Pronouns that point back to the course discussed earlier (English + Chinese)
PRONOUNS = re.compile(
    r"\b(it|its|this|that|they|them|their|those|these|the course|the same one)\b"
    r"|它|牠|她|(?<!其)他|佢|這|这|此|該|该|嗰個|那個|那个|呢個|呢門|呢科|呢堂",
    re.IGNORECASE,
)
# Questions about a course attribute, even with no pronoun ("And the credits?", "多少學分?")
ATTRIBUTES = re.compile(
    r"credit|prerequisite|assess|level|description|outcome"
    r"|學分|学分|先修|先決|先决|評核|评核|考核|要求|簡介|简介|內容|内容|程度",
    re.IGNORECASE,
)
# Questions that search for courses, so they are NOT about the previous course
SEARCHING = re.compile(
    r"\b(which|what courses|any|all|list|recommend|show me)\b"
    r"|哪|咩|邊|什麼|什么|有冇|有沒有|有没有|推薦|推荐|所有",
    re.IGNORECASE,
)

def is_followup(question: str) -> bool:
    if PRONOUNS.search(question):
        return True
    return (len(question) <= 40 and bool(ATTRIBUTES.search(question))
            and not SEARCHING.search(question))

def language_note(question: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", question):
        return "(Reply in Chinese, using the same script, Traditional or Simplified, as the question.)"
    return "(Reply in English.)"

def codes_from_history(history: list[dict], max_back: int = 8) -> list[str]:
    """Course code(s) most recently discussed in the conversation."""
    known = set(df["code"])
    recent = history[-max_back:]

    def explicit(text):
        found = [c.replace(" ", "").upper() for c in CODE_PATTERN.findall(text)]
        return [c for c in dict.fromkeys(found) if c in known]

    # 1. what the user themselves asked about (code, acronym or course name)
    for msg in reversed(recent):
        if msg["role"] == "user":
            ids = explicit(msg["content"]) or find_courses(msg["content"])
            if ids:
                return ids[:3]
    # 2. otherwise, courses the assistant mentioned in its last answers
    for msg in reversed(recent):
        if msg["role"] == "assistant":
            ids = explicit(msg["content"])
            if ids and len(ids) <= 5:
                return ids[:3]
    return []

def retrieve(question: str, history: list[dict] | None = None, k: int = 3):
    """Returns (items, method). Each item: {"code", "name", "text"}."""
    codes = [c.replace(" ", "").upper() for c in CODE_PATTERN.findall(question)]
    valid = [c for c in codes if c in set(df["code"])]

    if valid:                                   # 1. exact course code
        r = collection.get(ids=valid)
        return _items(r["ids"], r["documents"]), "Exact course code"
    if codes:                                   # code asked for, but not in the data:
        similar = process.extract(codes[0], list(name_of), scorer=fuzz.ratio, limit=3)
        ids = [code for code, _score, _idx in similar]      # ...offer the courses with the closest codes
        r = collection.get(ids=ids)
        return _items(r["ids"], r["documents"]), "No matching course code"

    matched = find_courses(question)            # 2. acronym / fuzzy name
    if matched:
        r = collection.get(ids=matched)
        return _items(r["ids"], r["documents"]), "Acronym / fuzzy name match"

    if is_followup(question):                   # 3. follow-up ("How many credits is it?")
        ids = codes_from_history(history or [])
        if ids:
            r = collection.get(ids=ids)
            return _items(r["ids"], r["documents"]), "Previous conversation (follow-up)"
        return [], "Unclear follow-up"          # "it" refers to nothing we know

    r = collection.query(query_texts=[question], n_results=k)   # 4. semantic search
    return _items(r["ids"][0], r["documents"][0]), "Semantic search"


def clarify_reply(question: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", question):
        return "請問你指的是哪一個課程？請告訴我課程編號或名稱，例如 COMP4001。"
    return "Which course do you mean? Please tell me a course code or name, for example COMP4001."

# ---- 3. LLM via OpenRouter (OpenAI-compatible API) ----
llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

SYSTEM_PROMPT = """You are a friendly course FAQ assistant for university courses.
Rules:
1. Answer ONLY using the course information in the provided context.
2. If the context does not contain the answer at all, reply with exactly: NO_INFO
   (nothing else). If it contains only part of the answer, answer that part and
   say which part is missing.
3. LANGUAGE: Reply in English by default. Only reply in Chinese if the user's
   question is written in Chinese. Never use any other language.
4. Be concise.
5. FORMAT: Use plain text. Short "-" bullet lists and **bold** are fine.
   Do not use tables, headings or code blocks."""

# "List all courses" requests (English + Chinese)
LIST_ALL = re.compile(
    r"\b(all|every|entire|full)\b[\w\s]{0,20}\bcourses?\b"
    r"|\bwhat\s+courses?\s+(do you have|are there|are available|can i ask)"
    r"|\bhow many courses\b"
    r"|\bcourse list\b|\blist of (the )?courses\b|\blist (the )?courses\b"
    r"|所有課程|所有课程|全部課程|全部课程|有咩課程|有哪些課程|有哪些课程|有什麼課程|有什么课程"
    r"|課程列表|课程列表|課程清單|课程清单|有幾多個?課程|有多少個?課程|有多少门课程",
    re.IGNORECASE,
)

def list_all_courses(question: str) -> str:
    """Build the full course list directly from the data (no LLM, so nothing is missed)."""
    zh = re.search(r"[\u4e00-\u9fff]", question)
    head = f"資料庫中共有 {len(df)} 個課程：" if zh else f"There are {len(df)} courses in the database:"
    rows = ["| 課程編號 | 課程名稱 | 學分 |" if zh else "| Code | Name | Credits |", "|---|---|---|"]
    for _, r in df.sort_values("code").iterrows():
        rows.append(f"| {r.code} | {r['name']} | {r.credits} |")
    tail = "你可以問我任何一個課程的詳情。" if zh else "Ask me about any of them for details."
    return head + "\n\n" + "\n".join(rows) + "\n\n" + tail

OFFICIAL_URL = "https://www.polyu.edu.hk/internalsearch"

# Detects a reply that starts by saying "I have no information" (backup for NO_INFO)
NO_INFO_START = re.compile(
    r"^\W*(sorry\W+|抱歉\W*)?(i\s+)?((don't|do not|cannot|can't|couldn't)\s+(have|find|provide)\b[^.]{0,40}information"
    r"|have no information)"
    r"|^\W*(抱歉\W*)?(我)?(沒有|没有).{0,8}(資料|资料|資訊|信息)",
    re.IGNORECASE,
)

def is_no_info(reply: str) -> bool:
    r = reply.strip()
    return r.upper().startswith("NO_INFO") or (len(r) <= 200 and bool(NO_INFO_START.search(r)))

def no_info_reply(question: str, items: list[dict], missing_code: str = "") -> str:
    """Fixed-format 'cannot answer' reply, so references and link are always shown."""
    zh = re.search(r"[\u4e00-\u9fff]", question)
    if missing_code:
        first = f"資料庫中沒有課程 {missing_code}。" if zh else f"I couldn't find course {missing_code} in the database."
    else:
        first = "我沒有這方面的資料。" if zh else "I don't have information about that."
    lines = [first]
    if items:
        lines.append("")
        lines.append("以下是我可以提供資料的部分課程：" if zh else "For reference, here are some courses I can provide information about:")
        lines += [f"- {i['code']} {i['name']}" for i in items]
    lines.append("")
    lines.append(f"你也可以查看官方課程網頁：{OFFICIAL_URL}" if zh else f"You can also check the official course page: {OFFICIAL_URL}")
    return "\n".join(lines)

def answer(question: str, history: list[dict]):
    if LIST_ALL.search(question):
        return list_all_courses(question), [], "Full course list"

    items, method = retrieve(question, history)
    sources = [{"code": i["code"], "name": i["name"]} for i in items]

    if method == "Unclear follow-up":
        return clarify_reply(question), [], method

    if method == "No matching course code":      # asked for a code that does not exist
        missing = CODE_PATTERN.findall(question)[0].replace(" ", "").upper()
        return no_info_reply(question, items, missing), sources, method

    context = "\n".join(i["text"] for i in items)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]
    messages.append({
        "role": "user",
        "content": (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"{language_note(question)}"
        ),
    })
    resp = llm.chat.completions.create(model=MODEL, messages=messages, temperature=0.2)
    reply = resp.choices[0].message.content.strip()
    if is_no_info(reply):
        reply = no_info_reply(question, items)
    return reply, sources, method