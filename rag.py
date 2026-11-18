import os, re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from rapidfuzz import fuzz

load_dotenv()


# ---- 1. Load data and build vector DB (runs once at startup) ----
df = pd.read_csv("data/courses.csv").fillna("None")
df["code"] = df["code"].str.upper().str.strip()

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

# Words that signal a follow-up about the course discussed earlier
FOLLOWUP_PATTERN = re.compile(
    r"\b(it|its|this|that|they|them|their|those|these|the course|the same one)\b|它|這|这|該|该|呢",
    re.IGNORECASE,
)

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
            if ids:
                return ids[:3]
    return []

def retrieve(question: str, history: list[dict] | None = None, k: int = 3):
    """Returns (items, method). Each item: {"code", "name", "text"}."""
    codes = [c.replace(" ", "").upper() for c in CODE_PATTERN.findall(question)]
    valid = [c for c in codes if c in set(df["code"])]

    if valid:                                   # 1. exact course code
        r = collection.get(ids=valid)
        return _items(r["ids"], r["documents"]), "Exact course code"
    if codes:                                   # code asked for, but not in the data
        return [], "No matching course code"

    matched = find_courses(question)            # 2. acronym / fuzzy name
    if matched:
        r = collection.get(ids=matched)
        return _items(r["ids"], r["documents"]), "Acronym / fuzzy name match"

    if history and FOLLOWUP_PATTERN.search(question):   # 3. follow-up ("How many credits is it?")
        ids = codes_from_history(history)
        if ids:
            r = collection.get(ids=ids)
            return _items(r["ids"], r["documents"]), "Previous conversation (follow-up)"

    r = collection.query(query_texts=[question], n_results=k)   # 4. semantic search
    return _items(r["ids"][0], r["documents"][0]), "Semantic search"

# ---- 3. LLM via OpenRouter (OpenAI-compatible API) ----
llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

SYSTEM_PROMPT = """You are a friendly course FAQ assistant for university courses.
Rules:
1. Answer ONLY using the course information in the provided context.
2. If the answer is not in the context, say you don't have that information
   and suggest checking the official course page "https://www.polyu.edu.hk/internalsearch".
3. LANGUAGE: Reply in English by default. Only reply in Chinese if the user's
   question is written in Chinese. Never use any other language.
4. Be concise.
5. FORMAT: Use plain text. Short "-" bullet lists and **bold** are fine.
   Do not use tables, headings or code blocks."""

def answer(question: str, history: list[dict]):
    items, method = retrieve(question, history)
    if items:
        context = "\n".join(i["text"] for i in items)
    else:
        context = "The course code the user asked about does not exist in the database."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]
    messages.append({
        "role": "user",
        "content": (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "(Reply in English unless the question above is written in Chinese.)"
        ),
    })
    resp = llm.chat.completions.create(model=MODEL, messages=messages, temperature=0.2)
    sources = [{"code": i["code"], "name": i["name"]} for i in items]
    return resp.choices[0].message.content, sources, method