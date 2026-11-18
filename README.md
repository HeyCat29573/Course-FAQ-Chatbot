# Course FAQ Chatbot

COMP4431 Artificial Intelligence (2026/27) — Group 11, The Hong Kong Polytechnic University, November 2026, Project: *Domain-specific Chatbot Development*

A chatbot that answers questions about university courses (course code, name, description, credits, prerequisites, assessment). It uses Retrieval-Augmented Generation (RAG): relevant course records are retrieved first, then an LLM (DeepSeek via OpenRouter) writes an answer grounded only in those records.

## Features

### Basic features

- Answers questions about courses in the defined domain, in English or Chinese.
- Looks up a course by code (e.g. "What is COMP4001?") and answers from the course data only.
- Handles follow-up questions using the recent chat history (e.g. "What are its prerequisites?").
- Says so when a course code is not in the database instead of guessing.

### Additional features

1. **Hybrid retrieval pipeline**
   - Exact course-code match (regex), for example `COMP4001`.
   - Acronym and fuzzy name matching with `rapidfuzz`, for example "HCI" or a typo such as "internshp".
   - Semantic search with ChromaDB and a multilingual sentence-transformer model as the fallback, for example "which course teaches interface design?".
2. **Source display**: each answer shows which course(s) it was based on and which retrieval method found them.
3. **Read Aloud (text-to-speech)**: replies are spoken using neural voices generated on the backend with `edge-tts` (not the browser's Web Speech API). The English or Chinese voice is chosen automatically. Each reply has a replay button, and a mute toggle turns auto-read off.
4. **2D animated avatar**: an SVG avatar that blinks when idle, bobs while thinking, and opens its mouth in time with the speech. The lip-sync reads the loudness of the playing audio through the Web Audio API.
5. **User interface**: light/dark mode (remembers your choice), suggested-question bubbles on the start screen, and follow-up question bubbles after every answer.

## How It Works

```
User question
   │
   ▼
Retrieval:  exact course code ──► acronym / fuzzy name ──► semantic search (ChromaDB)
   │
   ▼
Prompt = system prompt + retrieved course records + recent history + question
   │
   ▼
DeepSeek (via OpenRouter)  ──►  answer + sources
   │
   ▼
Browser: chat UI + source tags  ──►  /tts (edge-tts)  ──►  audio + avatar lip-sync
```

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Retrieval:** ChromaDB, sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`), rapidfuzz, pandas
- **LLM:** DeepSeek through the OpenRouter API (OpenAI-compatible client)
- **Speech synthesis:** edge-tts
- **Frontend:** plain HTML, CSS and JavaScript (no framework)

## Project Structure

```
course-bot/
├── app.py              # FastAPI server: /chat, /tts and the web page
├── rag.py              # Retrieval (exact / fuzzy / semantic), system prompt, LLM call
├── voice.py            # Text-to-speech (edge-tts)
├── data/
│   └── courses.csv     # Knowledge base
├── static/
│   └── index.html      # Chat interface (avatar, dark mode, suggestions)
├── requirements.txt
├── .env.example        # Template for your API key
└── README.md
```

## Setup and Run

### 1. Requirements

- Python 3.12 or newer
- An OpenRouter API key (https://openrouter.ai)
- An internet connection (for the LLM and for the text-to-speech voices)

### 2. Install

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure the API key

Copy `.env.example` to `.env` and fill in your own key:

```
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_MODEL=deepseek/deepseek-chat
```

The model name must match a DeepSeek model listed on OpenRouter. The `.env` file is not included in the repository, so each person needs their own key.


### 4. Start the chatbot

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 in your browser. The first start is slow because the embedding model is downloaded once.

## Example Questions

| Question | What it shows |
|---|---|
| `What is COMP4001?` | Exact course-code lookup |
| `What is HCI?` | Acronym matching |
| `tell me about internshp` | Fuzzy matching of a typo |
| `Which course teaches interface design?` | Semantic search |
| `What are its prerequisites?` | Follow-up using chat history |
| `What is COMP9999?` | Course not in the database |
| A question in Chinese | Chinese answer and Chinese voice |

## Prompt and Knowledge Base Files

- **System prompt:** `SYSTEM_PROMPT` in `rag.py`
- **Knowledge base:** `data/courses.csv`

## Technical Challenges and Solutions

| Problem | Fix |
|---|---|
| "HCI" was not recognised as *Human-Computer Interaction* | Added auto-generated acronyms, an alias table and `rapidfuzz` name matching before semantic search |
| `Expected IDs to be a non-empty list` when a course code is not in the data | Check codes against the known list and return "no matching course code", so the bot says it has no data |
| Answers came back in Spanish | Explicit language rule in the system prompt, repeated next to the question |
| Raw `**` Markdown symbols shown in replies | Safe Markdown renderer in the page (bold, lists, links, tables) plus a format rule in the prompt |
| "How many credits is it?" returned other courses | Retrieval now uses the chat history to find the course discussed earlier |
| Voice read out Markdown and table symbols | Strip these symbols before text-to-speech |

## Limitations

- The chatbot only knows the courses in `data/courses.csv`. Anything outside it gets a "not available" reply.
- Follow-up detection is rule-based (pronouns such as "it" or 他, and attribute words such as "credits" or 學分). A follow-up with neither, for example "and then?", is treated as a new question.
- Read Aloud needs an internet connection (edge-tts).
- Answer quality depends on how complete the course descriptions in the CSV are.
- There is no speech input; the chatbot is used by typing.

## Group Members

| Student ID | Name |
|---|---|
| 26033735d | Chow Ho Nam   |
| 26015586d | To Yee Chun   |
| 26033949d | Cheng Him Ho  |
| 26015945d | Kwok Long Hei |

## Licence and acknowledgments

Coursework for COMP4431 at The Hong Kong Polytechnic University. Only for learning use.

- Thanks to **Claude (Anthropic)** for serving as an advanced AI pair programmer, providing architectural guidance, and technical explanations throughout the development of this repository.
