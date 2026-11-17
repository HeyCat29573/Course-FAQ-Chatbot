from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from rag import answer
import voice

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []   # [{"role": "user"/"assistant", "content": "..."}]

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        reply, sources, method = answer(req.message, req.history)
        return {"reply": reply, "sources": sources, "method": method}
    except Exception as e:
        return {"reply": f"Sorry, something went wrong: {e}", "sources": [], "method": ""}

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def tts(req: TTSRequest):
    try:
        audio = await voice.synthesize(req.text)
        return Response(content=audio, media_type="audio/mpeg")
    except Exception as e:
        return Response(content=str(e), status_code=500)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")