import re
import edge_tts

EN_VOICE = "en-US-AriaNeural"
ZH_VOICE = "zh-HK-HiuMaanNeural"   # Cantonese; use "zh-CN-XiaoxiaoNeural" for Mandarin

def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))

def clean_for_speech(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)       # links
    text = re.sub(r"[*#`_>]+", "", text)           # markdown symbols
    return re.sub(r"\s+", " ", text).strip()

async def synthesize(text: str) -> bytes:
    voice = ZH_VOICE if has_chinese(text) else EN_VOICE
    comm = edge_tts.Communicate(clean_for_speech(text), voice)
    audio = b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return audio