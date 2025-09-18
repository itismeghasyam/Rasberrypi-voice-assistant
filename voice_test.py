#!/usr/bin/env python3
import os, shlex, subprocess, json, time
from pathlib import Path

import sounddevice as sd
import wave
import requests

PROJECT_DIR = Path.cwd()   # use current working dir
RECORDED_WAV = PROJECT_DIR / "recorded.wav"
SAMPLE_RATE = 16000
WHISPER_EXE_PATH = str(Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli")
WHISPER_MODEL = str(Path.home() / "whisper.cpp" / "models" / "ggml-tiny.bin")
OLLAMA_URL = "http://192.168.0.102:11434/api/generate"  # change if needed
OLLAMA_MODEL = "mistral"
DURATION = 6
_tts_engine = None
def get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        try:
            import pyttsx3
            _tts_engine = pyttsx3.init()
            _tts_engine.setProperty('rate', 150)
        except Exception:
            _tts_engine = None
    return _tts_engine

def speak_text(text: str):
    text = (text or "").strip()
    if not text:
        return
    print("[TTS] Speak:", text)
    try:
        subprocess.run(["espeak", text], check=True)
    except Exception as e:
        print("[TTS] espeak failed:", e)

def record_wav(path=RECORDED_WAV, duration=DURATION, s_r=SAMPLE_RATE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[REC] Recording {duration}s at {s_r}Hz -> {path}")
    audio = sd.rec(int(duration * s_r), samplerate=s_r, channels=1, dtype='int16')
    sd.wait()
    data = audio.flatten().tobytes()
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(s_r)
        wf.writeframes(data)
    print(f"[REC] Saved: {path}")
    return str(path)


import tempfile
from pathlib import Path

def transcribe_audio(wav_path):
    """
    Run whisper.cpp on wav_path and return the transcription text.
    Tries to use --no-prints + --output-txt and handles both
    possible output filenames: recorded.txt and recorded.wav.txt.
    Falls back to heuristic parsing if needed.
    """
    from pathlib import Path
    import shlex, subprocess, re

    wav_path = Path(wav_path)
    exe = Path(WHISPER_EXE_PATH)
    model = Path(WHISPER_MODEL)

    if not exe.exists():
        print("[STT] Whisper binary not found:", exe)
        return ""
    if not model.exists():
        print("[STT] Whisper model not found:", model)
        return ""

    cmd = [str(exe), "-m", str(model), "-f", str(wav_path), "--no-prints", "--output-txt"]
    print("[STT] Running:", " ".join(shlex.quote(c) for c in cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("[STT] Transcription timed out")
        return ""

    cand1 = wav_path.with_suffix(".txt")           # recorded.txt
    cand2 = Path(str(wav_path) + ".txt")           # recorded.wav.txt
    for p in (cand1, cand2):
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8").strip()
                if txt:
                    print(f"[STT] (from file) Transcribed (from {p.name}):", txt)
                    return txt
            except Exception as e:
                print(f"[STT] could not read {p}:", e)

    raw = (proc.stdout or "") + "\n" + (proc.stderr or "")
    raw = raw.strip()
    if not raw:
        return ""

    print("[STT] raw output snippet:", raw[:1000].replace("\n", " ") + ("..." if len(raw)>1000 else ""))

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    filtered = []
    for line in lines:
        low = line.lower()
        if "-->" in line and "[" in line:
            continue
        if low.startswith("whisper_") or low.startswith("whisper") and "transcrib" not in low:
            continue
        if "output_txt" in low or "saving output" in low or "total time" in low or "load time" in low:
            continue
        if all(c.isdigit() or c.isspace() or c in ".:-'\"" for c in line):
            continue
        filtered.append(line)

    if filtered:
        candidate = filtered[-1]
        if "]" in candidate:
            candidate = candidate.split("]")[-1].strip()
        print("[STT] (heuristic) Transcribed:", candidate)
        return candidate

    m = re.search(r"([A-Za-z0-9][A-Za-z0-9 ,\?\.\!'\-]{1,})", raw)
    if m:
        txt = m.group(1).strip()
        print("[STT] (fallback regex) Transcribed:", txt)
        return txt

    return ""


def generate_response_ollama(user_text, timeout=30):
    # strong system message to prevent persona hallucination
    system_prefix = (
        "SYSTEM: You are a concise, factual assistant. "
        "Do NOT invent personal biography, names, ages, or experiences. "
        "If asked about yourself, reply: 'I am an AI assistant running on a server.' "
        "Answer in 1-2 short sentences.\n\n"
    )
    prompt = f"{system_prefix}User: {user_text}\nAssistant:"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 48,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print("[LLM] Ollama error:", e)
        return "Sorry, I couldn't reach the model."
    if isinstance(data, dict):
        for key in ("response","text","completion"):
            if key in data and isinstance(data[key], str):
                return data[key].strip()
    return json.dumps(data)[:1000]

def main():
    wav = record_wav()
    # quick playback check - comment out if not needed
    # subprocess.run(["aplay", wav])
    trans = transcribe_audio(wav)
    if not trans:
        print("[MAIN] Nothing transcribed.")
        speak_text("Sorry, I did not hear anything.")
        return
    # print exact transcription for debugging
    print("[MAIN] Final transcription to send to model:", repr(trans))
    resp = generate_response_ollama(trans)
    print("[MAIN] Model replied:", resp)
    speak_text(resp)

if __name__ == "__main__":
    main()
