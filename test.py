
import os
import shlex
import subprocess
import json
import time
from pathlib import Path

import sounddevice as sd
import wave
import requests
from pathlib import Path
import shlex, subprocess, re

PROJECT_DIR = Path.cwd()   
RECORDED_WAV = PROJECT_DIR / "recorded.wav"
SAMPLE_RATE = 16000
DURATION = 6

# whisper 
WHISPER_EXE_PATH = str(Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli")
WHISPER_MODEL = str(Path.home() / "whisper.cpp" / "models" / "ggml-tiny.bin")

# Local LLM
LLAMA_CLI = str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli")
LOCAL_MODEL = str(Path.home() / "models"/ "gpt2.Q3_K_M.gguf")

# Ollama 
OLLAMA_URL = "http://192.168.0.102:11434/api/generate"
OLLAMA_MODEL = "mistral"


_tts_engine = None

def speak_text(text: str):
    text = (text or "").strip()
    if not text:
        return
    print("[TTS] Speak:", text)
    # Direct espeak call (robust for Pi + bluetooth)
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

def transcribe_audio(wav_path):

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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=800)
    except subprocess.TimeoutExpired:
        print("[STT] Transcription timed out")
        return ""

    # whisper.cpp may write either "recorded.txt" or "recorded.wav.txt"
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
        if low.startswith("whisper_") or (low.startswith("whisper") and "transcrib" not in low):
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

def generate_response_local_llama(LOCAL_MODEL,prompt_text, n_predict=16, threads=4, temperature=0.1):
    exe = Path(LLAMA_CLI)
    model = Path(LOCAL_MODEL)

    if not exe.exists() or not model.exists():
        missing = []
        if not exe.exists():
            missing.append("llama-cli")
        if not model.exists():
            missing.append("local model")
        print("[LLM] Local llama not available (missing {}). Falling back to Ollama.".format(", ".join(missing)))
        return None, None, None

    prompt_escaped = prompt_text
    cmd = [
        str(exe),
        "-m", str(model),
        "-p", prompt_escaped,
        "-n", str(n_predict),
        "-t", str(threads),
        "--temp", str(temperature),
    ]

   
    print("[LLM] Running local llama:", " ".join(shlex.quote(c) for c in cmd))
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=800 + n_predict * 3)
    except subprocess.TimeoutExpired:
        print("[LLM] Local llama generation timed out")
        return None, None, None
    elapsed = time.time() - start

    out = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    if not out:
        return "", elapsed, 0

    
    generated = out
    if prompt_text.strip() and prompt_text.strip() in out:
        # split by the prompt and take the remainder
        parts = out.split(prompt_text.strip())
        if len(parts) > 1:
            generated = parts[-1].strip()

    
    lines = [l for l in generated.splitlines() if l.strip()]
    if lines:
        generated = "\n".join(lines).strip()

    # approximate tokens by whitespace-splitting (rough)
    tokens = len(generated.split())
    tps = tokens / elapsed if elapsed > 0 else 0.0

    print(f"[LLM] Local generation finished in {elapsed:.2f}s, approx tokens={tokens}, TPS={tps:.2f}")
    return generated, elapsed, tokens

def generate_response_ollama(user_text, timeout=30):
    # fallback method if local llama isn't available
    system_prefix = (
        "SYSTEM: You are a concise, factual assistant. "
        "Do NOT invent personal biography, names, ages, or experiences. "
        "Answer in 1-2 short sentences.\n\n"
    )
    prompt = f"{system_prefix}User: {user_text}\nAssistant:"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "temperature": 0.2,
        "top_p": 0.2,
        "max_tokens": 64,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print("[LLM] Ollama error:", e)
        return "Sorry, I couldn't reach the model"
    if isinstance(data, dict):
        for key in ("response", "text", "completion"):
            if key in data and isinstance(data[key], str):
                return data[key].strip()
    return json.dumps(data)[:1000]

def main():
    
    text = "What is one plus one"
    
    generated, elapsed, tokens = generate_response_local_llama(LOCAL_MODEL=str(Path.home() / "models"/ "gpt2.Q3_K_M.gguf"), prompt_text=text, n_predict=16, threads=2, temperature=0.8)
    print("[MAIN] Model replied (local):", generated)
    if elapsed is not None:
        approx_tps = (tokens / elapsed) if elapsed and tokens else 0.0
        print(f"[BENCH] elapsed={elapsed:.3f}s tokens={tokens} approx_TPS={approx_tps:.2f}")
    
    generated, elapsed, tokens = generate_response_local_llama(LOCAL_MODEL=str(Path.home() / "models"/ "gpt2.Q4_K_M.gguf"), prompt_text=text, n_predict=16, threads=2, temperature=0.8)
    print("[MAIN] Model replied (local):", generated)
    if elapsed is not None:
        approx_tps = (tokens / elapsed) if elapsed and tokens else 0.0
        print(f"[BENCH] elapsed={elapsed:.3f}s tokens={tokens} approx_TPS={approx_tps:.2f}")

    generated, elapsed, tokens = generate_response_local_llama(LOCAL_MODEL=str(Path.home() / "models"/ "gpt2-medium-Q4_K_M.gguf"), prompt_text=text, n_predict=16, threads=2, temperature=0.8)
    print("[MAIN] Model replied (local):", generated)
    if elapsed is not None:
        approx_tps = (tokens / elapsed) if elapsed and tokens else 0.0
        print(f"[BENCH] elapsed={elapsed:.3f}s tokens={tokens} approx_TPS={approx_tps:.2f}")

if __name__ == "__main__":
    main()
