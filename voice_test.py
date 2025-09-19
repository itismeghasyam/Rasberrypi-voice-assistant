
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
LOCAL_MODEL = str(Path.home() / "models" / "gpt2.Q3_K_M.gguf")

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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
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
# Insert these functions into your script. Requires `psutil` (pip install psutil).

import time
import shlex
import subprocess
import threading
import re
import os
from pathlib import Path

try:
    import psutil
except Exception:
    psutil = None  # ResourceSampler will be a no-op if psutil isn't installed

# -------------------------
# Resource sampler (optional)
# -------------------------
class ResourceSampler:
    """
    Sample current process RSS and CPU% at `interval` seconds in background.
    Call start() before a heavy operation and stop() afterwards.
    summary() returns a dict: peak_rss, avg_rss, peak_cpu, avg_cpu, samples.
    If psutil is not available this becomes a minimal no-op sampler.
    """
    def __init__(self, interval=0.05):
        self.interval = float(interval)
        self._running = False
        self._thread = None
        self.samples = []  # [(ts, rss_bytes, cpu_percent), ...]
        if psutil:
            self._proc = psutil.Process(os.getpid())
        else:
            self._proc = None

    def _loop(self):
        # warm-up cpu_percent baseline
        try:
            if self._proc: self._proc.cpu_percent(interval=None)
        except Exception:
            pass
        while self._running:
            try:
                if self._proc:
                    rss = self._proc.memory_info().rss
                    cpu = self._proc.cpu_percent(interval=None)
                else:
                    rss = 0
                    cpu = 0.0
                self.samples.append((time.time(), rss, cpu))
            except Exception:
                # ignore sampling failures
                pass
            time.sleep(self.interval)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def summary(self):
        if not self.samples:
            return {"peak_rss": 0, "avg_rss": 0, "peak_cpu": 0.0, "avg_cpu": 0.0, "samples": 0}
        rss_vals = [s[1] for s in self.samples]
        cpu_vals = [s[2] for s in self.samples]
        return {
            "peak_rss": max(rss_vals),
            "avg_rss": sum(rss_vals) / len(rss_vals),
            "peak_cpu": max(cpu_vals),
            "avg_cpu": sum(cpu_vals) / len(cpu_vals),
            "samples": len(self.samples),
        }


# small TTS wrapper

def speak_text_timed(text: str, cmd=None):
    """
    Very small wrapper that runs a TTS command and returns elapsed seconds.
    Default uses `espeak text`. If you use a different TTS, pass `cmd` as a list,
    e.g. ['tts-cli', '--out', '/tmp/out.wav', text] or similar.
    """
    text = (text or "").strip()
    if not text:
        return 0.0
    if cmd is None:
        # default simple espeak call; replace if you need a different TTS
        cmd = ["espeak", text]
    else:
        # allow passing a format-string or a list; convert format-string to shell-safe list
        if isinstance(cmd, str):
            # user provided something like "mytts --text '{}'"
            safe = cmd.format(shlex.quote(text))
            cmd = shlex.split(safe)
        elif isinstance(cmd, (list, tuple)):
            # if list contains {} we substitute with text
            cmd = [c.format(text) if ("{}" in c or "{text}" in c) else c for c in cmd]
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        # swallow TTS errors but still return elapsed
        pass
    return time.time() - t0

# llama110 main function
def llama110(prompt_text: str,
             llama_cli_path: str = None,
             model_path: str = None,
             n_predict: int = 16,
             threads: int = 4,
             temperature: float = 0.2,
             sampler: ResourceSampler = None,
             tts_after: bool = False,
             tts_cmd = None,
             timeout_seconds: int = 240):
    

    # sensible defaults (do not overwrite your other model; use explicit path if you have one)
    if llama_cli_path is None:
        llama_cli_path = str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli")
    if model_path is None:
        model_path = str(Path.home() / "Downloads" / "llama2.c-stories110M-pruned50.Q3_K_M.gguf")

    exe = Path(llama_cli_path)
    model = Path(model_path)

    result = {
        "generated": "",
        "model_reply_time": 0.0,
        "model_inference_time": None,
        "tokens": 0,
        "tts_time": 0.0,
        "resource": {"peak_rss":0, "avg_rss":0, "peak_cpu":0.0, "avg_cpu":0.0, "samples":0},
        "raw_stdout": None,
        "raw_stderr": None,
    }

    if sampler is None:
        sampler = ResourceSampler(interval=0.05)  # local no-op if psutil missing

    if not exe.exists():
        raise FileNotFoundError(f"llama-cli binary not found at: {exe}")

    if not model.exists():
        raise FileNotFoundError(f"llama model not found at: {model}")

    # Build command (adjust flags to your llama-cli flavor if different)
    cmd = [
        str(exe),
        "-m", str(model),
        "-p", prompt_text,
        "-n", str(n_predict),
        "-t", str(threads),
        "--temp", str(temperature),
        # (keep top-k/top-p or other flags out so you can control externally)
    ]

    # start sampling and call subprocess
    sampler.start()
    t_before = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
        t_after = time.time()
    except subprocess.TimeoutExpired as e:
        t_after = time.time()
        sampler.stop()
        # return partial info
        result.update({
            "generated": "",
            "model_reply_time": t_after - t_before,
            "model_inference_time": None,
            "tokens": 0,
            "raw_stdout": getattr(e, "stdout", None) or "",
            "raw_stderr": getattr(e, "stderr", None) or "TIMEOUT",
        })
        return result

    sampler.stop()

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    raw_combined = stdout + ("\n" + stderr if stderr else "")
    result["raw_stdout"] = stdout
    result["raw_stderr"] = stderr

    # wall-clock subprocess time:
    model_reply_time = t_after - t_before
    result["model_reply_time"] = model_reply_time

    # Heuristic: if CLI echoes the prompt, remove the prompt from output to isolate generation
    generated = stdout
    if prompt_text and prompt_text.strip() and prompt_text.strip() in stdout:
        parts = stdout.split(prompt_text.strip())
        if len(parts) > 1:
            generated = parts[-1].strip()

    # Fallback: if stdout empty but stderr holds text
    if not generated and stderr:
        generated = stderr.strip()

    result["generated"] = generated
    result["tokens"] = len(generated.split())

    # Attempt to parse model-reported inference timings, e.g. lines like "inference time: 123.45 ms"
    # This is best-effort: look for numbers followed by 'ms' or 's' near keywords.
    inference_time = None
    # common patterns: "inference time: 123.4 ms", "total time: 1.23s", "real time: 0.123s"
    patterns = [
        r"inference(?:_| |-)?time[:=]?\s*([0-9]*\.?[0-9]+)\s*(ms|s)\b",
        r"model(?:_| |-)?time[:=]?\s*([0-9]*\.?[0-9]+)\s*(ms|s)\b",
        r"total(?:_| |-)?time[:=]?\s*([0-9]*\.?[0-9]+)\s*(ms|s)\b",
        r"elapsed[:=]?\s*([0-9]*\.?[0-9]+)\s*(ms|s)\b",
        r"([0-9]*\.?[0-9]+)\s*ms\s*per\s*token",
        r"([0-9]*\.?[0-9]+)\s*ms\b",
        r"([0-9]*\.?[0-9]+)\s*s\b",
    ]
    for pat in patterns:
        for match in re.finditer(pat, raw_combined, flags=re.IGNORECASE):
            num = match.group(1)
            unit = match.group(2) if match.lastindex and match.lastindex >= 2 else None
            try:
                val = float(num)
                if unit and unit.lower().startswith("ms"):
                    val = val / 1000.0
                # prefer a small positive non-zero inference_time
                if val > 0:
                    inference_time = val
                    break
            except Exception:
                continue
        if inference_time is not None:
            break

    # If we didn't find a specific model-reported time, keep None (so caller can differentiate)
    result["model_inference_time"] = inference_time

    # Resource summary
    result["resource"] = sampler.summary()

    # Optionally run TTS and measure
    if tts_after:
        tts_elapsed = speak_text_timed(result["generated"], cmd=tts_cmd)
        result["tts_time"] = tts_elapsed

    return result


def qwen25(prompt_text: str,
           llama_cli_path: str = None,
           model_path: str = None,
           n_predict: int = 16,
           threads: int = 4,
           temperature: float = 0.1,
           sampler: ResourceSampler = None,
           instruct_prefix: str = "Answer in one short sentence.",
           tts_after: bool = False,
           tts_cmd = None,
           timeout_seconds: int = 180):
    
    # sensible defaults
    if llama_cli_path is None:
        llama_cli_path = LLAMA_CLI if 'LLAMA_CLI' in globals() else str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli")
    if model_path is None:
        model_path = str(Path.home() / "Downloads" / "qwen2.5-0.5b-instruct-q3_k_m.gguf")

    exe = Path(llama_cli_path)
    model = Path(model_path)

    result = {
        "generated": "",
        "model_reply_time": 0.0,
        "model_inference_time": None,
        "tokens": 0,
        "tts_time": 0.0,
        "resource": {"peak_rss":0, "avg_rss":0, "peak_cpu":0.0, "avg_cpu":0.0, "samples":0},
        "raw_stdout": "",
        "raw_stderr": ""
    }

    if sampler is None:
        sampler = ResourceSampler(interval=0.05)

    if not exe.exists():
        raise FileNotFoundError(f"llama-cli binary not found at: {exe}")
    if not model.exists():
        raise FileNotFoundError(f"qwen model not found at: {model}")

    # build final prompt (prepend prefix if requested)
    if instruct_prefix:
        full_prompt = f"{instruct_prefix} {prompt_text.strip()}"
    else:
        full_prompt = prompt_text

    # command — adjust flags if your llama-cli uses different flag names
    cmd = [
        str(exe),
        "-m", str(model),
        "-p", full_prompt,
        "-n", str(n_predict),
        "-t", str(threads),
        "--temp", str(temperature),
    ]

    # run with resource sampling
    sampler.start()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
        t1 = time.time()
    except subprocess.TimeoutExpired as e:
        t1 = time.time()
        sampler.stop()
        result.update({
            "raw_stdout": getattr(e, "stdout", "") or "",
            "raw_stderr": getattr(e, "stderr", "") or "TIMEOUT",
            "model_reply_time": t1 - t0
        })
        return result

    sampler.stop()

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    result["raw_stdout"] = stdout
    result["raw_stderr"] = stderr

    result["model_reply_time"] = t1 - t0

    # isolate generation: remove the prompt echo if present
    generated = stdout
    if full_prompt.strip() and full_prompt.strip() in generated:
        parts = generated.split(full_prompt.strip())
        if len(parts) > 1:
            generated = parts[-1].strip()

    # if stdout empty, fallback to stderr
    if not generated and stderr:
        generated = stderr.strip()

    # last-ditch: filter diagnostic lines from combined output and take last useful line
    if not generated:
        combined = (stdout + "\n" + stderr).strip()
        lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
        filtered = [ln for ln in lines if not any(k in ln.lower() for k in ("loaded", "mem", "tokens", "total time", "trace", "init"))]
        if filtered:
            generated = filtered[-1]

    result["generated"] = (generated or "").strip()
    result["tokens"] = len(result["generated"].split())

    # Best-effort parse of any "inference time" or "total time" numbers in the raw output
    inference_time = None
    raw_combined = (stdout + "\n" + stderr).strip()
    patterns = [
        r"inference(?:_| |-)?time[:=]?\s*([0-9]*\.?[0-9]+)\s*(ms|s)\b",
        r"total(?:_| |-)?time[:=]?\s*([0-9]*\.?[0-9]+)\s*(ms|s)\b",
        r"elapsed[:=]?\s*([0-9]*\.?[0-9]+)\s*(ms|s)\b",
        r"([0-9]*\.?[0-9]+)\s*ms\s*per\s*token",
        r"([0-9]*\.?[0-9]+)\s*ms\b",
        r"([0-9]*\.?[0-9]+)\s*s\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, raw_combined, flags=re.IGNORECASE):
            num = m.group(1)
            unit = m.group(2) if m.lastindex and m.lastindex >= 2 else None
            try:
                v = float(num)
                if unit and unit.lower().startswith("ms"):
                    v = v / 1000.0
                if v > 0:
                    inference_time = v
                    break
            except Exception:
                continue
        if inference_time is not None:
            break
    result["model_inference_time"] = inference_time

    # resource summary
    result["resource"] = sampler.summary()

    # optional TTS (measured)
    if tts_after and result["generated"]:
        try:
            tts_elapsed = speak_text_timed(result["generated"])  # will swallow exceptions
        except Exception:
            tts_elapsed = speak_text_timed(result["generated"]) if 'speak_text_timed' in globals() else 0.0
        result["tts_time"] = tts_elapsed

    return result



def generate_response_local_llama(prompt_text, n_predict=128, threads=4, temperature=0.1):
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
        "--top-p", "0.2",
        "--top-k", "40",
    ]

   
    print("[LLM] Running local llama:", " ".join(shlex.quote(c) for c in cmd))
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120 + n_predict * 3)
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

def format_bytes(n):
    # small helper (exists earlier in previous code, but include here if not)
    for unit in ("B","KB","MB","GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"

def main():
    total_start = time.time()
    """
    #  record
    wav = record_wav()
    if not wav:
        print("[MAIN] Recording failed.")
        return
    """
    wav = str(Path.home() / "Rasberrypi-voice-assistant" / "recorded.wav")
    
    #  STT
    stt_start = time.time()
    transcribed = transcribe_audio(wav)
    stt_elapsed = time.time() - stt_start
    print(f"[MAIN] Transcription (STT time {stt_elapsed:.3f}s): {repr(transcribed)}")

    if not transcribed:
        print("[MAIN] No transcription found.")
        return

    
    short_prefix = "Answer in one short sentence. "
    prompt_to_model = short_prefix + transcribed.strip()
    sampler = ResourceSampler(interval=0.05)
    """
    # llama110 
    
    llama2_model_path = str(Path.home() / "Downloads" / "llama2.c-stories110M-pruned50.Q3_K_M.gguf")
    try:
        res = llama110(
            prompt_text=prompt_to_model,
            llama_cli_path=LLAMA_CLI,
            model_path=llama2_model_path,
            n_predict=64,        # limit generation length
            threads=4,
            temperature=0.1,     # low temp to prefer concise answers
            sampler=sampler,
            tts_after=False,
            timeout_seconds=90
        )
    except FileNotFoundError as e:
        print("[MAIN] Error launching llama110:", e)
        return
"""

    qwen_path = str(Path.home() / "Downloads" / "qwen2.5-0.5b-instruct-q3_k_m.gguf")

    res = qwen25(
        prompt_text=transcribed,
        llama_cli_path=LLAMA_CLI,    
        model_path=qwen_path,
        n_predict=16,
        threads=4,
        temperature=0.2,         
        sampler=sampler,
        instruct_prefix="Answer in one short sentence.",  
        tts_after=False,
        timeout_seconds=250)

        
        
    generated = res.get("generated", "")
    model_reply_time = res.get("model_reply_time", 0.0)
    model_inference_time = res.get("model_inference_time", None)
    tokens = res.get("tokens", 0)

    
    cleaned = generated
    if prompt_to_model.strip() and prompt_to_model.strip() in cleaned:
        
        cleaned = cleaned.split(prompt_to_model.strip(), 1)[-1].strip()

    
    cleaned_lines = [ln for ln in cleaned.splitlines() if not any(k in ln.lower() for k in ("loaded", "mem", "tokens", "total time", "trace"))]
    cleaned = "\n".join(cleaned_lines).strip()

    
    print("\n[MAIN] Model reply:")
    print(generated if len(generated) < 2000 else generated[:2000] + "\n... (truncated)")

    
    
    tts_time = speak_text_timed(generated)  
        

    total_elapsed = time.time() - total_start
    stats = res.get("resource", {})

    # 9) Print bench summary
    print("\n--- BENCH SUMMARY ---")
    print(f"STT time: {stt_elapsed:.3f}s")
    print(f"Model reply time (wall): {model_reply_time:.3f}s")
    if model_inference_time is not None:
        print(f"Model inference time (parsed): {model_inference_time:.3f}s")
    else:
        print("Model inference time (parsed): N/A")
    print(f"Tokens produced: {tokens}")
    print(f"TTS time: {tts_time:.3f}s")
    print(f"Total end-to-end: {total_elapsed:.3f}s")
    if isinstance(stats, dict):
        peak = stats.get("peak_rss", 0)
        avg = int(stats.get("avg_rss", 0))
        print(f"Peak RSS: {format_bytes(peak)}, Avg RSS: {format_bytes(avg)}")
        print(f"Peak CPU%: {stats.get('peak_cpu',0.0):.1f}%, Avg CPU%: {stats.get('avg_cpu',0.0):.1f}% (samples={stats.get('samples',0)})")
    print("----------------------\n")

    # 10) return full result for programmatic use
    return {"transcription": transcribed, "generated": generated, "res": res, "bench_total": total_elapsed}


if __name__ == "__main__":
    main()
