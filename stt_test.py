import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import pipeline
import torch
import time
import threading
import argparse
import json
from pathlib import Path
import wave
import psutil
import sounddevice as sd
import subprocess, shlex, re

# whisper.cpp config
WHISPER_EXE_PATH = str(Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli")
WHISPER_MODEL = str(Path.home() / "whisper.cpp" / "models" / "ggml-tiny.bin")


from faster_whisper import WhisperModel
from vosk import Model as VoskModel, KaldiRecognizer

# Config
PROJECT_DIR = Path.cwd()
RECORDED_WAV = PROJECT_DIR / "recorded.wav"
SAMPLE_RATE = 16000
DURATION_SEC = 6


#Whisper config
WHISPER_EXE_PATH = str(Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli")
WHISPER_MODEL = str(Path.home() / "whisper.cpp" / "models" / "ggml-tiny.bin")


# Faster-Whisper defaults 
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CPU_THREADS = 4



def record_wav(path=RECORDED_WAV, duration=DURATION_SEC, sr=SAMPLE_RATE):
    print(f"[REC] Recording {duration}s @ {sr}Hz -> {path}")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    print(f"[REC] Saved: {path}")
    return str(path)


class ResourceSampler:
    """Background sampler for process CPU% and RSS with blocking intervals for accuracy."""
    def __init__(self, pid=None, interval=0.2):
        self.proc = psutil.Process(pid or psutil.Process().pid)
        self.interval = interval
        self._stop = threading.Event()
        self.cpu = []
        self.mem = []

    def start(self):
        # prime measurement window
        self.proc.cpu_percent(None)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            self.cpu.append(self.proc.cpu_percent(interval=self.interval))
            self.mem.append(self.proc.memory_info().rss)

    def stop(self):
        self._stop.set()
        self._t.join()

    def summary(self):
        avg = sum(self.cpu) / len(self.cpu) if self.cpu else 0.0
        mx = max(self.cpu) if self.cpu else 0.0
        cur_rss = self.proc.memory_info().rss
        peak_rss = max(self.mem) if self.mem else cur_rss
        return {
            "cpu_avg": avg,
            "cpu_max": mx,
            "rss_cur_mb": cur_rss / (1024**2),
            "rss_peak_mb": peak_rss / (1024**2),
        }


def print_summary(results, include_record=False):
    cores = psutil.cpu_count(logical=True) or 1
    print("\n===== Summary =====")
    for r in results:
        parts = [f"model: {r['model']}"]
        if 'record_time_s' in r:
            parts.append(f"record_s: {r['record_time_s']:.2f}")
        parts.append(f"total_s: {r['total_time_s']:.2f}")
        if include_record and 'total_e2e_s' in r:
            parts.append(f"e2e_s: {r['total_e2e_s']:.2f}")
        parts.extend([
            f"load_s: {r['load_time_s']:.2f}",
            f"xcribe_s: {r['transcribe_time_s']:.2f}",
            f"cpu_avg%: {r['cpu_avg_percent']:.1f}",
            f"cpu_max%: {r['cpu_max_percent']:.1f}",
            f"avg_norm%: {r['cpu_avg_percent']/cores:.1f}",
            f"max_norm%: {r['cpu_max_percent']/cores:.1f}",
            f"peak_MB: {r['rss_peak_mb']:.1f}",
            f"lang: {r['detected_language']}",
            f"chars: {r['chars']}",
        ])
        print(", ".join(parts))


# Faster-Whisper   
def benchmark_once_fw(audio_path, model_size, language=None):
    print(f"\n===== Benchmark (FW): model='{model_size}', device='{DEVICE}', compute_type='{COMPUTE_TYPE}' =====")
    if CPU_THREADS:
        print(f"[INFO] Limiting CPU threads to {CPU_THREADS}")
    proc = psutil.Process()
    cpu_before = proc.cpu_times()
    wall_start = time.time()

    sampler = ResourceSampler(interval=0.15)
    sampler.start()

    t0 = time.time()
    model = WhisperModel(
        model_size,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS
    )
    load_time = time.time() - t0
    print(f"[LOAD] Model loaded in {load_time:.2f}s")

    t1 = time.time()
    segments, info = model.transcribe(audio_path, language=language, task="transcribe")
    transcribe_time = time.time() - t1

    sampler.stop()
    wall = time.time() - wall_start
    cpu_after = proc.cpu_times()
    user_cpu = cpu_after.user - cpu_before.user
    sys_cpu = cpu_after.system - cpu_before.system
    res = sampler.summary()

    transcript = " ".join(seg.text for seg in segments).strip()

    print(f"[INFO] Language: {info.language}")
    print(f"[TIME] Load: {load_time:.2f}s | Transcribe: {transcribe_time:.2f}s | Total: {wall:.2f}s")
    print(f"[CPU ] user: {user_cpu:.2f}s | system: {sys_cpu:.2f}s | avg%: {res['cpu_avg']:.1f} | max%: {res['cpu_max']:.1f}")
    print(f"[MEM ] RSS now: {res['rss_cur_mb']:.1f} MB | Peak (sampled): {res['rss_peak_mb']:.1f} MB")
    print(f"[TEXT] {transcript[:200]}{'...' if len(transcript) > 200 else ''}")

    return {
        "model": f"fw:{model_size}",
        "load_time_s": load_time,
        "transcribe_time_s": transcribe_time,
        "total_time_s": wall,
        "cpu_avg_percent": res["cpu_avg"],
        "cpu_max_percent": res["cpu_max"],
        "rss_peak_mb": res["rss_peak_mb"],
        "detected_language": info.language,
        "chars": len(transcript),
    }


#Whisper 
def benchmark_once_whispercpp(audio_path):
    print(f"\n===== Benchmark (whisper.cpp): exe='{WHISPER_EXE_PATH}', model='{WHISPER_MODEL}' =====")
    proc = psutil.Process()
    cpu_before = proc.cpu_times()
    wall_start = time.time()

    sampler = ResourceSampler(interval=0.15)
    sampler.start()

    # --- load time (binary + model startup) ---
    t0 = time.time()
    exe = Path(WHISPER_EXE_PATH)
    model = Path(WHISPER_MODEL)
    if not exe.exists():
        raise FileNotFoundError(f"Whisper binary not found: {exe}")
    if not model.exists():
        raise FileNotFoundError(f"Whisper model not found: {model}")
    load_time = time.time() - t0   # here, "load" is trivial, actual cost is inside run

    # --- transcription ---
    t1 = time.time()
    cmd = [str(exe), "-m", str(model), "-f", str(audio_path), "--no-prints", "--output-txt"]
    print("[STT] Running:", " ".join(shlex.quote(c) for c in cmd))
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print("[STT] Timeout expired")
        return None
    transcribe_time = time.time() - t1

    # read result
    txt_path1 = Path(audio_path).with_suffix(".txt")
    txt_path2 = Path(str(audio_path) + ".txt")
    transcript = ""
    for p in (txt_path1, txt_path2):
        if p.exists():
            transcript = p.read_text(encoding="utf-8").strip()
            break

    sampler.stop()
    wall = time.time() - wall_start
    cpu_after = proc.cpu_times()
    user_cpu = cpu_after.user - cpu_before.user
    sys_cpu  = cpu_after.system - cpu_before.system
    res = sampler.summary()

    print(f"[TIME] Load: {load_time:.2f}s | Transcribe: {transcribe_time:.2f}s | Total: {wall:.2f}s")
    print(f"[CPU ] user: {user_cpu:.2f}s | system: {sys_cpu:.2f}s | avg%: {res['cpu_avg']:.1f} | max%: {res['cpu_max']:.1f}")
    print(f"[MEM ] RSS now: {res['rss_cur_mb']:.1f} MB | Peak (sampled): {res['rss_peak_mb']:.1f} MB")
    print(f"[TEXT] {transcript[:200]}{'...' if len(transcript) > 200 else ''}")

    return {
        "model": "whisper.cpp-tiny",
        "load_time_s": load_time,
        "transcribe_time_s": transcribe_time,
        "total_time_s": wall,
        "cpu_avg_percent": res["cpu_avg"],
        "cpu_max_percent": res["cpu_max"],
        "rss_peak_mb": res["rss_peak_mb"],
        "detected_language": "unk",   # whisper.cpp doesn’t output lang unless you add --lang
        "chars": len(transcript),
    }


# Whisper-distil
def benchmark_once_distil(audio_path, model_id="distil-whisper/distil-small.en"):
    print(f"\n===== Benchmark (DISTIL): model='{model_id}' =====")
    proc = psutil.Process()
    cpu_before = proc.cpu_times()
    wall_start = time.time()

    sampler = ResourceSampler(interval=0.15)
    sampler.start()

    # Load model/pipeline
    t0 = time.time()
    # You can limit CPU threads if you want: torch.set_num_threads(4)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device="cpu"   # Pi = CPU
        # dtype left default (float32) for CPU stability
    )
    load_time = time.time() - t0
    print(f"[LOAD] Model loaded in {load_time:.2f}s")

    # Transcribe
    t1 = time.time()
    out = asr(audio_path)
    transcribe_time = time.time() - t1

    sampler.stop()
    wall = time.time() - wall_start
    cpu_after = proc.cpu_times()
    user_cpu = cpu_after.user - cpu_before.user
    sys_cpu  = cpu_after.system - cpu_before.system
    res = sampler.summary()

    transcript = (out.get("text") or "").strip()
    detected_language = "en"  # Distil-Whisper is English-only for now
    print(f"[INFO] Language: {detected_language}")
    print(f"[TIME] Load: {load_time:.2f}s | Transcribe: {transcribe_time:.2f}s | Total: {wall:.2f}s")
    print(f"[CPU ] user: {user_cpu:.2f}s | system: {sys_cpu:.2f}s | avg%: {res['cpu_avg']:.1f} | max%: {res['cpu_max']:.1f}")
    print(f"[MEM ] RSS now: {res['rss_cur_mb']:.1f} MB | Peak (sampled): {res['rss_peak_mb']:.1f} MB")
    print(f"[TEXT] {transcript[:200]}{'...' if len(transcript) > 200 else ''}")

    return {
        "model": f"distil:{model_id.split('/')[-1]}",
        "load_time_s": load_time,
        "transcribe_time_s": transcribe_time,
        "total_time_s": wall,
        "cpu_avg_percent": res["cpu_avg"],
        "cpu_max_percent": res["cpu_max"],
        "rss_peak_mb": res["rss_peak_mb"],
        "detected_language": detected_language,
        "chars": len(transcript),
    }


# Vosk
def _guess_lang_from_path(model_dir: str) -> str:
    name = Path(model_dir).name.lower()
    # crude heuristic: pick common language codes from folder name
    for code in ("en", "hi", "fr", "de", "es", "it", "ru", "pt", "zh", "ja"):
        if f"-{code}-" in name or name.endswith(f"-{code}") or name.startswith(f"{code}-"):
            return code
    return "unk"

def benchmark_once_vosk(audio_path, model_dir= str(Path.home() / "models" / "vosk-model-small-en-us-0.15")):
    print(f"\n===== Benchmark (VOSK): model_dir='{model_dir}' =====")
    proc = psutil.Process()
    cpu_before = proc.cpu_times()
    wall_start = time.time()

    sampler = ResourceSampler(interval=0.15)
    sampler.start()

    # Load model
    t0 = time.time()
    vmodel = VoskModel(str(model_dir))
    load_time = time.time() - t0
    print(f"[LOAD] Model loaded in {load_time:.2f}s")

    # Open audio
    wf = wave.open(audio_path, "rb")
    assert wf.getnchannels() == 1, "Vosk expects mono audio"
    assert wf.getsampwidth() == 2, "Vosk expects 16-bit PCM"
    sr = wf.getframerate()
    rec = KaldiRecognizer(vmodel, sr)
    rec.SetWords(True)

    # Transcribe (streaming)
    t1 = time.time()
    text_parts = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            j = json.loads(rec.Result())
            t = (j.get("text") or "").strip()
            if t:
                text_parts.append(t)
    j_final = json.loads(rec.FinalResult())
    t = (j_final.get("text") or "").strip()
    if t:
        text_parts.append(t)
    transcribe_time = time.time() - t1

    sampler.stop()
    wall = time.time() - wall_start
    cpu_after = proc.cpu_times()
    user_cpu = cpu_after.user - cpu_before.user
    sys_cpu = cpu_after.system - cpu_before.system
    res = sampler.summary()

    transcript = " ".join(text_parts).strip()
    detected_language = _guess_lang_from_path(model_dir)

    print(f"[INFO] Language (from model path): {detected_language}")
    print(f"[TIME] Load: {load_time:.2f}s | Transcribe: {transcribe_time:.2f}s | Total: {wall:.2f}s")
    print(f"[CPU ] user: {user_cpu:.2f}s | system: {sys_cpu:.2f}s | avg%: {res['cpu_avg']:.1f} | max%: {res['cpu_max']:.1f}")
    print(f"[MEM ] RSS now: {res['rss_cur_mb']:.1f} MB | Peak (sampled): {res['rss_peak_mb']:.1f} MB")
    print(f"[TEXT] {transcript[:200]}{'...' if len(transcript) > 200 else ''}")

    return {
        "model": f"vosk:{Path(model_dir).name}",
        "load_time_s": load_time,
        "transcribe_time_s": transcribe_time,
        "total_time_s": wall,
        "cpu_avg_percent": res["cpu_avg"],
        "cpu_max_percent": res["cpu_max"],
        "rss_peak_mb": res["rss_peak_mb"],
        "detected_language": detected_language,
        "chars": len(transcript),
    }


# CLI 
def parse_args():
    p = argparse.ArgumentParser(description="STT benchmarks (Faster-Whisper or Vosk)")
    p.add_argument("--engine", choices=["fw", "vosk","distil","whispercpp"], default="fw",
                   help="Engine to benchmark: fw (Faster-Whisper) or vosk")
    p.add_argument("--models", type=str, default="tiny",
                   help="FW: comma-separated models (e.g., tiny or tiny,base). Ignored for Vosk.")
    p.add_argument("--vosk-model-dir", type=str, default=None,
                   help="Vosk: path to unpacked model directory")
    p.add_argument("--repeat", type=int, default=1, help="Repeat runs per model (default 1)")
    p.add_argument("--language", type=str, default=None,
                   help="FW only: force language code (e.g., en)")
    p.add_argument("--skip-record", action="store_true",
                   help="Skip recording and reuse recorded.wav")
    p.add_argument("--include-record", action="store_true",
                   help="Add recording time to the reported e2e total")
    p.add_argument("--distil-model", type=str, default="distil-whisper/distil-small.en",
               help="HF model id for Distil-Whisper (English only)")
    return p.parse_args()


def main():
    args = parse_args()

    # Record (or reuse)
    record_time = 0.0
    if not args.skip_record or not RECORDED_WAV.exists():
        t0 = time.time()
        record_wav()
        record_time = time.time() - t0
        print(f"[TIME] Record: {record_time:.2f}s")

    results = []

    if args.engine == "fw":
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        for m in models:
            for i in range(args.repeat):
                if args.repeat > 1:
                    print(f"\n--- Run {i+1}/{args.repeat} for FW model {m} ---")
                res = benchmark_once_fw(str(RECORDED_WAV), m, language=args.language)
                res["record_time_s"] = record_time
                res["total_e2e_s"] = res["total_time_s"] + (record_time if args.include_record else 0.0)
                results.append(res)
    
    elif args.engine == "distil":
        res = benchmark_once_distil(str(RECORDED_WAV), model_id=args.distil_model)
        res["record_time_s"] = record_time
        res["total_e2e_s"] = res["total_time_s"] + (record_time if args.include_record else 0.0)
        results.append(res)
        
    elif args.engine == "whispercpp":
        res = benchmark_once_whispercpp(str(RECORDED_WAV))
        if res:
            res["record_time_s"] = record_time
            res["total_e2e_s"] = res["total_time_s"] + (record_time if args.include_record else 0.0)
            results.append(res)



    else:  # Vosk
        for i in range(args.repeat):
            if args.repeat > 1:
                print(f"\n--- Run {i+1}/{args.repeat} for Vosk ---")
            res = benchmark_once_vosk(str(RECORDED_WAV), model_dir=str(Path.home() / "models" / "vosk-model-small-en-us-0.15"))
            res["record_time_s"] = record_time
            res["total_e2e_s"] = res["total_time_s"] + (record_time if args.include_record else 0.0)
            results.append(res)
            
    

    print_summary(results, include_record=args.include_record)


if __name__ == "__main__":
    main()
