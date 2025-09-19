import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time
import threading
import argparse
from pathlib import Path
import wave
import psutil
import sounddevice as sd
from faster_whisper import WhisperModel

PROJECT_DIR = Path.cwd()
RECORDED_WAV = PROJECT_DIR / "recorded.wav"
SAMPLE_RATE = 16000
DURATION_SEC = 6
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

def fmt(s, w, align=">"):
    return f"{s:{align}{w}}"

def print_summary(results, include_record=False):
    cores = psutil.cpu_count(logical=True) or 1
    print("\n===== Summary =====")
    for r in results:
        parts = [
            f"model: {r['model']}",
            # include record and e2e only if present/desired
        ]
        if 'record_time_s' in r:
            parts.append(f"record_s: {r['record_time_s']:.2f}")
        parts.extend([
            f"total_s: {r['total_time_s']:.2f}",
        ])
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


def benchmark_once(audio_path, model_size, language=None):
    print(f"\n===== Benchmark: model='{model_size}', device='{DEVICE}', compute_type='{COMPUTE_TYPE}' =====")
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
        "model": model_size,
        "load_time_s": load_time,
        "transcribe_time_s": transcribe_time,
        "total_time_s": wall,
        "cpu_avg_percent": res["cpu_avg"],
        "cpu_max_percent": res["cpu_max"],
        "rss_peak_mb": res["rss_peak_mb"],
        "detected_language": info.language,
        "chars": len(transcript),
    }

def parse_args():
    p = argparse.ArgumentParser(description="Faster-Whisper CPU benchmark")
    p.add_argument("--models", type=str, default="tiny")
    p.add_argument("--repeat", type=int, default=1, help="Repeat runs per model (default 1)")
    p.add_argument("--language", type=str, default=None, help="Force language code (e.g., en)")
    p.add_argument("--skip-record", action="store_true", help="Skip recording and reuse recorded.wav")
    return p.parse_args()

def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not args.skip_record or not RECORDED_WAV.exists():
        record_wav()

    results = []
    for m in models:
        for i in range(args.repeat):
            if args.repeat > 1:
                print(f"\n--- Run {i+1}/{args.repeat} for model {m} ---")
            results.append(benchmark_once(str(RECORDED_WAV), m, language=args.language))

    print_summary(results, include_record=getattr(args, "include_record", False))

if __name__ == "__main__":
    main()
