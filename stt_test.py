import time
import threading
from pathlib import Path
import wave
import psutil
import sounddevice as sd
from faster_whisper import WhisperModel
from voice_test import PROJECT_DIR

RECORDED_WAV = PROJECT_DIR / "recorded.wav"
SAMPLE_RATE = 16000
DURATION_SEC = 6
MODEL_SIZES = ["tiny", "base", "small"]  # change as you like
DEVICE = "cpu"  
COMPUTE_TYPE = "int8"  
    

def record_wav(path=RECORDED_WAV, duration=DURATION_SEC, sr=SAMPLE_RATE):
    print(f"[REC] Recording {duration}s @ {sr}Hz -> {path}")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    print(f"[REC] Saved: {path}")
    return str(path)

class ResourceSampler:
    """Background sampler for CPU% and RSS."""
    def __init__(self, pid=None, interval=0.2):
        self.proc = psutil.Process(pid or psutil.Process().pid)
        self.interval = interval
        self._stop = threading.Event()
        self.cpu_samples = []
        self.mem_samples = []
        # prime cpu_percent to avoid first zero spike
        self.proc.cpu_percent(None)

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop.is_set():
            self.cpu_samples.append(self.proc.cpu_percent(interval=None))  # non-blocking
            mem = self.proc.memory_info().rss  # bytes
            self.mem_samples.append(mem)
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        self.thread.join()

    def summary(self):
        if self.cpu_samples:
            avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples)
            max_cpu = max(self.cpu_samples)
        else:
            avg_cpu = max_cpu = 0.0
        max_rss = max(self.mem_samples) if self.mem_samples else self.proc.memory_info().rss
        cur_rss = self.proc.memory_info().rss
        return {
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "current_rss_mb": cur_rss / (1024**2),
            "peak_rss_mb": max_rss / (1024**2),
        }

def fmt_secs(s): return f"{s:.2f}s"

def benchmark_once(audio_path, model_size):
    print(f"\n===== Benchmark: model='{model_size}', device='{DEVICE}', compute_type='{COMPUTE_TYPE}' =====")

    proc = psutil.Process()
    cpu_times_before = proc.cpu_times()
    wall_start = time.time()

    # Start background sampler
    sampler = ResourceSampler(interval=0.15)
    sampler.start()

    # 1) Load model
    t0 = time.time()
    model = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE_TYPE)
    load_time = time.time() - t0
    print(f"[LOAD] Model loaded in {fmt_secs(load_time)}")

    # 2) Transcribe
    t1 = time.time()
    segments, info = model.transcribe(audio_path)
    transcribe_time = time.time() - t1

    # Stop sampler and compute stats
    sampler.stop()
    wall_elapsed = time.time() - wall_start
    cpu_times_after = proc.cpu_times()

    # CPU times (process)
    user_cpu = cpu_times_after.user - cpu_times_before.user
    sys_cpu = cpu_times_after.system - cpu_times_before.system

    # Stitch transcript (optional)
    transcript = " ".join([seg.text for seg in segments]).strip()

    # Resource summary
    res = sampler.summary()

    print(f"[INFO] Language: {info.language}")
    print(f"[TIME] Load: {fmt_secs(load_time)} | Transcribe: {fmt_secs(transcribe_time)} | Total: {fmt_secs(wall_elapsed)}")
    print(f"[CPU ] user: {user_cpu:.2f}s | system: {sys_cpu:.2f}s | avg%: {res['avg_cpu_percent']:.1f} | max%: {res['max_cpu_percent']:.1f}")
    print(f"[MEM ] RSS now: {res['current_rss_mb']:.1f} MB | Peak (sampled): {res['peak_rss_mb']:.1f} MB")
    print(f"[TEXT] {transcript[:200]}{'...' if len(transcript) > 200 else ''}")

    return {
        "model": model_size,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "load_time_s": load_time,
        "transcribe_time_s": transcribe_time,
        "total_time_s": wall_elapsed,
        "cpu_user_s": user_cpu,
        "cpu_system_s": sys_cpu,
        "cpu_avg_percent": res["avg_cpu_percent"],
        "cpu_max_percent": res["max_cpu_percent"],
        "rss_current_mb": res["current_rss_mb"],
        "rss_peak_mb": res["peak_rss_mb"],
        "detected_language": info.language,
        "chars": len(transcript),
    }

def main():
    audio_file = record_wav()
    results = []
    for m in MODEL_SIZES:
        results.append(benchmark_once(audio_file, m))

    # Pretty print summary table
    print("\n===== Summary =====")
    cols = [
        "model","total_time_s","load_time_s","transcribe_time_s",
        "cpu_avg_percent","cpu_max_percent",
        "rss_peak_mb","detected_language","chars"
    ]
    header = " | ".join(f"{c:>18}" for c in cols)
    print(header)
    print("-"*len(header))
    for r in results:
        row = " | ".join(f"{str(round(r[c],2) if isinstance(r[c], (int,float)) else r[c]):>18}" for c in cols)
        print(row)

if __name__ == "__main__":
    main()
