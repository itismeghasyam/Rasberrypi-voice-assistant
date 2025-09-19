import subprocess
import time
import psutil
import os
import piper
from pathlib import Path
def speak_text_espeak(text: str) -> None:
    
    text = (text or "").strip()
    if not text:
        return
    
    print("[TTS] Speak:", text)

    try:
        subprocess.run(["espeak", text], check=True)
    except Exception as e:
        print("[TTS] espeak failed:", e)

def speak_text_piper(text: str, model_path=str(Path.home()/ "Rasberrypi-voice-assistant" / "voices"/ "en_US-amy-medium.onnx")):
  
    text = (text or "").strip()
    if not text:
        return

    print("[TTS] Piper Speak:", text)

    try:
        model = piper.PiperVoice.load(model_path)

        # Write synthesized audio
        with open("piper_output.wav", "wb") as f:
            model.synthesize(text, f)
            f.flush()
            os.fsync(f.fileno())

        # Ensure PulseAudio accepts format
        subprocess.run([
            "ffmpeg", "-y", "-i", "piper_output.wav",
            "-ar", "22050", "-ac", "1", "piper_fixed.wav"
        ], check=True)

        # Play via PulseAudio (Bluetooth-friendly)
        subprocess.run(["paplay", "piper_fixed.wav"], check=True)

    except Exception as e:
        print("[TTS] Piper failed:", e)


def benchmark_tts(tts_func, text: str, engine_name: str):
    """
    Benchmark a TTS function: measures response time and resource usage.
    """
    process = psutil.Process(os.getpid())

    # Capture before metrics
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB

    # Time execution
    start_time = time.time()
    tts_func(text)
    end_time = time.time()

    # Capture after metrics
    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB

    result = {
        "engine": engine_name,
        "text": text,
        "response_time_sec": round(end_time - start_time, 4),
        "cpu_usage_percent": cpu_after - cpu_before,
        "memory_usage_mb": round(mem_after - mem_before, 4),
    }
    return result



if __name__ == "__main__":
    input_text = "Hello, How are you"
    result = benchmark_tts(speak_text_piper, input_text, "piper")

    print("\n=== Benchmark Result ===")
    for k, v in result.items():
        print(f"{k}: {v}")
