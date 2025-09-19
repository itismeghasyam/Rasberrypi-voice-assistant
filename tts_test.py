import subprocess
import time
import psutil
import os
import piper
from pathlib import Path
import numpy as np
import wave
def speak_text_espeak(text: str) -> None:
    
    text = (text or "").strip()
    if not text:
        return
    
    print("[TTS] Speak:", text)

    try:
        subprocess.run(["espeak", text], check=True)
    except Exception as e:
        print("[TTS] espeak failed:", e)

model_path = str(Path.home() / "Rasberrypi-voice-assistant" / "voices" / "en_US-amy-medium.onnx")

def speak_text_piper(text: str, model_path=model_path):
    """
    Speak text using Piper TTS engine and play via PulseAudio (BT speakers).
    Converts Piper's raw PCM into a valid WAV.
    """
    text = (text or "").strip()
    if not text:
        return

    print("[TTS] Piper Speak:", text)

    try:
        # Load Piper model
        model = piper.PiperVoice.load(model_path)

        # Capture raw PCM output into memory buffer
        buffer = io.BytesIO()
        model.synthesize(text, buffer)

        # Get raw bytes and convert to numpy
        raw_bytes = buffer.getvalue()
        samples = np.frombuffer(raw_bytes, dtype=np.int16)

        # Save as proper WAV
        with wave.open("piper_output.wav", "wb") as wf:
            wf.setnchannels(1)        # mono
            wf.setsampwidth(2)        # 16-bit
            wf.setframerate(22050)    # sample rate (Piper default for Amy medium)
            wf.writeframes(samples.tobytes())

        # Play via PulseAudio (Bluetooth)
        subprocess.run(["paplay", "piper_output.wav"], check=True)

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
