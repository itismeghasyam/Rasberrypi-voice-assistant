import subprocess
import time
import psutil
import os
import piper
from pathlib import Path
import numpy as np
import wave
import io 
from kittentts import KittenTTS
import soundfile as sf

def speak_text_espeak(text: str) -> None:
    
    text = (text or "").strip()
    if not text:
        return
    
    print("[TTS] Speak:", text)

    try:
        subprocess.run(["espeak", text], check=True)
    except Exception as e:
        print("[TTS] espeak failed:", e)

# Base path (no extension, Piper will append .onnx and .onnx.json)
model_base = Path.home() / "Rasberrypi-voice-assistant" / "voices" / "en_US-amy-medium.onnx"

def speak_text_kitten(text: str,
                      voice: str = "expr-voice-2-f",
                      speed: float = 1.0):
    """
    Use KittenTTS package to synthesize text and play via PulseAudio/Bluetooth speaker.
    Returns the path of output wav and sample rate.
    """
    text = (text or "").strip()
    if not text:
        return None, None

    print("[TTS][Kitten] Speak:", text)
    try:
        # Initialize model (you can optionally cache this globally)
        model = KittenTTS()

        # Generate audio
        audio_data = model.generate(text=text, voice=voice, speed=speed)  # this returns a numpy array
        sr = model.sample_rate  # usually 24000 Hz with KittenTTS by default according to docs :contentReference[oaicite:3]{index=3}

        # Save to temporary wav file
        wav_path = "kitten_output.wav"
        sf.write(wav_path, audio_data, sr, subtype='PCM_16')
        
        # Play the wav via PulseAudio
        subprocess.run(["paplay", wav_path], check=True)

        return wav_path, sr

    except Exception as e:
        print("[TTS][Kitten] failed:", e)
        return None, None


def speak_text_piper(text: str, model_path="/home/kushal/Rasberrypi-voice-assistant/voices/en_US-amy-medium.onnx"):
    """
    Use Piper CLI to synthesize text and stream audio to aplay.
    """
    text = (text or "").strip()
    if not text:
        return

    print("[TTS] Piper Speak:", text)

    try:
        # Run Piper CLI exactly like your working shell command
        cmd = [
            "piper",
            "-m", model_path,
            "--output-raw"
        ]

        proc = subprocess.Popen(
            ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
            stdin=subprocess.PIPE
        )

        # Feed text into Piper, stream raw audio to aplay
        piper_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=proc.stdin)
        piper_proc.communicate(input=text.encode("utf-8"))

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
    result = benchmark_tts(speak_text_kitten, input_text, "kitten")

    print("\n=== Benchmark Result ===")
    for k, v in result.items():
        print(f"{k}: {v}")
