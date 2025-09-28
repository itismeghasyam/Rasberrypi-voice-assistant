from pathlib import Path

PROJECT_DIR = Path.cwd()
RECORDED_WAV = PROJECT_DIR / "recorded.wav"
SAMPLE_RATE = 8000

CHUNK_DURATION =  0.75

DEFAULT_SILENCE_TIMEOUT = 1.5  # seconds of inactivity before auto-stopping
DEFAULT_SILENCE_THRESHOLD = 1100.0  # RMS amplitude threshold for silence detection

WHISPER_EXE = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = Path.home() / "whisper.cpp" / "models" / "ggml-tiny.bin"


PIPER_MODEL_PATH = Path.home() / "Rasberrypi-voice-assistant" / "voices" / "en_US-amy-medium.onnx"
WHISPER_SERVER_URL = "http://127.0.0.1:8080"
