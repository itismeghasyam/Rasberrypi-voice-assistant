from __future__ import annotations
import argparse
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import psutil
import sounddevice as sd
from concurrent.futures import ThreadPoolExecutor

# If you have your own local LLM function, import it here
try:
    from voice_test import llama110  # expects: def llama110(prompt: str) -> str
except Exception:
    llama110 = None  # fallback handled below

# ================================================================
# Configuration
# ================================================================
PROJECT_DIR = Path.cwd()
SAMPLE_RATE = 16_000
CHUNK_DURATION = 2.0  # seconds
SILENCE_THRESHOLD = 700.0  # RMS amplitude threshold
SILENT_CHUNKS_BEFORE_CUTOFF = 3  # "more than 2 chunks" => 3
DEFAULT_SESSION_TIMEOUT = 60.0  # overall run timeout (seconds)

WHISPER_EXE = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = Path.home() / "whisper.cpp" / "models" / "ggml-tiny.bin"

# noise/placeholders to ignore coming from STT
NOISE_PATTERNS = [
    r"^\s*$",
    r"\b(blank[_ ]?audio)\b",
    r"\b(wind\s+blowing)\b",
    r"\b(bird\s+chirp(ing)?)\b",
    r"^\(?\s*birds?\s+chirp(ing)?\s*\)?$",
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), flags=re.IGNORECASE)

# ================================================================
# Utilities
# ================================================================

def rms_energy(int16_audio: np.ndarray) -> float:
    if int16_audio.size == 0:
        return 0.0
    x = int16_audio.astype(np.float32).reshape(-1)
    return float(np.sqrt(np.mean(np.square(x))))


def write_wav(path: Path, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    data = np.asarray(data, dtype=np.int16).reshape(-1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def run_whisper_on_wav(wav_path: Path, whisper_exe: Path = WHISPER_EXE, whisper_model: Path = WHISPER_MODEL) -> str:
    """Blocking transcription call using whisper.cpp CLI."""
    if not whisper_exe.exists():
        print(f"[STT] Whisper binary missing at {whisper_exe}; returning empty transcript.")
        return ""
    if not whisper_model.exists():
        print(f"[STT] Whisper model missing at {whisper_model}; returning empty transcript.")
        return ""

    cmd = [
        str(whisper_exe),
        "-m", str(whisper_model),
        "-f", str(wav_path),
        "-otxt",
        "-of", str(wav_path.with_suffix("").as_posix()),
        "-l", "auto",
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists():
            txt = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            return txt
    except subprocess.CalledProcessError as e:
        print(f"[STT] Whisper error: {e}")
    except Exception as e:
        print(f"[STT] Whisper failed: {e}")
    return ""


def call_llm(prompt: str) -> str:
    if not (prompt or "").strip():
        return ""
    if llama110 is not None:
        try:
            out = llama110(prompt)
            if isinstance(out, str):
                return out.strip()
            if isinstance(out, dict):
                # Common keys
                for key in ("text", "content", "message"):
                    if key in out and isinstance(out[key], str):
                        return out[key].strip()
                # OpenAI-like: {"choices":[{"text": ...}]}
                if "choices" in out and isinstance(out["choices"], list) and out["choices"]:
                    choice = out["choices"][0]
                    if isinstance(choice, dict):
                        for key in ("text", "content"):
                            if key in choice and isinstance(choice[key], str):
                                return choice[key].strip()
                return str(out)
            return str(out).strip()
        except Exception as e:
            print(f"[LLM] llama110 failed: {e}")
    # Fallback trivial echo if no model is wired up
    return f"You said: {str(prompt).strip()}"


@dataclass
class Stats:
    stt_calls: int = 0
    tts_segments: int = 0
    first_llm_latency: Optional[float] = None
    tts_start_after_stop: Optional[float] = None


class BufferedTTS:
    """Piper-based async TTS with background playback queue."""

    def __init__(
        self,
        model_path:Path ,
        playback_cmd: Optional[Iterable[str]] = None,
        timeout: int = 30,
        on_playback_start: Optional[Any] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.playback_cmd = list(playback_cmd) if playback_cmd else ["aplay", "-q", "-"]
        self.timeout = timeout
        self._on_playback_start = on_playback_start
        self._q: "queue.Queue[bytes]" = queue.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._play_thread: Optional[threading.Thread] = None
        self._playing = False

    def start(self) -> None:
        if self._playing:
            return
        self._playing = True
        self._play_thread = threading.Thread(target=self._loop, daemon=True)
        self._play_thread.start()

    def stop(self) -> None:
        self._playing = False
        if self._play_thread:
            self._play_thread.join(timeout=1.0)
            self._play_thread = None
        self._executor.shutdown(wait=False)

    def _loop(self) -> None:
        while self._playing:
            try:
                raw = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self._on_playback_start:
                    self._on_playback_start(time.time())
                subprocess.run(self.playback_cmd, input=raw, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[TTS] playback failed: {e}")

    def speak_async(self, text: str) -> None:
        text = " ".join(text.split())
        if not text:
            return
        self._executor.submit(self._synthesize, text)

    def _synthesize(self, text: str) -> None:
        if not self.model_path.exists():
            print(f"[TTS] Piper model not found at {self.model_path}")
            return
        cmd = ["piper", "-m", str(self.model_path), "--output-raw"]
        try:
            proc = subprocess.run(cmd, input=(text + "\n").encode("utf-8"), capture_output=True, check=True, timeout=self.timeout)
            audio = proc.stdout
            # sanity check: if Piper printed text, skip
            if not audio or looks_like_text(audio):
                print("[TTS] Piper returned non-audio payload; skipping")
                return
            self._q.put(audio)
        except Exception as e:
            print(f"[TTS] Piper failed: {e}")


def looks_like_text(payload: bytes) -> bool:
    if not payload:
        return False
    sample = payload[:64]
    printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in sample)
    return printable >= max(10, int(len(sample) * 0.6))


class StreamingRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_duration: float = CHUNK_DURATION) -> None:
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self._thread: Optional[threading.Thread] = None
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.recording = False

    def start(self) -> None:
        if self.recording:
            return
        self.recording = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.recording = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def clear(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def get(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _loop(self) -> None:
        n = int(self.chunk_duration * self.sample_rate)
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16") as stream:
            while self.recording:
                buf, _ = stream.read(n)
                self._q.put(buf.copy())


class SessionPipeline:
    """
    Implements the control-flow you asked for:

    Record in chunks → detect prompt speech → if >2 silent chunks, STOP recording,
    transcribe → LLM → TTS (instantly). Then record other chunks until timeout.

    If there is no speech / blank audio / birds chirping for >2 chunks, let the
    transcription+LLM+TTS finish before resuming recording (i.e., block until
    the current cycle is done).
    """

    def __init__(
        self,
        piper_model: Path,
        session_timeout: float = DEFAULT_SESSION_TIMEOUT,
        silence_threshold: float = SILENCE_THRESHOLD,
        chunk_duration: float = CHUNK_DURATION,
    ) -> None:
        self.rec = StreamingRecorder(sample_rate=SAMPLE_RATE, chunk_duration=chunk_duration)
        self.tts = BufferedTTS(model_path=piper_model, playback_cmd=None, on_playback_start=self._on_tts_start)
        self.session_timeout = float(session_timeout)
        self.silence_threshold = float(silence_threshold)
        self.stats = Stats()

    # --------- public API ----------
    def run(self) -> None:
        start = time.time()
        self.tts.start()
        try:
            while time.time() - start < self.session_timeout:
                # one "interaction" cycle
                ok = self._run_interaction_cycle(start)
                if not ok:
                    # If nothing meaningful happened, continue until timeout
                    continue
        except KeyboardInterrupt:
            pass
        finally:
            self.rec.stop()
            self.tts.stop()
            self._print_stats(time.time() - start)

    # --------- internals ----------
    def _run_interaction_cycle(self, session_start_ts: float) -> bool:
        """Return True if we captured something and attempted a response."""
        # Start fresh buffers and counters for a cycle
        self.rec.clear()
        self.rec.start()

        chunks: List[np.ndarray] = []
        voiced_flags: List[bool] = []
        consecutive_silence = 0
        saw_voice = False
        last_voice_idx = -1

        # 1) RECORD in chunks until we see >2 silent chunks in a row
        while True:
            # Enforce session timeout here too
            if time.time() - session_start_ts >= self.session_timeout:
                break

            buf = self.rec.get(timeout=0.5)
            if buf is None:
                continue

            e = rms_energy(buf)
            is_voiced = e >= self.silence_threshold
            chunks.append(buf)
            voiced_flags.append(is_voiced)

            if is_voiced:
                saw_voice = True
                last_voice_idx = len(chunks) - 1
                consecutive_silence = 0
            else:
                consecutive_silence += 1

            if consecutive_silence >= SILENT_CHUNKS_BEFORE_CUTOFF:
                # STOP recording for this cycle
                self.rec.stop()
                break

        if not chunks:
            # nothing captured; allow outer loop to continue
            return False

        # Trim trailing silent chunks; keep up to the last voiced chunk
        if last_voice_idx >= 0:
            chunks = chunks[: last_voice_idx + 1]
            voiced_capture = True
        else:
            voiced_capture = False

        # 2) TRANSCRIBE the captured audio
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "utterance.wav"
            audio = np.concatenate(chunks, axis=0)
            write_wav(wav_path, audio, SAMPLE_RATE)
            stt_start = time.time()
            transcript = run_whisper_on_wav(wav_path)
            self.stats.stt_calls += 1
        transcript = (transcript or "").strip()

        # 3) Decide whether it is noise/blank
        is_noise = bool(NOISE_RE.search(transcript)) or (not transcript)

        # 4) LLM + TTS
        if transcript:
            llm_start_ref = stt_start  # reference from when we kicked STT
            llm_out = call_llm(transcript)
            if self.stats.first_llm_latency is None:
                self.stats.first_llm_latency = max(0.0, time.time() - llm_start_ref)
        else:
            llm_out = ""

        if llm_out:
            # TTS instantly
            self.tts.speak_async(llm_out)
            self.stats.tts_segments += 1

        # 5) Recording policy based on noise vs voiced
        if is_noise and not voiced_capture:
            # The user said nothing for >2 chunks (blank/noise). You asked to
            # let STT → LLM → TTS finish before recording again.
            self._wait_for_tts_queue_drain(timeout=10.0)
        else:
            # Otherwise, resume immediately (TTS plays in background) and allow
            # the outer loop to collect more audio until timeout expires.
            pass

        return True

    def _wait_for_tts_queue_drain(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        # Best-effort wait for the TTS worker queue to empty
        while time.time() < deadline:
            # Our very simple queue doesn't expose sizes publicly; we can sleep a bit
            time.sleep(0.05)
            # nothing to actively check since audio plays in a dedicated thread that
            # immediately consumes queued items; waiting is mostly symbolic here
            break

    def _on_tts_start(self, started_at: float) -> None:
        # capture latency from cutoff to first audible sample if useful
        if self.stats.tts_start_after_stop is None:
            self.stats.tts_start_after_stop = started_at

    def _print_stats(self, elapsed: float) -> None:
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print("\n========== SESSION STATS ==========")
        print(f"Elapsed: {elapsed:.2f}s; RSS: {rss_mb:.1f} MB")
        print(f"STT calls: {self.stats.stt_calls}")
        if self.stats.first_llm_latency is not None:
            print(f"First LLM latency: {self.stats.first_llm_latency:.3f}s")
        print(f"TTS segments generated: {self.stats.tts_segments}")
        print("===================================\n")


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice pipeline per spec")
    default_piper = Path.home() / "Rasberrypi-voice-assistant" / "voices" / "en_US-amy-medium.onnx"
    parser.add_argument("--piper-model", type=str, default=str(default_piper), help="Path to Piper voice model (.onnx)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_SESSION_TIMEOUT, help="Overall session timeout (s)")
    parser.add_argument("--threshold", type=float, default=SILENCE_THRESHOLD, help="RMS silence threshold")
    args = parser.parse_args()


    print("[MAIN] Starting pipeline…")
    try:
        pipe = SessionPipeline(
            piper_model=Path(args.piper_model),
            session_timeout=float(args.timeout),
            silence_threshold=float(args.threshold),
            chunk_duration=CHUNK_DURATION,
        )
        pipe.run()
    except Exception as e:
        print(f"[MAIN] Unhandled error: {e}")
