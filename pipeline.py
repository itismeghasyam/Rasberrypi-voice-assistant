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


def run_whisper_on_wav(wav_path: Path, whisper_exe: Path, whisper_model: Path) -> str:
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
            # ... (rest of the LLM parsing logic is fine)
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
    """
    FIXED: Piper-based async TTS with a persistent playback process.
    This avoids spawning multiple 'aplay' processes and ensures smooth audio streaming.
    """

    def __init__(
        self,
        model_path: Path,
        playback_cmd: Optional[Iterable[str]] = None,
        synthesis_timeout: int = 90,  # FIX: Increased default timeout
        on_playback_start: Optional[Any] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.playback_cmd = list(playback_cmd) if playback_cmd else ["aplay", "-q", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"]
        self.synthesis_timeout = synthesis_timeout
        self._on_playback_start = on_playback_start
        self._q: "queue.Queue[bytes | None]" = queue.Queue() # None is a sentinel for shutdown
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts_synth")
        self._play_thread: Optional[threading.Thread] = None
        self._is_running = threading.Event()

    def start(self) -> None:
        if self._is_running.is_set():
            return
        self._is_running.set()
        self._play_thread = threading.Thread(target=self._loop, daemon=True)
        self._play_thread.start()

    def stop(self) -> None:
        if not self._is_running.is_set():
            return
        # Signal the synthesis pool to finish then signal the play loop to exit
        self._executor.shutdown(wait=True)
        self._q.put(None) # Sentinel value to stop the loop
        self._is_running.clear()

        if self._play_thread:
            self._play_thread.join(timeout=2.0)
            self._play_thread = None

    def drain(self) -> None:
        """FIX: Blocks until all pending synthesis tasks are complete."""
        # This checks the internal counter of the thread pool executor
        while self._executor._work_queue.qsize() > 0:
            time.sleep(0.05)
        # Also wait for the audio queue to be fully consumed
        self._q.join()


    def _loop(self) -> None:
        """FIX: Runs a single persistent playback subprocess."""
        proc = None
        try:
            proc = subprocess.Popen(self.playback_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if self._on_playback_start and proc.stdin:
                self._on_playback_start(time.time())

            while self._is_running.is_set():
                try:
                    raw = self._q.get(timeout=0.2)
                    if raw is None: # Sentinel check
                        break
                    if proc and proc.stdin:
                        proc.stdin.write(raw)
                        self._q.task_done()
                except queue.Empty:
                    continue
                except (BrokenPipeError, OSError) as e:
                    print(f"[TTS] Playback process ended unexpectedly: {e}")
                    break
        finally:
            if proc and proc.stdin:
                proc.stdin.close()
            if proc:
                proc.wait(timeout=1.0)


    def speak_async(self, text: str) -> None:
        text = " ".join(text.split())
        if not text or not self._is_running.is_set():
            return
        self._executor.submit(self._synthesize, text)

    def _synthesize(self, text: str) -> None:
        if not self.model_path.exists():
            print(f"[TTS] Piper model not found at {self.model_path}")
            return
        cmd = ["piper", "-m", str(self.model_path), "--output-raw"]
        try:
            # FIX: Using the configurable, longer timeout
            proc = subprocess.run(
                cmd,
                input=(text + "\n").encode("utf-8"),
                capture_output=True,
                check=True,
                timeout=self.synthesis_timeout
            )
            audio = proc.stdout
            if not audio or looks_like_text(audio):
                print("[TTS] Piper returned non-audio payload; skipping")
                return
            self._q.put(audio)
        except subprocess.TimeoutExpired:
            print(f"[TTS] Piper timed out after {self.synthesis_timeout}s for text: '{text[:50]}...'")
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
    def __init__(
        self,
        piper_model: Path,
        whisper_exe: Path,
        whisper_model: Path,
        session_timeout: float = DEFAULT_SESSION_TIMEOUT,
        silence_threshold: float = SILENCE_THRESHOLD,
        chunk_duration: float = CHUNK_DURATION,
    ) -> None:
        self.rec = StreamingRecorder(sample_rate=SAMPLE_RATE, chunk_duration=chunk_duration)
        self.tts = BufferedTTS(model_path=piper_model, on_playback_start=self._on_tts_start)
        self.whisper_exe = whisper_exe
        self.whisper_model = whisper_model
        self.session_timeout = float(session_timeout)
        self.silence_threshold = float(silence_threshold)
        self.stats = Stats()

    def run(self) -> None:
        start = time.time()
        self.tts.start()
        try:
            while time.time() - start < self.session_timeout:
                ok = self._run_interaction_cycle(start)
                if not ok:
                    continue
        except KeyboardInterrupt:
            print("\n[MAIN] Interrupted by user.")
        finally:
            print("[MAIN] Shutting down...")
            self.rec.stop()
            self.tts.stop()
            self._print_stats(time.time() - start)

    def _run_interaction_cycle(self, session_start_ts: float) -> bool:
        self.rec.clear()
        self.rec.start()
        print("[VAD] Listening for voice...")

        chunks: List[np.ndarray] = []
        consecutive_silence = 0
        last_voice_idx = -1

        while True:
            if time.time() - session_start_ts >= self.session_timeout:
                print("[MAIN] Session timeout reached during recording.")
                break

            buf = self.rec.get(timeout=0.5)
            if buf is None:
                continue

            is_voiced = rms_energy(buf) >= self.silence_threshold
            chunks.append(buf)

            if is_voiced:
                last_voice_idx = len(chunks) - 1
                consecutive_silence = 0
            else:
                consecutive_silence += 1

            if last_voice_idx != -1 and consecutive_silence >= SILENT_CHUNKS_BEFORE_CUTOFF:
                self.rec.stop()
                print("[VAD] Silence detected. Processing...")
                break

        if not chunks or last_voice_idx == -1:
            return False

        chunks = chunks[: last_voice_idx + 1]

        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "utterance.wav"
            audio = np.concatenate(chunks, axis=0)
            write_wav(wav_path, audio, SAMPLE_RATE)

            print("[STT] Transcribing...")
            stt_start = time.time()
            transcript = run_whisper_on_wav(wav_path, self.whisper_exe, self.whisper_model)
            self.stats.stt_calls += 1
        transcript = (transcript or "").strip()
        print(f"[STT] Transcript: '{transcript}'")

        is_noise = bool(NOISE_RE.search(transcript)) or (not transcript)

        if not is_noise:
            print("[LLM] Getting response...")
            llm_start_ref = time.time()
            llm_out = call_llm(transcript)
            print(f"[LLM] Response: '{llm_out}'")
            if self.stats.first_llm_latency is None:
                self.stats.first_llm_latency = max(0.0, time.time() - llm_start_ref)

            if llm_out:
                print("[TTS] Synthesizing speech...")
                self.tts.speak_async(llm_out)
                self.stats.tts_segments += 1
        else:
            print("[MAIN] Transcript classified as noise or empty. Waiting for user.")
            # FIX: Use the new, effective drain method
            self.tts.drain()

        return True

    def _on_tts_start(self, started_at: float) -> None:
        if self.stats.tts_start_after_stop is None:
            self.stats.tts_start_after_stop = started_at

    def _print_stats(self, elapsed: float) -> None:
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print("\n========== SESSION STATS ==========")
        print(f"Elapsed: {elapsed:.2f}s; RSS: {rss_mb:.1f} MB")
        print(f"STT calls: {self.stats.stt_calls}")
        if self.stats.first_llm_latency is not None:
            # CLARIFICATION: This latency is for the LLM call only, not STT + LLM
            print(f"First LLM-only latency: {self.stats.first_llm_latency:.3f}s")
        print(f"TTS segments generated: {self.stats.tts_segments}")
        print("===================================\n")


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A voice assistant pipeline using Whisper, a local LLM, and Piper.")
    
    # FIX: Added CLI arguments for all necessary paths
    parser.add_argument("--piper-model", default= str(Path.home() / "Rasberrypi-voice-assistant" / "voices" / "en_US-amy-medium.onnx") ,  type=str, required=True, help="Path to Piper voice model (.onnx)")
    parser.add_argument("--whisper-exe", type=str, default=str(Path.home() / "whisper.cpp/main"), help="Path to whisper.cpp executable")
    parser.add_argument("--whisper-model", type=str, default=str(Path.home() / "whisper.cpp/models/ggml-tiny.en.bin"), help="Path to whisper.cpp model file")

    parser.add_argument("--timeout", type=float, default=DEFAULT_SESSION_TIMEOUT, help="Overall session timeout in seconds")
    parser.add_argument("--threshold", type=float, default=SILENCE_THRESHOLD, help="RMS amplitude for silence detection")
    args = parser.parse_args()

    # Verify paths exist before starting
    piper_path = Path(args.piper_model)
    whisper_exe_path = Path(args.whisper_exe)
    whisper_model_path = Path(args.whisper_model)

    if not piper_path.exists():
        print(f"[ERROR] Piper model not found at: {piper_path}")
        exit(1)
    if not whisper_exe_path.exists():
        print(f"[ERROR] Whisper executable not found at: {whisper_exe_path}")
        exit(1)
    if not whisper_model_path.exists():
        print(f"[ERROR] Whisper model not found at: {whisper_model_path}")
        exit(1)

    print("[MAIN] Starting pipeline… Press Ctrl+C to exit.")
    try:
        pipe = SessionPipeline(
            piper_model=piper_path,
            whisper_exe=whisper_exe_path,
            whisper_model=whisper_model_path,
            session_timeout=float(args.timeout),
            silence_threshold=float(args.threshold),
            chunk_duration=CHUNK_DURATION,
        )
        pipe.run()
    except Exception as e:
        print(f"[MAIN] Unhandled error: {e}")