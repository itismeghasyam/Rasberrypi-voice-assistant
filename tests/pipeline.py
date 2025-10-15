
from __future__ import annotations
import re
import argparse

import contextlib
import json
import math
import os
import queue
import shutil
import subprocess

import tempfile
import threading
import time
import wave

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple


import numpy as np
import psutil
import sounddevice as sd
from concurrent.futures import Future, ThreadPoolExecutor


from voice_test import llama110

# ================================================================
# Configuration
# ================================================================
PROJECT_DIR = Path.cwd()
RECORDED_WAV = PROJECT_DIR / "recorded.wav"
SAMPLE_RATE = 16000

CHUNK_DURATION = 2.0  # seconds

DEFAULT_SILENCE_TIMEOUT = 10.0  # seconds of inactivity before auto-stopping
DEFAULT_SILENCE_THRESHOLD = 700.0  # RMS amplitude threshold for silence detection

WHISPER_EXE = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = Path.home() / "whisper.cpp" / "models" / "ggml-tiny.bin"


PIPER_MODEL_PATH = Path.home() / "Rasberrypi-voice-assistant" / "voices" / "en_US-amy-medium.onnx"

# ================================================================
# Streaming Audio Recorder
# ================================================================


class StreamingRecorder:
    """Capture microphone input continuously and expose fixed-size chunks."""

    def __init__(self, chunk_duration: float = CHUNK_DURATION, sample_rate: int = SAMPLE_RATE):
        self.chunk_duration = float(chunk_duration)
        self.sample_rate = int(sample_rate)
        self.chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.recording = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start capturing microphone audio in a background thread."""

        if self.recording:
            return
        self.recording = True
        self._thread = threading.Thread(target=self._record_loop, name="StreamingRecorder", daemon=True)
        self._thread.start()

    def _record_loop(self) -> None:
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16") as stream:
            while self.recording:
                audio_chunk, _ = stream.read(chunk_samples)
                # Copy to detach from PortAudio's buffers
                self.chunk_queue.put(audio_chunk.copy())

    def get_chunk(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            return self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self) -> None:
        """Remove any queued audio chunks without blocking."""

        try:
            while True:
                self.chunk_queue.get_nowait()
        except queue.Empty:
            return

    def stop(self) -> None:
        """Signal the recorder to stop and wait for the background thread."""

        self.recording = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None


# ================================================================
# Parallel Speech-to-Text (whisper.cpp)

# ================================================================


class ParallelSTT:

    """Process audio chunks asynchronously using the whisper.cpp CLI."""

    def __init__(
        self,
        num_workers: int = 2,

        sample_rate: int = SAMPLE_RATE,
        whisper_exe: Path = WHISPER_EXE,
        whisper_model: Path = WHISPER_MODEL,
        whisper_threads: Optional[int] = None,
        emit_partials: bool = False,
    ) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max(1, num_workers))
        self.sample_rate = int(sample_rate)
        self.whisper_exe = Path(whisper_exe)
        self.whisper_model = Path(whisper_model)
        self.whisper_threads = int(whisper_threads) if whisper_threads else None
        self.emit_partials = bool(emit_partials)

        self._chunks: List[bytes] = []
        self._transcript_lock = threading.Lock()
        self._emitted_transcript = ""
        self._last_partial = ""
        self._temp_dir = Path(tempfile.mkdtemp(prefix="pipeline_stt_"))

        if not self.whisper_exe.exists():
            raise FileNotFoundError(f"Whisper binary not found: {self.whisper_exe}")
        if not self.whisper_model.exists():
            raise FileNotFoundError(f"Whisper model not found: {self.whisper_model}")

    # --------------------- Whisper helpers -----------------------

    def _write_wav(self, audio_bytes: bytes, prefix: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", prefix=f"{prefix}_", dir=self._temp_dir, delete=False
        )
        tmp_path = Path(tmp.name)
        tmp.close()
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_bytes)
        return tmp_path

    def _run_whisper(self, wav_path: Path, timeout: int = 60) -> str:
        cmd = [
            str(self.whisper_exe),
            "-m",
            str(self.whisper_model),
            "-f",
            str(wav_path),
            "--no-prints",
            "--output-txt",
        ]
        if self.whisper_threads:
            cmd += ["-t", str(self.whisper_threads)]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print("[STT][Whisper] Transcription timed out")
            return ""
        except Exception as exc:
            print(f"[STT][Whisper] Error running whisper.cpp: {exc}")
            return ""

        text = ""
        candidates = [wav_path.with_suffix(".txt"), Path(str(wav_path) + ".txt")]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception as exc:
                print(f"[STT][Whisper] Failed reading {candidate.name}: {exc}")
                text = ""
            finally:
                candidate.unlink(missing_ok=True)
            if text:
                break

        return text

    def _process_chunk_whisper(self, audio_bytes: bytes, chunk_id: int) -> Dict[str, Any]:
        wav_path = self._write_wav(audio_bytes, f"chunk{chunk_id}")
        try:
            text = self._run_whisper(wav_path, timeout=45)
        finally:
            wav_path.unlink(missing_ok=True)

        text = (text or "").strip()
        if not text:
            return {"chunk_id": chunk_id, "text": "", "is_final": False}

        with self._transcript_lock:
            if text.lower() == self._last_partial.lower():
                return {"chunk_id": chunk_id, "text": "", "is_final": False}
            self._last_partial = text
            self._emitted_transcript = (f"{self._emitted_transcript} {text}").strip()

        return {"chunk_id": chunk_id, "text": text, "is_final": False}

    def _finalize_whisper(self, chunk_id: int, mark_final: bool) -> Dict[str, Any]:
        with self._transcript_lock:
            chunks = list(self._chunks)
            emitted = self._emitted_transcript.strip()
            self._last_partial = ""
            self._chunks.clear()

        if not chunks:
            return {"chunk_id": chunk_id, "text": "", "is_final": True}

        wav_path = self._write_wav(b"".join(chunks), f"session{chunk_id}")
        try:
            full_text = (self._run_whisper(wav_path, timeout=120) or "").strip()
        finally:
            wav_path.unlink(missing_ok=True)

        new_text = full_text
        if emitted and full_text.lower().startswith(emitted.lower()):
            new_text = full_text[len(emitted) :].strip()

        with self._transcript_lock:
            self._emitted_transcript = full_text

        return {"chunk_id": chunk_id, "text": new_text, "is_final": mark_final}

    def _empty_chunk(self, chunk_id: int) -> Dict[str, Any]:
        return {"chunk_id": chunk_id, "text": "", "is_final": False}


    # -------------------------- Public API -----------------------

    def submit_chunk(self, audio_chunk: np.ndarray, chunk_id: int) -> Future:

        audio_bytes = np.ascontiguousarray(audio_chunk, dtype=np.int16).tobytes()
        with self._transcript_lock:
            self._chunks.append(audio_bytes)

        if not self.emit_partials:
            return self.executor.submit(self._empty_chunk, chunk_id)
        return self.executor.submit(self._process_chunk_whisper, audio_bytes, chunk_id)

    def finalize(self, chunk_id: int, mark_final: bool = True) -> Optional[Future]:
        return self.executor.submit(self._finalize_whisper, chunk_id, mark_final)

    def reset(self) -> None:
        with self._transcript_lock:
            self._chunks.clear()
            self._emitted_transcript = ""
            self._last_partial = ""

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
        with contextlib.suppress(Exception):
            shutil.rmtree(self._temp_dir, ignore_errors=True)


# ================================================================
# Streaming LLM via llama110
# ================================================================


class StreamingLLM:
    """Accumulate recognized text and asynchronously invoke llama110 when ready."""

    def __init__(self, llama_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.context_buffer: List[str] = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.llama_kwargs = llama_kwargs or {}

        self.llama_kwargs.setdefault("n_predict", 12)
        self.llama_kwargs.setdefault("threads", os.cpu_count() or 4)
        self.llama_kwargs.setdefault("temperature", 0.6)

    def process_incremental(self, text_chunk: str, is_final: bool = False) -> Optional[Future]:
        chunk = (text_chunk or "").strip()
        if chunk:
            self.context_buffer.append(chunk)

        if not self.context_buffer and not chunk and not is_final:
            return None

        should_generate = False
        if is_final:
            should_generate = bool(self.context_buffer)
        elif self._should_respond():
            should_generate = True

        if not should_generate:
            return None

        prompt_text = " ".join(self.context_buffer).strip()
        self.context_buffer.clear()
        if not prompt_text:
            return None

        return self.executor.submit(self._generate_response, prompt_text)

    def _should_respond(self) -> bool:
        current = " ".join(self.context_buffer)
        return any(marker in current for marker in [".", "?", "!"])

    def _generate_response(self, text: str) -> str:
        prompt = f"Answer concisely: {text}".strip()
        call_kwargs = {
            "llama_cli_path": self.llama_kwargs.get("llama_cli_path"),
            "model_path": self.llama_kwargs.get("model_path"),

            "n_predict": self.llama_kwargs.get("n_predict", 12),
            "threads": self.llama_kwargs.get("threads", os.cpu_count() or 4),
            "temperature": self.llama_kwargs.get("temperature", 0.6),

            "sampler": self.llama_kwargs.get("sampler"),
            "tts_after": False,
            "tts_cmd": None,
            "timeout_seconds": self.llama_kwargs.get("timeout_seconds", 240),
        }

        try:
            result = llama110(prompt_text=prompt, **call_kwargs)
        except FileNotFoundError as exc:
            print(f"[LLM] {exc}")
            return "I could not load the local LLM."
        except Exception as exc:
            print(f"[LLM] Error invoking llama110: {exc}")
            return "I encountered an error while thinking about that."

        response = (result or {}).get("generated", "")
        if not response:
            fallback = (result or {}).get("raw_stdout") or ""
            response = fallback.strip()

        return self._clean_response(response)

    @staticmethod
    def _clean_response(response: str) -> str:
        text = (response or "").strip()
        leading_quotes = ('"', "'", "`", "â€œ", "â€", "â€˜", "â€™")
        while text and text[0] in leading_quotes:
            text = text[1:].lstrip()
        if text.startswith("?"):
            # Strip leading question mark artifacts such as ?" or ? 'Hello'
            trimmed = text[1:].lstrip()
            if trimmed and trimmed[0].isalpha():
                text = trimmed
        if text.startswith("\"") and len(text) > 1:
            text = text[1:].lstrip()
        return text


    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)


# ================================================================
# Piper-based TTS with playback buffering
# ================================================================



# ================================================================
# Parallel Voice Assistant Orchestrator
# ================================================================


@dataclass

class PendingOutput:
    timestamp: float
    segments_expected: int
    latency_recorded: bool = False


@dataclass

class PipelineStats:
    stt_chunks: int = 0
    llm_responses: int = 0
    tts_segments: int = 0
    total_latency: float = 0.0
    start_time: float = field(default_factory=time.time)

    stt_latencies: List[float] = field(default_factory=list)
    llm_latencies: List[float] = field(default_factory=list)
    tts_generation_latencies: List[float] = field(default_factory=list)
    input_to_output_latencies: List[float] = field(default_factory=list)
    pending_outputs: Deque[PendingOutput] = field(default_factory=deque)

    recording_stop_time: Optional[float] = None
    recording_to_first_llm_latency: Optional[float] = None
    recording_stop_to_first_tts_latency: Optional[float] = None



class ParallelVoiceAssistant:
    """Coordinate the streaming recorder, STT workers, llama110, and Piper TTS."""

    def __init__(
        self,
        chunk_duration: float = CHUNK_DURATION,
        sample_rate: int = SAMPLE_RATE,
        stt_workers: int = 2,

        whisper_exe: Path = WHISPER_EXE,
        whisper_model: Path = WHISPER_MODEL,
        whisper_threads: Optional[int] = None,
        emit_stt_partials: bool = False,
        piper_model_path: Path = PIPER_MODEL_PATH,
        llama_kwargs: Optional[Dict[str, Any]] = None,

        output_device: Optional[Any] = None,
        playback_cmd: Optional[Iterable[str]] = None,
        use_subprocess_playback: bool = True,
        silence_timeout: float = DEFAULT_SILENCE_TIMEOUT,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
        
        

    ) -> None:
        self._chunk_duration = float(chunk_duration)
        self.recorder = StreamingRecorder(chunk_duration=chunk_duration, sample_rate=sample_rate)
        self.stt = ParallelSTT(
            num_workers=stt_workers,
            sample_rate=sample_rate,
            whisper_exe=whisper_exe,
            whisper_model=whisper_model,
            whisper_threads=whisper_threads,
            emit_partials=emit_stt_partials,
        )
        self.llm = StreamingLLM(llama_kwargs=llama_kwargs)

        self.tts = BufferedTTS(
            model_path=piper_model_path,
            playback_cmd=playback_cmd,
            output_device=output_device,

            use_subprocess=use_subprocess_playback,

            on_playback_start=self._on_tts_playback_start,
            on_playback_error=self._on_tts_playback_error,
        )

        self.stt_futures: "queue.Queue[Tuple[int, Future, float]]" = queue.Queue()
        self.llm_futures: "queue.Queue[Tuple[Future, float, float]]" = queue.Queue()
        self.stats = PipelineStats()
        self._stt_done = threading.Event()
        self._pending_lock = threading.Lock()

        self._activity_lock = threading.Lock()
        self._stop_lock = threading.Lock()

        self._activity_event = threading.Event()
        self._last_voice_time = time.time()
        self._has_detected_speech = False
        self._first_voice_time: Optional[float] = None
        self._recording_stop_time: Optional[float] = None
        self._silence_timeout = float(silence_timeout)
        self._silence_threshold = float(silence_threshold)

        self._stop_requested = False
        self._stop_reason: Optional[str] = None
        self._consecutive_silent_chunks = 0
        self._silent_chunks_before_stop = 2
        
        self._noise_blacklist = {
            "wind blowing",
            "bird chirping",
            "blank_audio",
            "blank audio",
            "[blank_audio]",
            "(wind blowing)",
            "(bird chirping)",
        }

        # optional: regex to catch bracket/parenthesis or small variations
        self._noise_regex = re.compile(
            r"\b(blank[_ ]?audio|wind blowing|bird chirping)\b", flags=re.IGNORECASE
        )

        self._tts_futures_lock = threading.Lock()
        self._pending_tts_futures: Set[Future] = set()

        self._chunk_activity: Dict[int, bool] = {}
        self._awaiting_transcript_chunks = 0
        self._awaiting_transcript_started_at: Optional[float] = None
        self._awaiting_transcript_chunk_limit = max(2, int(math.ceil(4.0 / max(0.1, self._chunk_duration))))
        self._awaiting_transcript_timeout = max(3.0, self._chunk_duration * 2.5)
        self._stt_flush_in_progress = False
        self._next_finalize_id = 1_000_000
        self._active_flush_ids: Set[int] = set()



    def _register_activity(self) -> None:
        now = time.time()
        with self._activity_lock:
            if not self._has_detected_speech:
                self._first_voice_time = now
            self._has_detected_speech = True
            self._last_voice_time = now
        self._activity_event.set()

    def _reset_awaiting_transcript_state(self) -> None:
        self._awaiting_transcript_chunks = 0
        self._awaiting_transcript_started_at = None

    def _should_force_intermediate_transcription(self) -> bool:
        if self._stt_flush_in_progress:
            return False
        if self._awaiting_transcript_chunks >= self._awaiting_transcript_chunk_limit:
            return True
        if self._awaiting_transcript_started_at is not None:
            elapsed = time.time() - self._awaiting_transcript_started_at
            if elapsed >= self._awaiting_transcript_timeout:
                return True
        return False

    def _queue_intermediate_transcription(self, reason: str) -> None:
        if self._stt_flush_in_progress:
            return

        flush_id = self._next_finalize_id
        future = self.stt.finalize(flush_id, mark_final=False)
        if future is None:
            return

        self._stt_flush_in_progress = True
        self._active_flush_ids.add(flush_id)
        self.stt_futures.put((flush_id, future, time.time()))
        print(reason)
        self._reset_awaiting_transcript_state()
        self._next_finalize_id += 1

    def _is_silent_chunk(self, audio_chunk: np.ndarray) -> bool:
        if audio_chunk.size == 0:
            return True
        audio_view = np.asarray(audio_chunk, dtype=np.int16)
        if audio_view.ndim > 1:
            audio_view = audio_view.reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(audio_view.astype(np.float32)))))
        return rms < self._silence_threshold


    def _request_stop(self, reason: str) -> None:
        with self._stop_lock:
            if self._stop_requested:
                return
            self._stop_requested = True
            self._stop_reason = reason
            now = time.time()
            if self._recording_stop_time is None:
                self._recording_stop_time = now
            self.stats.recording_stop_time = self._recording_stop_time
        print(reason)
        self.recorder.stop()
        self.recorder.clear_queue()
        self._activity_event.set()

    def _handle_silent_audio_chunk(self) -> None:
        if self._stop_requested:
            return
        self._consecutive_silent_chunks += 1
        if self._consecutive_silent_chunks < self._silent_chunks_before_stop:
            return
        if self._has_detected_speech:
            message = (
                f"[MAIN] No speech detected for {self._consecutive_silent_chunks} consecutive chunks; stopping recorder."
            )
        else:
            message = (
                f"[MAIN] Silence persisted for {self._consecutive_silent_chunks} consecutive chunks; stopping recorder."
            )
        self._request_stop(message)


    def _reference_timestamp_for_output(self, input_timestamp: float) -> float:
        candidates = [input_timestamp]
        with self._activity_lock:
            candidates.append(self._last_voice_time)
        if self._recording_stop_time is not None:
            candidates.append(self._recording_stop_time)
        return max(candidates)

    def run(self, duration: Optional[float] = None) -> None:
        max_duration = duration if duration and duration > 0 else None
        if max_duration is not None:
            print(
                f"[MAIN] Starting streaming assistant (max {max_duration:.1f}s, "
                f"silence timeout {self._silence_timeout:.1f}s)"
            )
        else:
            print(f"[MAIN] Starting streaming assistant (silence timeout {self._silence_timeout:.1f}s)")

        start_time = time.time()
        self.stats.start_time = start_time
        self.stats.recording_stop_time = None
        self.stats.recording_to_first_llm_latency = None

        self.stats.recording_stop_to_first_tts_latency = None

        self._stt_done.clear()
        self._activity_event.clear()
        with self._activity_lock:
            self._has_detected_speech = False
            self._last_voice_time = start_time
            self._first_voice_time = None
        self._recording_stop_time = None

        with self._tts_futures_lock:
            self._pending_tts_futures.clear()

        with self._stop_lock:
            self._stop_requested = False
            self._stop_reason = None
        self._consecutive_silent_chunks = 0
        self._reset_awaiting_transcript_state()
        self._stt_flush_in_progress = False
        self._active_flush_ids.clear()
        self._next_finalize_id = 1_000_000


        self.recorder.start()
        self.tts.start_playback()

        stt_thread = threading.Thread(target=self._stt_pipeline, name="STTPipeline", daemon=True)
        llm_thread = threading.Thread(target=self._llm_pipeline, name="LLMPipeline", daemon=True)

        stt_thread.start()
        llm_thread.start()

        try:

            while True:
                if self._stop_requested:
                    break
                now = time.time()
                if max_duration is not None and now - start_time >= max_duration:
                    self._request_stop(
                        f"[MAIN] Max duration {max_duration:.1f}s reached; wrapping up."
                    )

                    break

                with self._activity_lock:
                    has_voice = self._has_detected_speech
                    last_voice = self._last_voice_time

                if has_voice and (now - last_voice) >= self._silence_timeout:

                    self._request_stop(
                        f"[MAIN] Detected {self._silence_timeout:.1f}s of silence; stopping recorder."
                    )

                    break

                wait_timeout = 0.5
                if has_voice:
                    remaining = self._silence_timeout - (now - last_voice)
                    wait_timeout = max(0.1, min(0.5, remaining))

                triggered = self._activity_event.wait(timeout=wait_timeout)
                if triggered:
                    self._activity_event.clear()

        except KeyboardInterrupt:
            print("\n[MAIN] Interrupted by user")
            self._request_stop("[MAIN] Interrupted by user; stopping recorder.")
        finally:
            self.recorder.stop()
            if self._recording_stop_time is None:
                self._recording_stop_time = time.time()

            self.stats.recording_stop_time = self._recording_stop_time

        stt_thread.join(timeout=5.0)

        finalize_future = self.stt.finalize(self.stats.stt_chunks + 1)
        if finalize_future is not None:

            self.stt_futures.put((self.stats.stt_chunks + 1, finalize_future, time.time()))

            self._process_stt_results(wait=True)

        # Signal the LLM pipeline that no more text is coming once final STT results are queued.
        self._stt_done.set()
        llm_thread.join()

        self.stt.shutdown()
        self.llm.shutdown()
        self._wait_for_tts_completion()
        self.tts.stop()

        elapsed = time.time() - start_time
        self._print_stats(elapsed)

    # --------------------------- Pipelines -----------------------

    def _stt_pipeline(self) -> None:
        chunk_id = 0
        try:
            while self.recorder.recording or not self.recorder.chunk_queue.empty():
                audio_chunk = self.recorder.get_chunk(timeout=0.5)
                if audio_chunk is None:
                    self._process_stt_results(wait=False)
                    continue

                if self._stop_requested and not self.recorder.recording:
                    self.recorder.clear_queue()
                    self._process_stt_results(wait=False)
                    break

                # remember recorder sample rate for VAD logic if needed
                setattr(self, "_recorder_sample_rate", self.recorder.sample_rate)

                # Decide whether this chunk is silent/noisy for logging, but DO NOT call
                # _handle_silent_audio_chunk() here. We wait for the STT result so we
                # only count a chunk as "silent" once the model actually returns nothing
                # useful for that chunk (avoids double-counting).
                is_silent = self._is_silent_chunk(audio_chunk)
                self._chunk_activity[chunk_id] = not is_silent
                if is_silent:
                    # don't mark stop here; just log and continue to submit to STT so
                    # the model can confirm whether it's empty/noise
                    # (This prevents short/quiet speech from being mis-classified.)
                    # Optional: print RMS for debugging:
                    try:
                        audio_view = np.asarray(audio_chunk, dtype=np.int16)
                        if audio_view.ndim > 1:
                            audio_view = audio_view.reshape(-1)
                        rms = float(np.sqrt(np.mean(np.square(audio_view.astype(np.float32)))))
                    except Exception:
                        rms = 0.0
                    print(f"[STT] Chunk {chunk_id}: low energy (RMS {rms:.1f}), submitting to STT for verification")
                else:
                    # Defer activity tracking until Whisper confirms actual text for
                    # this chunk. High-energy noise without a transcript shouldn't
                    # refresh the silence timeout window.
                    pass

                # Submit to STT as usual (we rely on _process_stt_results to treat
                # empty/noise transcriptions as silent and call _handle_silent_audio_chunk()).
                future = self.stt.submit_chunk(audio_chunk, chunk_id)
                self.stt_futures.put((chunk_id, future, time.time()))

                self.stats.stt_chunks += 1
                chunk_id += 1

                self._process_stt_results(wait=False)

            self._process_stt_results(wait=True)
        finally:
            # Ensure any exceptions don't leave futures undispatched.
            pass

    def _process_stt_results(self, wait: bool) -> None:

        pending: List[Tuple[int, Future, float]] = []
        while not self.stt_futures.empty():
            chunk_id, future, start_time = self.stt_futures.get()

            if wait or future.done():
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"[STT Pipeline] Future for chunk {chunk_id} failed: {exc}")
                    continue
            else:

                pending.append((chunk_id, future, start_time))

                continue

            if not result:
                continue

            res_chunk_id = result.get("chunk_id", chunk_id)
            if chunk_id in self._active_flush_ids or res_chunk_id in self._active_flush_ids:
                self._active_flush_ids.discard(chunk_id)
                self._active_flush_ids.discard(res_chunk_id)
                self._stt_flush_in_progress = False


            latency = max(0.0, time.time() - start_time)
            self.stats.stt_latencies.append(latency)


            text = (result.get("text") or "").strip()
            is_final = bool(result.get("is_final"))

            # Normalize for noise checks
            normalized = (text or "").strip().lower()
            had_activity = self._chunk_activity.pop(res_chunk_id, False)

            # Check blacklist exact matches first, then regex for variants
            is_noise = False
            if normalized:
                if normalized in self._noise_blacklist:
                    is_noise = True
                elif self._noise_regex.search(normalized):
                    is_noise = True

            if is_noise or not normalized:
                if had_activity and not normalized:
                    # Speech energy was observed for this chunk, but Whisper did not

                    # return any transcript yet. Treat as ongoing speech for a short
                    # period, but force an intermediate transcription if it persists.

                    print(
                        f"[STT] Chunk {res_chunk_id}: (speech detected, awaiting transcription)"
                    )

                    self._awaiting_transcript_chunks += 1
                    if self._awaiting_transcript_started_at is None:
                        self._awaiting_transcript_started_at = time.time()
                    if self._awaiting_transcript_chunks >= self._silent_chunks_before_stop:
                        # Ambient noise can keep RMS high while Whisper emits nothing; treat
                        # this as silence so we still stop after the configured no-speech chunks.
                        self._reset_awaiting_transcript_state()
                        self._handle_silent_audio_chunk()
                        continue
                    if self._should_force_intermediate_transcription():
                        elapsed = 0.0
                        if self._awaiting_transcript_started_at is not None:
                            elapsed = time.time() - self._awaiting_transcript_started_at
                        self._queue_intermediate_transcription(
                            f"[STT] Forcing intermediate transcription after {elapsed:.1f}s without text"
                        )

                    continue

                # Treat as silent/noise: increment silent-chunk logic and DO NOT feed to LLM
                print(f"[STT] Chunk {res_chunk_id}: {text} (treated as noise/empty)")
                self._reset_awaiting_transcript_state()
                self._handle_silent_audio_chunk()
                # We still want to surface the log, but skip registering activity and LLM trigger
                # continue to next future
                continue

            # Otherwise it's valid speech
            self._reset_awaiting_transcript_state()
            self._register_activity()
            self._consecutive_silent_chunks = 0

            print(f"[STT] Chunk {res_chunk_id}: {text}")



            llm_trigger_time = time.time()
            llm_future = self.llm.process_incremental(text, is_final=is_final)
            if llm_future is not None:
                reference_time = self._reference_timestamp_for_output(llm_trigger_time)
                self.llm_futures.put((llm_future, llm_trigger_time, reference_time))


        for item in pending:
            self.stt_futures.put(item)

    def _llm_pipeline(self) -> None:
        segment_id = 0
        while not self._stt_done.is_set() or not self.llm_futures.empty():
            try:

                llm_future, _submit_time, reference_timestamp = self.llm_futures.get(timeout=0.5)

            except queue.Empty:
                continue

            response = ""
            try:
                response = llm_future.result(timeout=300)

                response_ready_time = time.time()

            except Exception as exc:
                print(f"[LLM Pipeline] Error: {exc}")
                continue


            latency = max(0.0, response_ready_time - reference_timestamp)

            self.stats.llm_latencies.append(latency)

            if self.stats.recording_to_first_llm_latency is None:
                first_voice_time: Optional[float]
                with self._activity_lock:
                    first_voice_time = self._first_voice_time

                if first_voice_time is not None:
                    self.stats.recording_to_first_llm_latency = max(
                        0.0, response_ready_time - first_voice_time
                    )
                elif self._recording_stop_time is not None:
                    self.stats.recording_to_first_llm_latency = max(
                        0.0, response_ready_time - self._recording_stop_time
                    )
                else:
                    self.stats.recording_to_first_llm_latency = max(
                        0.0, response_ready_time - reference_timestamp
                    )


            response = (response or "").strip()
            if not response:
                continue

            print(f"[LLM] Response: {response[:120]}{'...' if len(response) > 120 else ''}")
            self.stats.llm_responses += 1


            sentences = self._split_sentences(response)
            if not sentences:
                sentences = [response]

            tts_jobs: List[Tuple[Future, float]] = []
            for sentence in sentences:
                submit_time = time.time()
                future = self.tts.generate_and_queue(sentence, segment_id)
                if future is not None:
                    self.stats.tts_segments += 1
                    with self._tts_futures_lock:
                        self._pending_tts_futures.add(future)
                    tts_jobs.append((future, submit_time))
                segment_id += 1

            if not tts_jobs:
                continue


            pending_timestamp = self._reference_timestamp_for_output(reference_timestamp)
            pending = PendingOutput(timestamp=pending_timestamp, segments_expected=len(tts_jobs))

            with self._pending_lock:
                self.stats.pending_outputs.append(pending)

            for future, submit_time in tts_jobs:
                future.add_done_callback(
                    lambda fut, start=submit_time, pending_ref=pending: self._on_tts_generated(fut, start, pending_ref)
                )

    def _on_tts_generated(self, future: Future, start_time: float, pending: PendingOutput) -> None:
        try:
            result = future.result()
        except Exception as exc:
            print(f"[TTS Pipeline] Generation failed: {exc}")
            self._handle_failed_tts_generation(pending)
        else:
            if result:
                latency = max(0.0, time.time() - start_time)
                self.stats.tts_generation_latencies.append(latency)
            else:
                self._handle_failed_tts_generation(pending)
        finally:
            with self._tts_futures_lock:
                self._pending_tts_futures.discard(future)

    def _handle_failed_tts_generation(self, pending: PendingOutput) -> None:
        with self._pending_lock:
            pending.segments_expected = max(0, pending.segments_expected - 1)
            if pending.segments_expected == 0:
                try:
                    self.stats.pending_outputs.remove(pending)
                except ValueError:
                    pass

    def _on_tts_playback_start(self, file_path: str, started_at: float) -> None:

        if (
            self.stats.recording_stop_to_first_tts_latency is None
            and self._recording_stop_time is not None
        ):
            self.stats.recording_stop_to_first_tts_latency = max(
                0.0, started_at - self._recording_stop_time
            )

        with self._pending_lock:
            while self.stats.pending_outputs:
                pending = self.stats.pending_outputs[0]
                if pending.segments_expected <= 0:
                    self.stats.pending_outputs.popleft()
                    continue

                if not pending.latency_recorded:
                    latency = max(0.0, started_at - pending.timestamp)
                    self.stats.input_to_output_latencies.append(latency)
                    pending.latency_recorded = True

                pending.segments_expected = max(0, pending.segments_expected - 1)
                if pending.segments_expected == 0:
                    self.stats.pending_outputs.popleft()
                break
        try:
            if getattr(self, "recorder", None) and getattr(self.recorder, "recording", False):
                self._request_stop("[MAIN] Stopping recorder during TTS to avoid feedback.")
        except Exception as e:
            self._log(f"[TTS] playback_start hook error: {e}")

    def _on_tts_playback_error(self) -> None:
        with self._pending_lock:
            if not self.stats.pending_outputs:
                return

            pending = self.stats.pending_outputs[0]
            pending.segments_expected = max(0, pending.segments_expected - 1)
            if pending.segments_expected == 0:
                self.stats.pending_outputs.popleft()

    # --------------------------- Helpers -------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences: List[str] = []
        current: List[str] = []
        for word in text.split():
            current.append(word)
            if any(word.endswith(p) for p in [".", "!", "?", ","]):
                sentences.append(" ".join(current))
                current = []
        if current:
            sentences.append(" ".join(current))
        return sentences


    @staticmethod
    def _print_latency_summary(label: str, samples: List[float]) -> None:
        if not samples:
            print(f"{label}: n/a")
            return

        avg = sum(samples) / len(samples)
        min_val = min(samples)
        max_val = max(samples)
        print(f"{label}: avg {avg:.2f}s (min {min_val:.2f}s, max {max_val:.2f}s, n={len(samples)})")


    def _print_stats(self, elapsed: float) -> None:
        print("\n--- PIPELINE STATS ---")
        print(f"Runtime: {elapsed:.2f}s")
        print(f"STT chunks processed: {self.stats.stt_chunks}")
        print(f"LLM responses generated: {self.stats.llm_responses}")
        print(f"TTS segments queued: {self.stats.tts_segments}")

        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Memory usage: {mem_mb:.1f} MB")

        self._print_latency_summary("STT chunk latency", list(self.stats.stt_latencies))
        self._print_latency_summary("LLM response latency", list(self.stats.llm_latencies))
        if self.stats.recording_to_first_llm_latency is not None:
            print(
                "Recording -> first LLM response: "
                f"{self.stats.recording_to_first_llm_latency:.2f}s"
            )
        else:
            print("Recording -> first LLM response: n/a")

        if self.stats.recording_stop_to_first_tts_latency is not None:
            print(
                "Recording stop -> first TTS audio: "
                f"{self.stats.recording_stop_to_first_tts_latency:.2f}s"
            )
        else:
            print("Recording stop -> first TTS audio: n/a")
        self._print_latency_summary("TTS generation latency", list(self.stats.tts_generation_latencies))
        self._print_latency_summary("Input -> first audio gap", list(self.stats.input_to_output_latencies))

        print("----------------------\n")

    def _wait_for_tts_completion(self, timeout: float = 15.0) -> None:
        if self.stats.tts_segments == 0:
            return

        deadline = time.time() + max(0.0, timeout)
        while True:
            with self._tts_futures_lock:
                pending_futures = len(self._pending_tts_futures)

            with self._pending_lock:
                pending_outputs = sum(
                    1 for pending in self.stats.pending_outputs if pending.segments_expected > 0
                )

            queue_empty = self.tts.speech_queue.empty()

            if pending_futures == 0 and pending_outputs == 0 and queue_empty:
                break

            if time.time() >= deadline:
                print("[TTS] Timeout waiting for pending audio playback; continuing shutdown.")
                break

            time.sleep(0.05)


# ================================================================
# Model Warm-up Helpers
# ================================================================


class ModelPreloader:
    """Utility helpers to warm up the local models so first inference is faster."""

    @staticmethod

    def warmup_whisper(
        whisper_exe: Path = WHISPER_EXE,
        whisper_model: Path = WHISPER_MODEL,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        print("[WARMUP] Priming whisper.cpp...")
        exe = Path(whisper_exe)
        model = Path(whisper_model)
        if not exe.exists():
            print(f"[WARMUP] Whisper binary missing at {exe}")
            return
        if not model.exists():
            print(f"[WARMUP] Whisper model missing at {model}")
            return

        tmp_dir = Path(tempfile.mkdtemp(prefix="whisper_warmup_"))
        wav_path = tmp_dir / "warmup.wav"
        success = True
        try:
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(b"\x00" * sample_rate)
            cmd = [
                str(exe),
                "-m",
                str(model),
                "-f",
                str(wav_path),
                "--no-prints",
                "--output-txt",
                "-t",
                "1",
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
        except Exception as exc:
            print(f"[WARMUP] Whisper warm-up failed: {exc}")
            success = False
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if success:
            print("[WARMUP] whisper.cpp ready")


    @staticmethod
    def warmup_llama(llama_kwargs: Optional[Dict[str, Any]] = None) -> None:
        print("[WARMUP] Warming up llama110...")
        kwargs = llama_kwargs or {}
        try:
            llama110(
                prompt_text="Hello",
                llama_cli_path=kwargs.get("llama_cli_path"),
                model_path=kwargs.get("model_path"),
                n_predict=8,
                threads=kwargs.get("threads", os.cpu_count() or 4),
                temperature=kwargs.get("temperature", 0.5),
                sampler=kwargs.get("sampler"),
                timeout_seconds=kwargs.get("timeout_seconds", 120),
            )
        except Exception as exc:
            print(f"[WARMUP] llama110 warm-up failed: {exc}")
        else:
            print("[WARMUP] llama110 ready")

    @staticmethod
    def warmup_piper(model_path: Path = PIPER_MODEL_PATH) -> None:
        print("[WARMUP] Warming up Piper...")
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"[WARMUP] Piper model missing at {model_path}")
            return
        cmd = ["piper", "-m", str(model_path), "--output_file", "/tmp/piper_warmup.wav"]
        try:
            subprocess.run(cmd, input="Warm up".encode("utf-8"), check=True, timeout=10)
        except Exception as exc:
            print(f"[WARMUP] Piper warm-up failed: {exc}")
        else:
            Path("/tmp/piper_warmup.wav").unlink(missing_ok=True)
            print("[WARMUP] Piper ready")


# ================================================================
# CLI entry point
# ================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming voice assistant pipeline")

    parser.add_argument("--duration", type=float, default=30.0, help="How long to run the streaming demo")
    parser.add_argument("--warmup", action="store_true", help="Run model warm-up steps before streaming")
    parser.add_argument("--piper-model", type=Path, default=PIPER_MODEL_PATH, help="Path to Piper .onnx model")
    parser.add_argument(
        "--whisper-cli",
        type=Path,
        default=WHISPER_EXE,
        help="Path to whisper.cpp CLI binary",
    )
    parser.add_argument(
        "--whisper-model",
        type=Path,
        default=WHISPER_MODEL,
        help="Path to whisper.cpp model",
    )
    parser.add_argument(
        "--whisper-threads",
        type=int,
        default=os.cpu_count() or 2,
        help="Threads to dedicate to whisper.cpp",
    )
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="Threads to pass to llama110")
    parser.add_argument("--n-predict", type=int, default=12, help="Tokens to generate with llama110")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature for llama110")
    parser.add_argument("--llama-cli", type=Path, default=None, help="Optional override for llama-cli path")
    parser.add_argument("--llama-model", type=Path, default=None, help="Optional override for llama model path")

    parser.add_argument("--output-device", type=str, default=None, help="sounddevice output (index or name) for Piper playback")
    parser.add_argument("--playback-cmd", nargs="+", default=None, help="Fallback playback command for Piper audio")
    parser.add_argument(
        "--force-subprocess-playback",
        action="store_true",

        help="Always use the playback command for Piper audio (default)",
    )
    parser.add_argument(
        "--direct-playback",
        action="store_true",
        help="Play Piper audio through sounddevice instead of piping to the CLI player",
    )

    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=DEFAULT_SILENCE_TIMEOUT,
        help="Seconds of silence before automatically stopping the recorder",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=DEFAULT_SILENCE_THRESHOLD,
        help="RMS amplitude threshold (int16) to treat chunks as silence",
    )

    parser.add_argument(
        "--enable-stt-partials",
        action="store_true",
        help="Run whisper.cpp on each chunk for incremental transcripts",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    llama_kwargs = {
        "threads": args.threads,
        "n_predict": args.n_predict,
        "temperature": args.temperature,
    }
    if args.llama_cli:
        llama_kwargs["llama_cli_path"] = str(args.llama_cli)
    if args.llama_model:
        llama_kwargs["model_path"] = str(args.llama_model)


    if args.warmup:
        ModelPreloader.warmup_whisper(args.whisper_cli, args.whisper_model)
        ModelPreloader.warmup_llama(llama_kwargs)
        ModelPreloader.warmup_piper(args.piper_model)

    use_subprocess_playback = True
    if args.direct_playback:
        use_subprocess_playback = False
    elif args.force_subprocess_playback:
        use_subprocess_playback = True


    assistant = ParallelVoiceAssistant(
        chunk_duration=CHUNK_DURATION,
        sample_rate=SAMPLE_RATE,
        stt_workers=2,

        whisper_exe=args.whisper_cli,
        whisper_model=args.whisper_model,
        whisper_threads=args.whisper_threads,
        emit_stt_partials=args.enable_stt_partials,
        piper_model_path=args.piper_model,
        llama_kwargs=llama_kwargs,

        output_device=args.output_device,
        playback_cmd=args.playback_cmd,
        use_subprocess_playback=use_subprocess_playback,

        silence_timeout=args.silence_timeout,
        silence_threshold=args.silence_threshold,
    )
    max_duration = args.duration if args.duration and args.duration > 0 else None
    assistant.run(duration=max_duration)



if __name__ == "__main__":
    main()