from __future__ import annotations
import re
import argparse

import contextlib
import json
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

# ================================================================
# Streaming recorder
# ================================================================

class StreamingRecorder:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_duration: float = CHUNK_DURATION,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
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
    ) -> None:
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.whisper_exe = whisper_exe
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def submit_chunk(self, audio_chunk: np.ndarray, chunk_id: int) -> Future:
        return self.executor.submit(self._run_whisper, audio_chunk, chunk_id)

    def _run_whisper(self, audio_chunk: np.ndarray, chunk_id: int) -> Dict[str, Any]:
        # Placeholder implementation; replace with real whisper invocation.
        return {"text": "", "is_final": True, "chunk_id": chunk_id}

    def finalize(self, next_chunk_id: int) -> Optional[Future]:
        return None

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)


# ================================================================
# Streaming LLM (placeholder)
# ================================================================

class StreamingLLM:
    def __init__(self, llama_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1)

    def process_incremental(self, text: str, is_final: bool = False) -> Optional[Future]:
        return self.executor.submit(lambda: "")

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)


# ================================================================
# Piper-based TTS with playback buffering
# ================================================================

@dataclass
class SpeechSegment:
    path: str
    raw: bytes
    sample_rate: int
    channels: int = 1
    sampwidth: int = 2
    text: str = ""


@dataclass
class PiperVoiceInfo:
    sample_rate: int = 22050
    speaker_id: Optional[int] = None
    channels: int = 1
    metadata_path: Optional[Path] = None


class BufferedTTS:
    """Generate speech with Piper asynchronously and stream playback via a CLI player."""

    def __init__(
        self,
        model_path: Path,
        playback_cmd: Optional[Iterable[str]] = None,
        output_device: Optional[Any] = None,
        use_subprocess: bool = False,
        on_playback_start: Optional[Callable[[str, float], None]] = None,
        on_playback_error: Optional[Callable[[], None]] = None,
        timeout: int = 30,
    ) -> None:
        self.model_path = model_path
        self.playback_cmd = playback_cmd
        self.output_device = output_device
        self.use_subprocess = use_subprocess
        self.on_playback_start = on_playback_start
        self.on_playback_error = on_playback_error
        self.timeout = timeout

        self._voice_info = PiperVoiceInfo()
        self.speech_queue: "queue.Queue[SpeechSegment]" = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._playback_env = os.environ.copy()
        self.playing = False
        self._playback_thread: Optional[threading.Thread] = None

    def start_playback(self) -> None:
        self.playing = True
        if not self._playback_thread:
            self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self._playback_thread.start()

    def _playback_loop(self) -> None:
        while self.playing:
            try:
                segment: SpeechSegment = self.speech_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._play_segment(segment)
            except Exception:
                pass

    def _play_segment(self, segment: SpeechSegment) -> bool:
        cmd = self.playback_cmd or ["aplay", "-q"]
        try:
            if cmd and cmd[-1] == "-" and segment.raw is not None:
                subprocess.run(
                    cmd,
                    input=segment.raw,
                    check=True,
                    env=self._playback_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.run(cmd + [segment.path], check=True, env=self._playback_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[TTS] Subprocess playback failed (exit {exc.returncode}) for {segment.path}: {exc}")
            return False
        except Exception as exc:
            print(f"[TTS] Subprocess playback failed for {segment.path}: {exc}")
            return False

    def generate_and_queue(self, text: str, segment_id: int) -> Optional[Future]:
        clean_text = " ".join((text or "").split())
        if not clean_text:
            return None
        return self.executor.submit(self._generate_speech, clean_text, segment_id)

    def _generate_speech(self, text: str, segment_id: int) -> Optional[SpeechSegment]:
        if not self.model_path.exists():
            print(f"[TTS] Piper model not found: {self.model_path}")
            return None

        utterance = " ".join((text or "").split())
        if not utterance:
            return None

        info = self._voice_info
        cmd = ["piper", "-m", str(self.model_path), "--output-raw"]
        if info.speaker_id is not None:
            cmd += ["--speaker", str(info.speaker_id)]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        keep_file = False
        try:
            input_bytes = (utterance + "\n").encode("utf-8")
            proc = subprocess.run(
                cmd,
                input=input_bytes,
                capture_output=True,
                check=True,
                timeout=self.timeout,
            )
            audio_bytes = proc.stdout
            if self._looks_like_text(audio_bytes):
                preview = audio_bytes[:120].decode("utf-8", errors="replace")
                print(
                    "[TTS] Piper returned textual output instead of audio; "
                    f"got: {preview!r}"
                )
                return None
            if not audio_bytes:
                print("[TTS] Piper returned no audio data")
                return None

            with wave.open(str(tmp_path), "wb") as wf:
                wf.setnchannels(info.channels or 1)
                wf.setsampwidth(2)
                wf.setframerate(info.sample_rate or 22050)
                wf.writeframes(audio_bytes)
            keep_file = True

            sample_rate = info.sample_rate or 22050
            segment = SpeechSegment(
                path=str(tmp_path),
                raw=audio_bytes,
                sample_rate=sample_rate,
                channels=info.channels or 1,
                text=utterance,
            )
            self.speech_queue.put(segment)
            return segment

        except subprocess.CalledProcessError as exc:
            print(f"[TTS] Piper returned error: {exc}")
        except Exception as exc:
            print(f"[TTS] Piper failed: {exc}")
        finally:
            if not keep_file and tmp_path.exists():
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return None

    @staticmethod
    def _looks_like_text(payload: bytes) -> bool:
        """Heuristic check to detect when Piper prints text instead of PCM."""

        if not payload:
            return False

        sample = payload[:64]
        printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in sample)
        # Random PCM rarely decodes into predominantly printable ASCII. Treat a
        # mostly printable prefix as an indication that Piper emitted text/logs.
        return printable >= max(10, len(sample) * 0.6)

    def stop(self) -> None:
        self.playing = False
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None
        self.executor.shutdown(wait=False)


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
    stt_latencies: List[float] = field(default_factory=list)
    llm_latencies: List[float] = field(default_factory=list)
    tts_generation_latencies: List[float] = field(default_factory=list)
    pending_outputs: Deque[PendingOutput] = field(default_factory=deque)
    start_time: Optional[float] = None
    recording_stop_time: Optional[float] = None
    recording_to_first_llm_latency: Optional[float] = None
    recording_stop_to_first_tts_latency: Optional[float] = None
    tts_segments: int = 0
    input_to_output_latencies: List[float] = field(default_factory=list)
    llm_responses: int = 0


class ParallelVoiceAssistant:
    def __init__(
        self,
        piper_model_path: Path,
        playback_cmd: Optional[Iterable[str]] = None,
        output_device: Optional[Any] = None,
        use_subprocess_playback: bool = False,
        llama_kwargs: Optional[Dict[str, Any]] = None,
        silence_timeout: float = DEFAULT_SILENCE_TIMEOUT,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
    ) -> None:
        self.recorder = StreamingRecorder()
        self.stt = ParallelSTT()
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

    def _register_activity(self) -> None:
        now = time.time()
        with self._activity_lock:
            if not self._has_detected_speech:
                self._first_voice_time = now
            self._has_detected_speech = True
            self._last_voice_time = now
        self._activity_event.set()

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
                time.sleep(0.01)
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
                    try:
                        audio_view = np.asarray(audio_chunk, dtype=np.int16)
                        if audio_view.ndim > 1:
                            audio_view = audio_view.reshape(-1)
                        rms = float(np.sqrt(np.mean(np.square(audio_view.astype(np.float32)))))
                    except Exception:
                        rms = 0.0
                    print(f"[STT] Chunk {chunk_id}: low energy (RMS {rms:.1f}), submitting to STT for verification")
                else:
                    self._register_activity()
                    self._consecutive_silent_chunks = 0

                # Submit to STT as usual (we rely on _process_stt_results to treat
                # empty/noise transcriptions as silent and call _handle_silent_audio_chunk()).
                future = self.stt.submit_chunk(audio_chunk, chunk_id)
                self.stt_futures.put((chunk_id, future, time.time()))

                self.stats.stt_chunks += 1
                chunk_id += 1

                self._process_stt_results(wait=False)

            self._process_stt_results(wait=True)
        finally:
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

            latency = max(0.0, time.time() - start_time)
            self.stats.stt_latencies.append(latency)

            text = (result.get("text") or "").strip()
            is_final = bool(result.get("is_final"))
            res_chunk_id = result.get("chunk_id", chunk_id)

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
                    # return any transcript yet. Treat as ongoing speech so we don't
                    # prematurely trigger silence handling.
                    print(
                        f"[STT] Chunk {res_chunk_id}: (speech detected, awaiting transcription)"
                    )
                    self._consecutive_silent_chunks = 0
                    continue

                # Treat as silent/noise: increment silent-chunk logic and DO NOT feed to LLM
                print(f"[STT] Chunk {res_chunk_id}: {text} (treated as noise/empty)")
                self._handle_silent_audio_chunk()
                continue

            # Otherwise it's valid speech
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
            # Only run this when no exception was raised
            if result:
                latency = max(0.0, time.time() - start_time)
                self.stats.tts_generation_latencies.append(latency)
            else:
                self._handle_failed_tts_generation(pending)
        finally:
            # Always remove the future from the pending set (even on errors)
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
                else:
                    break

    def _split_sentences(self, text: str) -> List[str]:
        # naive sentence splitter
        import re
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _print_stats(self, elapsed: float) -> None:
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

    def _print_latency_summary(self, name: str, values: List[float]) -> None:
        if not values:
            print(f"{name}: n/a")
            return
        values_sorted = sorted(values)
        p50 = values_sorted[len(values_sorted)//2]
        p90 = values_sorted[int(len(values_sorted)*0.9)]
        print(f"{name}: p50={p50:.3f}s p90={p90:.3f}s (n={len(values_sorted)})")

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
