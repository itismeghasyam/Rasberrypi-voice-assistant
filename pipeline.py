"""High-level streaming voice assistant pipeline with async Vosk STT, llama110 LLM, and Piper TTS.

This module consolidates the experimental streaming helpers that previously lived in
``voice_parallel.py`` while switching the speech-to-text stage to Vosk, the language model
stage to the ``llama110`` helper from :mod:`voice_test`, and the text-to-speech stage to
Piper.  The design mirrors the old parallel prototype: audio is captured continuously in
background threads, fed to an asynchronous STT worker pool, passed through a streaming LLM
interface, and finally voiced through a buffered TTS player.

The default configuration expects the following local assets:

* Vosk model directory: ``~/models/vosk-model-small-en-us-0.15``
* ``llama.cpp`` binary and llama110 model (see :func:`voice_test.llama110`)
* Piper voice model, defaulting to ``~/Rasberrypi-voice-assistant/voices/en_US-amy-medium.onnx``

All components are exposed as classes so they can be reused or extended individually.  Run
``python pipeline.py --help`` for a small CLI wrapper around :class:`ParallelVoiceAssistant`.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess

import tempfile
import threading
import time
import wave

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple



import numpy as np
import psutil
import sounddevice as sd
from concurrent.futures import Future, ThreadPoolExecutor

from vosk import KaldiRecognizer, Model as VoskModel

from voice_test import llama110

# ================================================================
# Configuration
# ================================================================
PROJECT_DIR = Path.cwd()
RECORDED_WAV = PROJECT_DIR / "recorded.wav"
SAMPLE_RATE = 16000

CHUNK_DURATION = 2.0  # seconds
DEFAULT_SILENCE_TIMEOUT = 10.0  # seconds of inactivity before auto-stopping
DEFAULT_SILENCE_THRESHOLD = 500.0  # RMS amplitude threshold for silence detection


WHISPER_EXE = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = Path.home() / "whisper.cpp" / "models" / "ggml-tiny.bin"

VOSK_MODEL_DIR = Path.home() / "models" / "vosk-model-small-en-us-0.15"
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

    def stop(self) -> None:
        """Signal the recorder to stop and wait for the background thread."""

        self.recording = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None


# ================================================================
# Parallel Speech-to-Text (Vosk default, Whisper fallback)
# ================================================================


class ParallelSTT:
    """Process audio chunks asynchronously using Vosk or Whisper."""

    def __init__(
        self,
        num_workers: int = 2,
        engine: str = "vosk",
        sample_rate: int = SAMPLE_RATE,
        vosk_model_dir: Path = VOSK_MODEL_DIR,
    ) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max(1, num_workers))
        self.engine = engine.lower()
        self.sample_rate = int(sample_rate)
        self.vosk_model_dir = Path(vosk_model_dir)

        self._last_partial_text = ""
        self._filler_tokens = {"huh", "uh", "um", "hmm", "h", "uhh"}


        if self.engine == "vosk":
            if not self.vosk_model_dir.exists():
                raise FileNotFoundError(
                    f"Vosk model directory not found: {self.vosk_model_dir}. "
                    "Download the model and place it at the configured path."
                )
            self.vosk_model = VoskModel(str(self.vosk_model_dir))
            self._recognizer_lock = threading.Lock()
            self._recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
        else:
            self.vosk_model = None
            self._recognizer_lock = threading.Lock()
            self._recognizer = None

        self.whisper_exe = WHISPER_EXE
        self.whisper_model = WHISPER_MODEL

    # ---------------------- Vosk processing ----------------------

    def _process_chunk_vosk(self, audio_chunk: np.ndarray, chunk_id: int, is_final: bool = False) -> Dict[str, Any]:
        audio_int16 = np.ascontiguousarray(audio_chunk, dtype=np.int16)
        audio_bytes = audio_int16.tobytes()

        with self._recognizer_lock:
            recognizer = self._recognizer
            if recognizer is None:
                recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
                self._recognizer = recognizer

            text = ""
            try:
                if recognizer.AcceptWaveform(audio_bytes):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")

                    self._last_partial_text = ""
                else:
                    partial = json.loads(recognizer.PartialResult())
                    text = partial.get("partial", "")
                    normalized = text.strip().lower()
                    if (
                        not normalized
                        or normalized == self._last_partial_text
                        or (len(normalized) < 4 and " " not in normalized)
                        or normalized in self._filler_tokens
                    ):
                        text = ""
                    else:
                        self._last_partial_text = normalized


                if is_final:
                    final = json.loads(recognizer.FinalResult())
                    final_text = final.get("text", "")
                    if final_text:
                        # Append any remaining text that FinalResult surfaced
                        text = (f"{text} {final_text}" if text else final_text).strip()
                    # Reset recognizer for the next session
                    self._recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)

                    self._last_partial_text = ""

            except Exception as exc:
                print(f"[STT][Vosk] Error decoding chunk {chunk_id}: {exc}")
                text = ""

        return {"chunk_id": chunk_id, "text": text.strip(), "is_final": is_final}

    def _finalize_vosk(self, chunk_id: int) -> Dict[str, Any]:
        with self._recognizer_lock:
            recognizer = self._recognizer
            text = ""
            if recognizer is not None:
                try:
                    final = json.loads(recognizer.FinalResult())
                    text = final.get("text", "")
                except Exception as exc:
                    print(f"[STT][Vosk] Error during finalization: {exc}")
                finally:
                    self._recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)

                    self._last_partial_text = ""

        return {"chunk_id": chunk_id, "text": text.strip(), "is_final": True}

    # --------------------- Whisper processing --------------------

    def _process_chunk_whisper(self, audio_chunk: np.ndarray, chunk_id: int) -> Dict[str, Any]:
        import tempfile

        audio_int16 = np.ascontiguousarray(audio_chunk, dtype=np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = Path(tmp_wav.name)
            with wave.open(tmp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())

        cmd = [
            str(self.whisper_exe),
            "-m",
            str(self.whisper_model),
            "-f",
            str(wav_path),
            "--no-prints",
            "--output-txt",
            "-t",
            "2",
        ]
        text = ""
        try:

            subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            txt_path = wav_path.with_suffix(".txt")
            alt_txt = Path(str(wav_path) + ".txt")
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
                txt_path.unlink(missing_ok=True)
            elif alt_txt.exists():
                text = alt_txt.read_text(encoding="utf-8", errors="ignore").strip()
                alt_txt.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[STT][Whisper] Error processing chunk {chunk_id}: {exc}")
        finally:
            wav_path.unlink(missing_ok=True)

        return {"chunk_id": chunk_id, "text": text, "is_final": False}

    # -------------------------- Public API -----------------------

    def submit_chunk(self, audio_chunk: np.ndarray, chunk_id: int) -> Future:
        if self.engine == "vosk":
            return self.executor.submit(self._process_chunk_vosk, audio_chunk, chunk_id, False)
        return self.executor.submit(self._process_chunk_whisper, audio_chunk, chunk_id)

    def finalize(self, chunk_id: int) -> Optional[Future]:
        if self.engine != "vosk":
            return None
        return self.executor.submit(self._finalize_vosk, chunk_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)


# ================================================================
# Streaming LLM via llama110
# ================================================================


class StreamingLLM:
    """Accumulate recognized text and asynchronously invoke llama110 when ready."""

    def __init__(self, llama_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.context_buffer: List[str] = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.llama_kwargs = llama_kwargs or {}

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
            "n_predict": self.llama_kwargs.get("n_predict", 96),
            "threads": self.llama_kwargs.get("threads", os.cpu_count() or 4),
            "temperature": self.llama_kwargs.get("temperature", 0.3),
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
        return response.strip()

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


@dataclass
class PiperVoiceInfo:
    sample_rate: int = 22050
    speaker_id: Optional[int] = None
    channels: int = 1
    metadata_path: Optional[Path] = None



class BufferedTTS:
    """Generate speech with Piper asynchronously and stream playback via paplay."""

    def __init__(
        self,
        model_path: Path = PIPER_MODEL_PATH,
        playback_cmd: Optional[Iterable[str]] = None,

        timeout: int = 30,

        output_device: Optional[Any] = None,
        use_subprocess: bool = False,
        on_playback_start: Optional[Callable[[str, float], None]] = None,
        on_playback_error: Optional[Callable[[], None]] = None,

    ) -> None:
        self.model_path = Path(model_path)
        self.timeout = int(timeout)
        self._voice_info = self._load_voice_info()
        self.speech_queue: "queue.Queue[SpeechSegment]" = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.playing = False
        self._playback_thread: Optional[threading.Thread] = None
        if playback_cmd:
            self.playback_cmd = list(playback_cmd)
        else:
            self.playback_cmd = [
                "aplay",
                "-r",
                str(self._voice_info.sample_rate),
                "-f",
                "S16_LE",
                "-t",
                "raw",
                "-",
            ]

        self.use_subprocess = bool(use_subprocess)
        if output_device is None:
            self.output_device = None
        elif isinstance(output_device, int):
            self.output_device = output_device
        elif isinstance(output_device, str):
            try:
                self.output_device = int(output_device)
            except ValueError:
                self.output_device = output_device
        else:
            self.output_device = output_device
        self.on_playback_start = on_playback_start
        self.on_playback_error = on_playback_error


        self._playback_env = os.environ.copy()
        if isinstance(self.output_device, str):
            # Hint to PulseAudio-based players which sink to target.
            self._playback_env.setdefault("PULSE_SINK", self.output_device)


    def _load_voice_info(self) -> PiperVoiceInfo:
        candidates = [
            self.model_path.with_suffix(self.model_path.suffix + ".json"),
            self.model_path.with_suffix(".json"),
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                metadata = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue

            audio = metadata.get("audio", {}) if isinstance(metadata, dict) else {}
            sample_rate = int(audio.get("sample_rate") or metadata.get("sample_rate", 22050))
            channels = int(audio.get("channels", 1) or 1)

            speaker_id: Optional[int] = None
            if "speaker_id" in metadata:
                try:
                    speaker_id = int(metadata.get("speaker_id"))
                except Exception:
                    speaker_id = None
            elif isinstance(metadata.get("speakers"), dict):
                speakers_dict: Dict[str, Any] = metadata.get("speakers", {})
                if speakers_dict:
                    first_key = next(iter(speakers_dict))
                    first_val = speakers_dict[first_key]
                    if isinstance(first_val, dict) and "id" in first_val:
                        try:
                            speaker_id = int(first_val["id"])
                        except Exception:
                            speaker_id = None
                    else:
                        try:
                            speaker_id = int(first_key)
                        except Exception:
                            speaker_id = None

            return PiperVoiceInfo(
                sample_rate=sample_rate or 22050,
                speaker_id=speaker_id,
                channels=channels or 1,
                metadata_path=candidate,
            )

        return PiperVoiceInfo()


    def start_playback(self) -> None:
        if self.playing:
            return
        self.playing = True
        self._playback_thread = threading.Thread(target=self._playback_loop, name="BufferedTTS", daemon=True)
        self._playback_thread.start()

    def _playback_loop(self) -> None:
        while self.playing:
            try:

                segment = self.speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not segment:
                continue

            played = False
            if not self.use_subprocess:
                played = self._play_via_sounddevice(segment)

            if not played and self.playback_cmd:
                played = self._play_via_subprocess(segment)

            if not played:
                print(f"[TTS] Playback failed for {segment.path}")
                self._notify_playback_error()

            try:
                if segment.path:
                    Path(segment.path).unlink(missing_ok=True)

            except OSError:
                pass

    def _notify_playback_error(self) -> None:
        if self.on_playback_error is None:
            return
        try:
            self.on_playback_error()
        except Exception as exc:
            print(f"[TTS] Playback error callback failed: {exc}")


    def _play_via_sounddevice(self, segment: SpeechSegment) -> bool:
        try:
            if segment.raw:
                audio = np.frombuffer(segment.raw, dtype=np.int16)
                max_val = float(np.iinfo(np.int16).max)
                audio = audio.astype(np.float32) / max_val
                if segment.channels > 1:
                    audio = audio.reshape(-1, segment.channels)
                sample_rate = segment.sample_rate
            else:
                with wave.open(segment.path, "rb") as wf:
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    frames = wf.getnframes()
                    audio_bytes = wf.readframes(frames)

                dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
                dtype = dtype_map.get(sampwidth)
                if dtype is None:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")

                audio = np.frombuffer(audio_bytes, dtype=dtype)
                if dtype == np.uint8:
                    audio = audio.astype(np.float32)
                    audio = (audio - 128.0) / 128.0
                else:
                    max_val = float(np.iinfo(dtype).max)
                    if not max_val:
                        raise ValueError("Invalid max value for dtype")
                    audio = audio.astype(np.float32) / max_val

                if channels > 1:
                    audio = audio.reshape(-1, channels)

            sd.stop()
            sd.play(audio, sample_rate, device=self.output_device, blocking=False)
            if self.on_playback_start:
                self.on_playback_start(segment.path, time.time())
            sd.wait()
            return True
        except Exception as exc:
            print(f"[TTS] Direct playback failed for {segment.path}: {exc}")
            return False

    def _play_via_subprocess(self, segment: SpeechSegment) -> bool:
        try:
            if self.on_playback_start:
                self.on_playback_start(segment.path, time.time())
            cmd = list(self.playback_cmd)
            if isinstance(self.output_device, str) and cmd:
                if (
                    cmd[0] == "paplay"
                    and not any(str(arg).startswith("--device=") for arg in cmd[1:])
                ):
                    cmd = cmd + [f"--device={self.output_device}"]
                elif cmd[0] == "aplay" and "-D" not in cmd:
                    cmd = cmd[:1] + ["-D", self.output_device] + cmd[1:]
            if any("{file}" in str(part) for part in cmd):
                resolved = [str(part).replace("{file}", segment.path) for part in cmd]
                subprocess.run(resolved, check=True, env=self._playback_env)
            elif cmd and cmd[-1] == "-" and segment.raw is not None:
                subprocess.run(cmd, input=segment.raw, check=True, env=self._playback_env)
            else:
                subprocess.run(cmd + [segment.path], check=True, env=self._playback_env)
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[TTS] Subprocess playback failed (exit {exc.returncode}) for {segment.path}: {exc}")
            return False
        except Exception as exc:
            print(f"[TTS] Subprocess playback failed for {segment.path}: {exc}")
            return False


    def generate_and_queue(self, text: str, segment_id: int) -> Optional[Future]:
        clean_text = (text or "").strip()
        if not clean_text:
            return None
        return self.executor.submit(self._generate_speech, clean_text, segment_id)


    def _generate_speech(self, text: str, segment_id: int) -> Optional[SpeechSegment]:

        if not self.model_path.exists():
            print(f"[TTS] Piper model not found: {self.model_path}")
            return None


        info = self._voice_info
        cmd = ["piper", "-m", str(self.model_path), "--output-raw"]
        if info.sample_rate:
            cmd += ["--sample-rate", str(info.sample_rate)]
        if info.speaker_id is not None:
            cmd += ["--speaker", str(info.speaker_id)]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        keep_file = False
        try:
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                check=True,
                timeout=self.timeout,
            )
            audio_bytes = proc.stdout
            if not audio_bytes:
                print("[TTS] Piper returned no audio data")
                return None

            with wave.open(str(tmp_path), "wb") as wf:
                wf.setnchannels(info.channels or 1)
                wf.setsampwidth(2)
                wf.setframerate(info.sample_rate or 22050)
                wf.writeframes(audio_bytes)
            keep_file = True

            segment = SpeechSegment(
                path=str(tmp_path),
                raw=audio_bytes,
                sample_rate=info.sample_rate or 22050,
                channels=info.channels or 1,
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



class ParallelVoiceAssistant:
    """Coordinate the streaming recorder, STT workers, llama110, and Piper TTS."""

    def __init__(
        self,
        chunk_duration: float = CHUNK_DURATION,
        sample_rate: int = SAMPLE_RATE,
        stt_workers: int = 2,
        vosk_model_dir: Path = VOSK_MODEL_DIR,
        piper_model_path: Path = PIPER_MODEL_PATH,
        llama_kwargs: Optional[Dict[str, Any]] = None,


        output_device: Optional[Any] = None,
        playback_cmd: Optional[Iterable[str]] = None,
        force_subprocess_playback: bool = False,
        silence_timeout: float = DEFAULT_SILENCE_TIMEOUT,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,


    ) -> None:
        self.recorder = StreamingRecorder(chunk_duration=chunk_duration, sample_rate=sample_rate)
        self.stt = ParallelSTT(num_workers=stt_workers, engine="vosk", sample_rate=sample_rate, vosk_model_dir=vosk_model_dir)
        self.llm = StreamingLLM(llama_kwargs=llama_kwargs)


        self.tts = BufferedTTS(
            model_path=piper_model_path,
            playback_cmd=playback_cmd,
            output_device=output_device,
            use_subprocess=force_subprocess_playback,
            on_playback_start=self._on_tts_playback_start,
            on_playback_error=self._on_tts_playback_error,
        )

        self.stt_futures: "queue.Queue[Tuple[int, Future, float]]" = queue.Queue()
        self.llm_futures: "queue.Queue[Tuple[Future, float, float]]" = queue.Queue()
        self.stats = PipelineStats()
        self._stt_done = threading.Event()
        self._pending_lock = threading.Lock()


        self._activity_lock = threading.Lock()
        self._activity_event = threading.Event()
        self._last_voice_time = time.time()
        self._has_detected_speech = False
        self._recording_stop_time: Optional[float] = None
        self._silence_timeout = float(silence_timeout)
        self._silence_threshold = float(silence_threshold)

    def _register_activity(self) -> None:
        with self._activity_lock:
            self._has_detected_speech = True
            self._last_voice_time = time.time()
        self._activity_event.set()

    def _is_silent_chunk(self, audio_chunk: np.ndarray) -> bool:
        if audio_chunk.size == 0:
            return True
        audio_view = np.asarray(audio_chunk, dtype=np.int16)
        if audio_view.ndim > 1:
            audio_view = audio_view.reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(audio_view.astype(np.float32)))))
        return rms < self._silence_threshold

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
        self._stt_done.clear()
        self._activity_event.clear()
        with self._activity_lock:
            self._has_detected_speech = False
            self._last_voice_time = start_time
        self._recording_stop_time = None



        self.recorder.start()
        self.tts.start_playback()

        stt_thread = threading.Thread(target=self._stt_pipeline, name="STTPipeline", daemon=True)
        llm_thread = threading.Thread(target=self._llm_pipeline, name="LLMPipeline", daemon=True)

        stt_thread.start()
        llm_thread.start()

        try:


            while True:
                now = time.time()
                if max_duration is not None and now - start_time >= max_duration:
                    print(f"[MAIN] Max duration {max_duration:.1f}s reached; wrapping up.")

                    self._recording_stop_time = now
                    self.stats.recording_stop_time = now

                    break

                with self._activity_lock:
                    has_voice = self._has_detected_speech
                    last_voice = self._last_voice_time

                if has_voice and (now - last_voice) >= self._silence_timeout:
                    print(f"[MAIN] Detected {self._silence_timeout:.1f}s of silence; stopping recorder.")

                    self._recording_stop_time = now
                    self.stats.recording_stop_time = now

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
            interrupt_time = time.time()
            self._recording_stop_time = interrupt_time
            self.stats.recording_stop_time = interrupt_time
        finally:
            self.recorder.stop()
            stop_time = time.time()
            if self._recording_stop_time is None:
                self._recording_stop_time = stop_time
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



                if not self._is_silent_chunk(audio_chunk):
                    self._register_activity()

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


            latency = max(0.0, time.time() - start_time)
            self.stats.stt_latencies.append(latency)


            text = (result.get("text") or "").strip()
            is_final = bool(result.get("is_final"))
            res_chunk_id = result.get("chunk_id", chunk_id)

            if text:

                self._register_activity()
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

            if (
                self.stats.recording_to_first_llm_latency is None
                and self._recording_stop_time is not None
            ):
                self.stats.recording_to_first_llm_latency = max(
                    0.0, response_ready_time - self._recording_stop_time
                )

            elif (
                self.stats.recording_to_first_llm_latency is None
                and self._recording_stop_time is None
            ):
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
                    tts_jobs.append((future, submit_time))
                segment_id += 1

            if not tts_jobs:
                continue


            pending_timestamp = self._reference_timestamp_for_output(input_timestamp)
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
            return

        if result:
            latency = max(0.0, time.time() - start_time)
            self.stats.tts_generation_latencies.append(latency)
        else:
            self._handle_failed_tts_generation(pending)

    def _handle_failed_tts_generation(self, pending: PendingOutput) -> None:
        with self._pending_lock:
            pending.segments_expected = max(0, pending.segments_expected - 1)
            if pending.segments_expected == 0:
                try:
                    self.stats.pending_outputs.remove(pending)
                except ValueError:
                    pass

    def _on_tts_playback_start(self, file_path: str, started_at: float) -> None:
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
        self._print_latency_summary("TTS generation latency", list(self.stats.tts_generation_latencies))
        self._print_latency_summary("Input -> first audio gap", list(self.stats.input_to_output_latencies))


        print("----------------------\n")


# ================================================================
# Model Warm-up Helpers
# ================================================================


class ModelPreloader:
    """Utility helpers to warm up the local models so first inference is faster."""

    @staticmethod
    def warmup_vosk(sample_rate: int = SAMPLE_RATE) -> None:
        print("[WARMUP] Loading Vosk model...")
        model = VoskModel(str(VOSK_MODEL_DIR))
        recognizer = KaldiRecognizer(model, sample_rate)
        recognizer.AcceptWaveform(b"\x00" * sample_rate)  # 1 second of silence
        recognizer.FinalResult()
        print("[WARMUP] Vosk ready")

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
        "--vosk-model",
        type=Path,
        default=VOSK_MODEL_DIR,
        help="Path to Vosk model directory",
    )
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="Threads to pass to llama110")

    parser.add_argument("--n-predict", type=int, default=16, help="Tokens to generate with llama110")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature for llama110")
    parser.add_argument("--llama-cli", type=Path, default=None, help="Optional override for llama-cli path")
    parser.add_argument("--llama-model", type=Path, default=None, help="Optional override for llama model path")

    parser.add_argument("--output-device", type=str, default=None, help="sounddevice output (index or name) for Piper playback")
    parser.add_argument("--playback-cmd", nargs="+", default=None, help="Fallback playback command for Piper audio")
    parser.add_argument(
        "--force-subprocess-playback",
        action="store_true",
        help="Skip direct sounddevice playback and always use the playback command",
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

        ModelPreloader.warmup_vosk()
        ModelPreloader.warmup_llama(llama_kwargs)
        ModelPreloader.warmup_piper(args.piper_model)

    assistant = ParallelVoiceAssistant(
        chunk_duration=CHUNK_DURATION,
        sample_rate=SAMPLE_RATE,
        stt_workers=2,
        vosk_model_dir=args.vosk_model,
        piper_model_path=args.piper_model,
        llama_kwargs=llama_kwargs,


        output_device=args.output_device,
        playback_cmd=args.playback_cmd,
        force_subprocess_playback=args.force_subprocess_playback,
        silence_timeout=args.silence_timeout,
        silence_threshold=args.silence_threshold,
    )
    max_duration = args.duration if args.duration and args.duration > 0 else None
    assistant.run(duration=max_duration)




if __name__ == "__main__":
    main()
