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
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
                else:
                    partial = json.loads(recognizer.PartialResult())
                    text = partial.get("partial", "")

                if is_final:
                    final = json.loads(recognizer.FinalResult())
                    final_text = final.get("text", "")
                    if final_text:
                        # Append any remaining text that FinalResult surfaced
                        text = (f"{text} {final_text}" if text else final_text).strip()
                    # Reset recognizer for the next session
                    self._recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
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


class BufferedTTS:
    """Generate speech with Piper asynchronously and stream playback via paplay."""

    def __init__(
        self,
        model_path: Path = PIPER_MODEL_PATH,
        playback_cmd: Optional[Iterable[str]] = None,
        timeout: int = 30,
    ) -> None:
        self.model_path = Path(model_path)
        self.timeout = int(timeout)
        self.speech_queue: "queue.Queue[str]" = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.playing = False
        self._playback_thread: Optional[threading.Thread] = None
        self.playback_cmd = list(playback_cmd) if playback_cmd else ["paplay"]

    def start_playback(self) -> None:
        if self.playing:
            return
        self.playing = True
        self._playback_thread = threading.Thread(target=self._playback_loop, name="BufferedTTS", daemon=True)
        self._playback_thread.start()

    def _playback_loop(self) -> None:
        while self.playing:
            try:
                audio_file = self.speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not audio_file:
                continue

            try:
                subprocess.run(self.playback_cmd + [audio_file], check=False)
            except Exception as exc:
                print(f"[TTS] Playback failed for {audio_file}: {exc}")
            finally:
                try:
                    Path(audio_file).unlink(missing_ok=True)
                except OSError:
                    pass

    def generate_and_queue(self, text: str, segment_id: int) -> Optional[Future]:
        clean_text = (text or "").strip()
        if not clean_text:
            return None
        return self.executor.submit(self._generate_speech, clean_text, segment_id)

    def _generate_speech(self, text: str, segment_id: int) -> Optional[str]:
        if not self.model_path.exists():
            print(f"[TTS] Piper model not found: {self.model_path}")
            return None

        output_file = Path(f"/tmp/tts_segment_{segment_id}.wav")
        cmd = [
            "piper",
            "-m",
            str(self.model_path),
            "--output_file",
            str(output_file),
        ]
        try:
            subprocess.run(cmd, input=text.encode("utf-8"), check=True, timeout=self.timeout)
            self.speech_queue.put(str(output_file))
            return str(output_file)
        except subprocess.CalledProcessError as exc:
            print(f"[TTS] Piper returned error: {exc}")
        except Exception as exc:
            print(f"[TTS] Piper failed: {exc}")
        finally:
            if not output_file.exists():
                try:
                    output_file.unlink(missing_ok=True)
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
class PipelineStats:
    stt_chunks: int = 0
    llm_responses: int = 0
    tts_segments: int = 0
    total_latency: float = 0.0
    start_time: float = field(default_factory=time.time)


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
    ) -> None:
        self.recorder = StreamingRecorder(chunk_duration=chunk_duration, sample_rate=sample_rate)
        self.stt = ParallelSTT(num_workers=stt_workers, engine="vosk", sample_rate=sample_rate, vosk_model_dir=vosk_model_dir)
        self.llm = StreamingLLM(llama_kwargs=llama_kwargs)
        self.tts = BufferedTTS(model_path=piper_model_path)

        self.stt_futures: "queue.Queue[tuple[int, Future]]" = queue.Queue()
        self.llm_futures: "queue.Queue[Future]" = queue.Queue()
        self.stats = PipelineStats()
        self._stt_done = threading.Event()

    def run(self, duration: float = 15.0) -> None:
        print(f"[MAIN] Starting streaming assistant for ~{duration:.1f}s")
        start_time = time.time()
        self.stats.start_time = start_time

        self.recorder.start()
        self.tts.start_playback()

        stt_thread = threading.Thread(target=self._stt_pipeline, name="STTPipeline", daemon=True)
        llm_thread = threading.Thread(target=self._llm_pipeline, name="LLMPipeline", daemon=True)

        stt_thread.start()
        llm_thread.start()

        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            print("\n[MAIN] Interrupted by user")
        finally:
            self.recorder.stop()

        stt_thread.join(timeout=5.0)

        finalize_future = self.stt.finalize(self.stats.stt_chunks + 1)
        if finalize_future is not None:
            self.stt_futures.put((self.stats.stt_chunks + 1, finalize_future))
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

                future = self.stt.submit_chunk(audio_chunk, chunk_id)
                self.stt_futures.put((chunk_id, future))
                self.stats.stt_chunks += 1
                chunk_id += 1

                self._process_stt_results(wait=False)

            self._process_stt_results(wait=True)
        finally:
            # Ensure any exceptions don't leave futures undispatched.
            pass

    def _process_stt_results(self, wait: bool) -> None:
        pending: List[tuple[int, Future]] = []
        while not self.stt_futures.empty():
            chunk_id, future = self.stt_futures.get()
            if wait or future.done():
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"[STT Pipeline] Future for chunk {chunk_id} failed: {exc}")
                    continue
            else:
                pending.append((chunk_id, future))
                continue

            if not result:
                continue

            text = (result.get("text") or "").strip()
            is_final = bool(result.get("is_final"))
            res_chunk_id = result.get("chunk_id", chunk_id)

            if text:
                print(f"[STT] Chunk {res_chunk_id}: {text}")

            llm_future = self.llm.process_incremental(text, is_final=is_final)
            if llm_future is not None:
                self.llm_futures.put(llm_future)

        for item in pending:
            self.stt_futures.put(item)

    def _llm_pipeline(self) -> None:
        segment_id = 0
        while not self._stt_done.is_set() or not self.llm_futures.empty():
            try:
                llm_future = self.llm_futures.get(timeout=0.5)
            except queue.Empty:
                continue

            response = ""
            try:
                response = llm_future.result(timeout=300)
            except Exception as exc:
                print(f"[LLM Pipeline] Error: {exc}")
                continue

            response = (response or "").strip()
            if not response:
                continue

            print(f"[LLM] Response: {response[:120]}{'...' if len(response) > 120 else ''}")
            self.stats.llm_responses += 1

            for sentence in self._split_sentences(response):
                future = self.tts.generate_and_queue(sentence, segment_id)
                if future is not None:
                    self.stats.tts_segments += 1
                segment_id += 1

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

    def _print_stats(self, elapsed: float) -> None:
        print("\n--- PIPELINE STATS ---")
        print(f"Runtime: {elapsed:.2f}s")
        print(f"STT chunks processed: {self.stats.stt_chunks}")
        print(f"LLM responses generated: {self.stats.llm_responses}")
        print(f"TTS segments queued: {self.stats.tts_segments}")

        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Memory usage: {mem_mb:.1f} MB")
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
    parser.add_argument("--no-warmup", action="store_true", help="Skip model warm-up steps")
    parser.add_argument("--piper-model", type=Path, default=PIPER_MODEL_PATH, help="Path to Piper .onnx model")
    parser.add_argument(
        "--vosk-model",
        type=Path,
        default=VOSK_MODEL_DIR,
        help="Path to Vosk model directory",
    )
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="Threads to pass to llama110")
    parser.add_argument("--n-predict", type=int, default=96, help="Tokens to generate with llama110")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature for llama110")
    parser.add_argument("--llama-cli", type=Path, default=None, help="Optional override for llama-cli path")
    parser.add_argument("--llama-model", type=Path, default=None, help="Optional override for llama model path")
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

    if not args.no_warmup:
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
    )
    assistant.run(duration=args.duration)


if __name__ == "__main__":
    main()
