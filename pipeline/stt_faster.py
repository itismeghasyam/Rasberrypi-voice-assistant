# stt_faster.py — drop-in persistent STT backend using faster-whisper
from typing import Optional, Any, Dict, List
import threading, time
from concurrent.futures import Future, ThreadPoolExecutor
import numpy as np

try:
    from faster_whisper import WhisperModel
except Exception as e:
    WhisperModel = None
    _import_err = e

from config import SAMPLE_RATE

class PersistentWhisperSTT:
    """
    In-process faster-whisper STT that keeps the model hot for the lifetime of the app.
    Public API matches your existing STT classes:
      - submit_chunk(audio_chunk: np.ndarray, chunk_id: int) -> Future[{"chunk_id","text","is_final"}]
      - finalize(chunk_id: int, mark_final: bool=True) -> Future[...]
      - empty_future(chunk_id) -> Future[...]
      - reset(), shutdown()
    """
    def __init__(
        self,
        num_workers: int = 2,
        sample_rate: int = SAMPLE_RATE,
        # Accept whisper.cpp-style kwargs so orchestrator can pass them without errors
        whisper_exe: Optional[object] = None,
        whisper_model: Optional[object] = None,
        whisper_threads: Optional[int] = None,
        emit_partials: bool = True,
        # faster-whisper specific
        model_name: str = "tiny",      # try "tiny" or "tiny.en"
        compute_type: str = "int8",    # good for Pi 4
        rolling_ms: int = 1200,        # short tail for quick partials
        language: Optional[str] = None # e.g., "en" to force English
    ) -> None:
        if WhisperModel is None:
            raise RuntimeError(f"faster-whisper is not installed: {_import_err}")

        self.executor = ThreadPoolExecutor(max_workers=max(1, num_workers))
        self.sample_rate = int(sample_rate)
        self.emit_partials = bool(emit_partials)
        self.language = language

        # Model stays hot in RAM
        self.model = WhisperModel(model_name, device="cpu", compute_type=compute_type)

        # Rolling and full buffers (PCM int16 bytes)
        self._rolling_ms = int(rolling_ms)
        self._rolling = bytearray()
        self._chunks: List[bytes] = []
        self._transcript_lock = threading.Lock()
        self._emitted_transcript = ""
        self._last_partial = ""

    # -------- helpers --------
    def empty_future(self, chunk_id: int) -> Future:
        return self.executor.submit(lambda cid=chunk_id: {"chunk_id": cid, "text": "", "is_final": False})

    def _pcm_bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        return pcm16 / 32768.0

    def _transcribe_array(self, audio_arr: np.ndarray, timeout_s: float = 30.0) -> str:
        segments, _info = self.model.transcribe(
            audio_arr,
            language=self.language,
            vad_filter=True,
            no_speech_threshold=0.35,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip()

    # -------- workers --------
    def _process_chunk_fw(self, audio_bytes: bytes, chunk_id: int) -> Dict[str, Any]:
        with self._transcript_lock:
            rolling = bytes(self._rolling)

        # Ensure a minimum tail so early words show up
        min_ms = 200.0
        min_bytes = int(self.sample_rate * 2 * (min_ms / 1000.0))
        if len(rolling) < min_bytes:
            rolling = rolling + b"\x00" * (min_bytes - len(rolling))

        try:
            text = self._transcribe_array(self._pcm_bytes_to_float32(rolling), timeout_s=20.0)
        except Exception as e:
            print(f"[STT][FW] transcribe error: {e}")
            text = ""

        return {"chunk_id": chunk_id, "text": text, "is_final": False}

    def _finalize_fw(self, chunk_id: int, mark_final: bool) -> Dict[str, Any]:
        with self._transcript_lock:
            chunks = list(self._chunks)
            emitted = self._emitted_transcript.strip()
            self._last_partial = ""
            self._chunks.clear()

        if not chunks:
            return {"chunk_id": chunk_id, "text": "", "is_final": True}

        full_pcm = b"".join(chunks)
        try:
            full_text = self._transcribe_array(self._pcm_bytes_to_float32(full_pcm), timeout_s=30.0)
        except Exception as e:
            print(f"[STT][FW] finalize error: {e}")
            full_text = ""

        new_text = full_text
        if emitted and full_text.lower().startswith(emitted.lower()):
            new_text = full_text[len(emitted):].strip()

        with self._transcript_lock:
            self._emitted_transcript = full_text

        return {"chunk_id": chunk_id, "text": new_text, "is_final": mark_final}

    # -------- public API --------
    def submit_chunk(self, audio_chunk: np.ndarray, chunk_id: int) -> Future:
        # `audio_chunk` is np.int16 from your recorder
        audio_bytes = np.ascontiguousarray(audio_chunk, dtype=np.int16).tobytes()
        with self._transcript_lock:
            self._chunks.append(audio_bytes)

            # Update rolling tail
            max_bytes = int(self.sample_rate * 2 * (self._rolling_ms / 1000.0))
            self._rolling.extend(audio_bytes)
            if len(self._rolling) > max_bytes:
                self._rolling = self._rolling[-max_bytes:]

            if not self.emit_partials:
                return self.executor.submit(lambda cid=chunk_id: {"chunk_id": cid, "text": "", "is_final": False})

        return self.executor.submit(self._process_chunk_fw, audio_bytes, chunk_id)

    def finalize(self, chunk_id: int, mark_final: bool = True) -> Optional[Future]:
        return self.executor.submit(self._finalize_fw, chunk_id, mark_final)

    def reset(self) -> None:
        with self._transcript_lock:
            self._chunks.clear()
            self._emitted_transcript = ""
            self._last_partial = ""
            self._rolling = bytearray()

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
