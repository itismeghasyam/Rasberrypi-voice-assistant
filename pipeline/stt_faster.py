# stt_faster.py — persistent faster-whisper STT with throttle/debounce for partials
from typing import Optional, Any, Dict, List
from concurrent.futures import Future, ThreadPoolExecutor
import threading, time, os
import numpy as np

# Silence ORT warnings early (must be set before import)
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

try:
    from faster_whisper import WhisperModel
except Exception as e:
    WhisperModel = None
    _import_err = e

from config import SAMPLE_RATE

class PersistentWhisperSTT:
    """
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
        # keep signature compatible with old ParallelSTT ctor:
        whisper_exe: Optional[object] = None,
        whisper_model: Optional[object] = None,
        whisper_threads: Optional[int] = None,
        emit_partials: bool = True,
        # faster-whisper params:
        model_name: str = "tiny.en",    # English-only gives a slight speed win
        compute_type: str = "int8",     # good on Pi 4
        window_ms: int = 1200,          # rolling window size for partials
        min_partial_interval_ms: int = 350,  # throttle partial frequency
        language: Optional[str] = "en",
        use_vad: bool = False,          # VAD adds overhead on Pi; keep off for partials
    ) -> None:
        if WhisperModel is None:
            raise RuntimeError(f"faster-whisper is not installed: {_import_err}")

        self.executor = ThreadPoolExecutor(max_workers=max(1, num_workers))
        self.sample_rate = int(sample_rate)
        self.emit_partials = bool(emit_partials)
        self.language = language
        self.use_vad = use_vad

        # Load once, keep hot
        self.model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
        print(f"[STT][FW] Loaded {model_name} ({compute_type}) — hot")

        # Rolling buffers + state
        self._window_bytes = int(self.sample_rate * 2 * (window_ms / 1000.0))
        self._rolling = bytearray()
        self._chunks: List[bytes] = []
        self._lock = threading.Lock()

        # Throttle / single-flight
        self._inflight = False
        self._last_partial_time = 0.0
        self._min_partial_interval = min_partial_interval_ms / 1000.0
        self._last_emitted_text = ""

    # ---------- helpers ----------
    def empty_future(self, chunk_id: int) -> Future:
        return self.executor.submit(lambda cid=chunk_id: {"chunk_id": cid, "text": "", "is_final": False})

    @staticmethod
    def _pcm_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        return pcm16 / 32768.0

    def _transcribe(self, pcm: np.ndarray) -> str:
        segs, _info = self.model.transcribe(
            pcm,
            language=self.language,
            vad_filter=self.use_vad,
            beam_size=1,           # greedy is faster
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        return "".join(s.text for s in segs).strip()

    # ---------- core workers ----------
    def _partial_worker(self, chunk_id: int) -> Dict[str, Any]:
        try:
            with self._lock:
                # Use current rolling window
                window = bytes(self._rolling)

            # Ensure at least ~200 ms so early words aren't dropped
            min_bytes = int(self.sample_rate * 2 * 0.2)
            if len(window) < min_bytes:
                window = window + b"\x00" * (min_bytes - len(window))

            text = self._transcribe(self._pcm_bytes_to_float32(window))

            # Deduplicate partials
            emit = ""
            with self._lock:
                if text and text != self._last_emitted_text:
                    emit = text
                    self._last_emitted_text = text
                self._last_partial_time = time.time()
        finally:
            with self._lock:
                self._inflight = False  # allow next partial

        return {"chunk_id": chunk_id, "text": emit, "is_final": False}

    def _finalize_worker(self, chunk_id: int, mark_final: bool) -> Dict[str, Any]:
        with self._lock:
            full_pcm = b"".join(self._chunks)
            # reset rolling + chunks for next turn
            self._rolling.clear()
            self._chunks.clear()
            self._last_emitted_text = ""
            self._inflight = False

        if not full_pcm:
            return {"chunk_id": chunk_id, "text": "", "is_final": True}

        text = self._transcribe(self._pcm_bytes_to_float32(full_pcm))
        return {"chunk_id": chunk_id, "text": text, "is_final": mark_final}

    # ---------- public API ----------
    def submit_chunk(self, audio_chunk: np.ndarray, chunk_id: int) -> Future:
        # audio_chunk is np.int16
        audio_bytes = np.ascontiguousarray(audio_chunk, dtype=np.int16).tobytes()
        now = time.time()

        with self._lock:
            # accumulate full utterance
            self._chunks.append(audio_bytes)
            # update rolling window
            self._rolling.extend(audio_bytes)
            if len(self._rolling) > self._window_bytes:
                self._rolling = self._rolling[-self._window_bytes:]

            # If partials disabled, return empty placeholder (orchestrator will force a flush later)
            if not self.emit_partials:
                return self.empty_future(chunk_id)

            # Throttle: if a decode is running, or last partial was < N ms ago, skip this chunk
            if self._inflight or (now - self._last_partial_time) < self._min_partial_interval:
                return self.empty_future(chunk_id)

            # Single-flight: mark as busy and launch one worker
            self._inflight = True

        return self.executor.submit(self._partial_worker, chunk_id)

    def finalize(self, chunk_id: int, mark_final: bool = True) -> Optional[Future]:
        # Final pass on the full utterance
        return self.executor.submit(self._finalize_worker, chunk_id, mark_final)

    def reset(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._rolling.clear()
            self._last_emitted_text = ""
            self._inflight = False
            self._last_partial_time = 0.0

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
