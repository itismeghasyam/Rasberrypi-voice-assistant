# pipeline/stt_faster.py

## Purpose
Persistent Faster-Whisper speech to text with a single shared CPU model and throttled partials. Keeps the model hot across requests to reduce latency.

## Overview
- Loads one shared `WhisperModel` on CPU with a lock.
- Accepts `np.int16` mono chunks from the recorder.
- Maintains a rolling byte window for partial recognition.
- Throttles interim decodes so partials are not spammy.
- Provides a simple Future based API compatible with the older STT class.

---

## Imports and globals
- `from concurrent.futures import ThreadPoolExecutor, Future` for async workers
- `import threading, time, os, numpy as np`
- `from faster_whisper import WhisperModel` with ORT logging level reduced before import
- `from config import SAMPLE_RATE` for the default sample rate
- Globals: `_SHARED_MODEL` and `_SHARED_LOCK` so all instances reuse the same Faster-Whisper model

---

## Public class: `PersistentWhisperSTT`

### Constructor signature
`PersistentWhisperSTT(num_workers=2, sample_rate=SAMPLE_RATE, whisper_exe=None, whisper_model=None, whisper_threads=None, emit_partials=True, model_name="tiny.en", compute_type="int8", window_ms=1200, min_partial_interval_ms=350, language="en", use_vad=False)`

Notes
- `whisper_*` args exist for compatibility with the older CLI path and are ignored here.
- Default model is `tiny.en` with `compute_type="int8"` which is a good fit for Raspberry Pi 4 CPU.
- `window_ms` defines how much recent audio is considered for partials.
- `min_partial_interval_ms` throttles interim updates.
- `use_vad=False` is kept off by default to avoid extra overhead on the Pi.

On init
- Validates that `faster_whisper` is importable or raises a `RuntimeError`.
- Creates a `ThreadPoolExecutor(max_workers=num_workers)`.
- Stores `sample_rate`, `emit_partials`, `language`, `use_vad`.
- Loads the shared model once via a helper that honors `compute_type` and `model_name`.
- Sets up rolling state:
  - `_window_bytes = int(sample_rate * 2 * (window_ms / 1000.0))`
  - `_rolling: bytearray`, `_chunks: List[bytes]`
  - `_lock = threading.Lock()`
  - `_inflight = False`, `_last_partial_time = 0.0`, `_min_partial_interval = min_partial_interval_ms / 1000.0`
  - `_last_emitted_text = ""`

### Helper methods
- `empty_future(chunk_id) -> Future`  
  Returns a Future that resolves to an empty non final result dict.
- `_pcm_bytes_to_float32(bytes) -> np.ndarray`  
  Converts PCM16 to float32 in range [-1, 1].
- `_transcribe(pcm: np.ndarray) -> str`  
  Calls `self.model.transcribe(...)` with `language`, optional VAD, greedy beam, and no conditioning, then joins segment text.

### Workers
- `_partial_worker(chunk_id) -> Dict`  
  Copies the current rolling window, pads to at least ~200 ms, runs `_transcribe`, deduplicates against the last emitted text, updates `last_partial_time`, releases `_inflight`, returns `{"chunk_id", "text", "is_final": False}`.
- `_finalize_worker(chunk_id, mark_final: bool) -> Dict`  
  Joins all buffered `_chunks`, clears state for the next utterance, runs `_transcribe` on the full PCM, and returns `{"chunk_id", "text", "is_final": mark_final}`.

### Public API
- `submit_chunk(audio_chunk: np.ndarray, chunk_id: int) -> Future`  
  - Accepts `np.int16` mono. Converts to bytes, appends to `_chunks`, updates `_rolling` capped at `_window_bytes`.  
  - If `emit_partials` is `False` returns `empty_future`.  
  - Applies throttle: if a worker is already running or the last partial was too recent, returns `empty_future`.  
  - Otherwise marks `_inflight = True` and submits `_partial_worker` to the executor.
- `finalize(chunk_id: int, mark_final: bool = True) -> Future`  
  Runs `_finalize_worker` on the accumulated audio for a full pass.
- `reset() -> None`  
  Clears buffers and throttling counters.
- `shutdown() -> None`  
  Shuts down the executor.

---

## Data flow summary
1. Recorder produces `np.int16` chunks.  
2. Orchestrator calls `submit_chunk` for voiced chunks, `empty_future` for silence.  
3. Periodic partials come from the rolling window when throttle allows.  
4. At boundaries or stop, orchestrator calls `finalize` to get the full utterance text.

---

## Why this is fast on Pi 4
- Single shared int8 model on CPU avoids repeated loads.
- Greedy decode with small window reduces compute per partial.
- Throttling keeps only one decode in flight and limits update rate.
- English only model `tiny.en` cuts params and memory.

---

## Handoff to the next stage
The orchestrator forwards emitted text to the local LLM. The `submit_chunk` and `finalize` return values are queued into the LLM pipeline which streams a short reply and hands sentences to eSpeak for playback.
