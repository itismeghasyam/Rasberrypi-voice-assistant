# pipeline/recorder.py

## Purpose
Continuously capture microphone audio on a background thread and hand off fixed-size PCM chunks to the STT pipeline.

## What it provides
- A `StreamingRecorder` class that:
  - starts and stops a background capture loop
  - exposes `get_chunk(timeout)` to fetch the next audio slice
  - yields mono 16-bit PCM at the configured sample rate

---

## Imports explained
- `threading`, `queue`  
  Worker thread and a thread-safe queue for producer to consumer handoff.
- `typing import Optional`  
  Return type for `get_chunk`.
- `numpy as np`  
  Buffer handling and type conversion.
- `sounddevice as sd`  
  PortAudio bindings used for microphone input.
- `from config import CHUNK_DURATION, SAMPLE_RATE`  
  Default chunk size in seconds and the audio sample rate.

---

## Public API

### Class
- `StreamingRecorder(chunk_duration: float = CHUNK_DURATION, sample_rate: int = SAMPLE_RATE)`

Attributes set on init
- `chunk_duration` (seconds)  
- `sample_rate` (Hz)  
- `chunk_queue: queue.Queue[np.ndarray]`  
- `recording: bool`  
- `_thread: Optional[threading.Thread]`

### Methods
- `start() -> None`  
  If not already running, flips `recording = True`, clears any stale items, and spawns a daemon thread that executes `_record_loop`.
- `get_chunk(timeout: float = 0.5) -> Optional[np.ndarray]`  
  Non-blocking style fetch. Returns an `np.ndarray` of `int16` samples when available, otherwise `None` after `timeout` seconds.
- `clear_queue() -> None`  
  Drains the queue without blocking so downstream can start fresh.
- `stop() -> None`  
  Signals the loop to stop, joins the worker thread briefly, and resets internal state.

---

## How the capture loop works

- `_record_loop()` runs on a daemon thread created by `start()`.
- It computes `chunk_samples = int(chunk_duration * sample_rate)`.
- It opens `sd.InputStream` with:
  - `samplerate = sample_rate`
  - `channels = 1` (mono)
  - `dtype = "int16"`
- While `recording` is true:
  - read `chunk_samples` frames from the stream
  - copy into a standalone `np.ndarray(dtype=np.int16)` so PortAudio buffers are not referenced
  - `put` the array into `chunk_queue`
- On exit it closes the stream and lets `stop()` finish cleanup.

Audio format of each chunk
- mono, 16-bit PCM, shape `(chunk_samples, )` or `(chunk_samples, 1)` depending on how the stream returns frames
- sample rate equals `sample_rate` from init

---

## Example usage

    from recorder import StreamingRecorder

    rec = StreamingRecorder()  # uses CHUNK_DURATION and SAMPLE_RATE defaults
    rec.start()
    try:
        while True:
            chunk = rec.get_chunk(timeout=0.5)
            if chunk is None:
                continue
            # hand 'chunk' to your STT engine here
    finally:
        rec.stop()

---

## Operational notes

- Threading  
  The recorder is a producer. STT consumes via `get_chunk`. Keep the timeout short to stay responsive to stop signals.

- Backpressure  
  If the consumer is slower than real time, the queue can grow. Consider dropping oldest items or increasing chunk size if that happens.

- Device selection  
  If you need a specific input device, configure the default input device in ALSA or pass `device` to `sd.InputStream` in your local fork.

- Levels and clipping  
  Make sure the mic gain is reasonable. Clipping will hurt recognition.

- Shutdown behavior  
  Always call `stop()` in a `finally` block to avoid dangling streams.

---

## Where the code goes next
`ParallelVoiceAssistant` starts the recorder, then the STT pipeline repeatedly calls `get_chunk(...)` and sends voiced chunks to Faster-Whisper.

