# pipeline/config.py

## Purpose
Central place for runtime defaults the rest of the pipeline imports. It keeps audio settings consistent and gives sensible thresholds for silence based auto-stop.

---

## Imports
- Standard library only (e.g., `os` or `pathlib` if paths need expanding). Nothing heavy gets imported here.

---

## Constants and what uses them

| Name | Type | Default | Used by | What it does |
|---|---:|---:|---|---|
| `SAMPLE_RATE` | int | `16000` | `recorder.py`, `stt_faster.py`, `orchestrator.py` | Global audio sample rate in Hz. Recorder opens the mic stream with this value. |
| `CHUNK_DURATION` | float | `0.5` | `recorder.py`, `orchestrator.py` | Size of each audio slice in seconds. Recorder computes `chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)`. |
| `DEFAULT_SILENCE_THRESHOLD` | int | `1300` | `orchestrator.py` | RMS threshold on `int16` audio. Chunks under this are considered silence. Tune per mic/noise. |
| `DEFAULT_SILENCE_TIMEOUT` | float | `1.5` | `orchestrator.py` | Seconds of continuous silence before auto-stop. |
| `WHISPER_EXE` | str | path to `whisper-cli` | `main.py` (legacy) | Default binary path for the older whisper.cpp CLI path. Safe to leave as is if you are using Faster-Whisper. |
| `WHISPER_MODEL` | str | path to tiny model | `main.py` (legacy) | Default model path for the whisper.cpp CLI path. Not used when Faster-Whisper is active. |

> Notes  
> - We are using Faster-Whisper and eSpeak right now, so the `WHISPER_*` values are mostly there for backward compatibility.  
> - No server URL is defined or used here.

---

## Tuning guide

- **Lower latency**  
  - Decrease `CHUNK_DURATION` to 0.25. You’ll get more frequent STT updates at the cost of slightly higher CPU overhead.  
  - Keep `SAMPLE_RATE` at 16 kHz for CPU friendliness unless you have a good reason to change it.

- **Robustness in noisy rooms**  
  - Raise `DEFAULT_SILENCE_THRESHOLD` a bit (e.g., 1600–2000) so background hum isn’t treated as speech.  
  - Increase `DEFAULT_SILENCE_TIMEOUT` to 2.0 or 2.5 so the session doesn’t stop between short pauses.

- **Sensitive mic / quiet room**  
  - Lower `DEFAULT_SILENCE_THRESHOLD` toward 900–1100.  
  - Keep `DEFAULT_SILENCE_TIMEOUT` around 1.5 for snappy endings.

---

## Example: who imports what

- `recorder.py`  
  - reads `SAMPLE_RATE` and `CHUNK_DURATION` to size the input stream and the queue chunks.

- `stt_faster.py`  
  - uses `SAMPLE_RATE` to interpret PCM chunks and to size its rolling window for partials.

- `orchestrator.py`  
  - reads `CHUNK_DURATION`, `SAMPLE_RATE`, `DEFAULT_SILENCE_THRESHOLD`, `DEFAULT_SILENCE_TIMEOUT` to decide chunking, silence detection, and when to auto-stop.

- `main.py`  
  - exposes CLI flags that can override some of these defaults at runtime and passes them through to the orchestrator.

---

## Minimal code examples

Importing defaults in another module:

    from config import SAMPLE_RATE, CHUNK_DURATION
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

Using silence controls:

    from config import DEFAULT_SILENCE_THRESHOLD, DEFAULT_SILENCE_TIMEOUT
    if rms_value < DEFAULT_SILENCE_THRESHOLD:
        maybe_silent_for += chunk_seconds
        if maybe_silent_for >= DEFAULT_SILENCE_TIMEOUT:
            request_stop()

---

## Where this file sits in the flow
`config.py` is read very early. After `main.py` parses flags, the orchestrator builds the pipeline using these values. From there, `recorder.py` and `stt_faster.py` keep using `SAMPLE_RATE` and `CHUNK_DURATION` to stay in sync.

