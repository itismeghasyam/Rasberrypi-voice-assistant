# pipeline/orchestrator.py

## Purpose
Coordinates the streaming assistant. It ties together the recorder, STT engine, LLM, and TTS, manages silence‑based stopping, and records latency metrics.

## Key imports
- Dataclasses and typing utilities for stats and queues
- `config` defaults for audio, silence, and paths
- `StreamingRecorder` for mic capture
- STT selection: try to import `PersistentWhisperSTT` from `stt_faster`, otherwise fall back to `ParallelSTT` from `stt`
- `StreamingLLM` and `speak_text_timed` for llama.cpp and eSpeak
- `BufferedTTS` is present in code but disabled in favor of eSpeak

## Data classes
- `PendingOutput` — tracks expected TTS segments and reference timestamps
- `PipelineStats` — counts chunks, responses, segments, and stores latency arrays:
  - `stt_latencies`, `llm_latencies`, `tts_generation_latencies`, `input_to_output_latencies`
  - First‑event timings like `recording_start_to_first_tts_latency`

## Class `ParallelVoiceAssistant(...)`

### Constructor parameters
- `chunk_duration`, `sample_rate`, `stt_workers`
- Whisper CLI parameters and `emit_stt_partials`
- `piper_model_path`, playback controls, and output device (unused with eSpeak)
- `llama_kwargs` for the LLM
- `silence_timeout`, `silence_threshold`
- `whisper_server` optional URL to use the HTTP STT path

### Initialization behavior
- Start a `StreamingRecorder`
- Choose STT:
  - If `whisper_server` is set, build `ParallelSTTHTTP` (not used in this project)
  - Otherwise build `PersistentWhisperSTT` if available, or fall back to `ParallelSTT`
- Initialize `StreamingLLM` with `llama_kwargs`
- Disable Piper and set `self.tts = None`
- Provide an eSpeak adapter with `self.espeak_tts = lambda text, segment_id: speak_text_timed(text)`
- Initialize queues, events, locks, and metrics state
- Set silence control thresholds and compute chunk budget for auto stop

## `run(duration=None)`

1. Reset state, stats, and events; record start time
2. Start the recorder
3. Launch two daemon threads:
   - `_stt_pipeline` for chunk processing and transcripts
   - `_llm_pipeline` for incremental LLM responses and TTS
4. Main loop:
   - Exit on explicit stop request, max duration, or silence timeout
   - Silence timeout triggers when no speech activity is seen for `silence_timeout` seconds
   - The loop waits on an activity event with a short timeout to stay responsive
5. On exit:
   - Stop the recorder and drain STT futures
   - Optionally run a final STT pass if the last voice was not recent
   - Signal LLM done, join threads, then shut down STT and LLM
   - Wait for any pending TTS completion if Piper were enabled
   - Print a detailed latency summary

## `_stt_pipeline()`

- Pull audio chunks from the recorder queue
- Compute RMS and decide if a chunk is silent using `_is_silent_chunk(...)`
- For voice chunks submit to `self.stt.submit_chunk(...)`, otherwise queue an empty future
- Track activity timestamps and chunk counters
- Periodically call `_process_stt_results(wait=False)`
- After recording stops, flush remaining futures with `_process_stt_results(wait=True)`

## `_process_stt_results(wait)`

- Drain completed futures into transcripts
- Filter out noise and placeholders using a small blacklist and regex
- Maintain latency stats per chunk
- Register activity when valid speech is seen
- Force intermediate transcription when Whisper has energy but no text for too long
- Forward valid text to the LLM pipeline with a reference timestamp for later latency math

## `_llm_pipeline()`

- Consume LLM futures as they complete
- Record LLM latency and first‑event timings
- Split the response into short sentences
- Speak each sentence with `espeak_tts` and update TTS‑related metrics
- Stop the recorder during TTS if needed to avoid feedback

## Helpers

- `_register_activity()` — marks the most recent voice activity and wakes the main loop
- `_reset_awaiting_transcript_state()` — clears the force‑flush counters
- `_should_force_intermediate_transcription()` — decides when to flush STT buffers early
- `_queue_intermediate_transcription(reason)` — enqueues a non‑final STT finalize
- `_is_silent_chunk(audio_chunk)` — RMS‑based silence test
- `_request_stop(reason)` — sets stop flags, halts recorder, and wakes threads
- `_reference_timestamp_for_output(ts)` — computes the baseline timestamp for latency accounting
- `_print_stats(elapsed)` — prints human‑readable latency summary

## Handoff

From here the assistant relies on:
- `recorder.py` for audio capture
- `stt_faster.py` or `stt.py` for STT execution
- `llm_model.py` for LLM streaming and eSpeak interface
