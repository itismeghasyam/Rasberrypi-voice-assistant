# pipeline/llm_model.py

## Purpose
Helpers for the local LLM and the tiny TTS shim. Provides:
- a timing parser for llama.cpp output
- a small `espeak` wrapper with elapsed time measurement
- a direct llama.cpp caller `llama110(...)` with sensible defaults and resource sampling
- `StreamingLLM` which buffers user text, decides when to answer, and runs the model asynchronously

---

## Imports at a glance
- `time`, `subprocess`, `shlex`, `re`, `os`, `Path` for timing, process exec, parsing, and paths 
- `ResourceSampler` to track CPU and memory during a call  
- `ThreadPoolExecutor`, `Future`, typing helpers for async work and signatures 

---

## Function `_parse_llm_timing(raw_text) -> Optional[float]`
Best effort extraction of a total inference time in seconds from llama.cpp logs.  
How it works:
- scans for several patterns like `inference time`, `model time`, `total time`, `elapsed`, or `ms per token`  
- converts ms to seconds and returns the first positive hit 

Returned value is `None` if nothing matches. Useful for analytics and latency summaries.

---

## Function `speak_text_timed(text, cmd=None) -> float`
Minimal TTS helper that runs a command and returns elapsed seconds.  
Defaults to `["espeak", text]`. You can also pass a format string or list to integrate a different TTS. The helper safely expands the command, runs it, and returns the measured duration; errors are swallowed so the pipeline keeps going. 

---

## Function `llama110(...) -> Dict`
A direct call to `llama.cpp` that returns a structured result including text, timings, token count, and resource stats.

Key defaults
- If `llama_cli_path` is not supplied, uses `~/llama.cpp/build/bin/llama-cli`  
- If `model_path` is not supplied, uses `~/Downloads/llama2.c-stories110M-pruned50.Q3_K_M.gguf` 

Safety checks
- Raises if the binary or model path does not exist 

Command it builds
- `llama-cli -m <model> -p <prompt> -n <n_predict> -t <threads> --temp <temperature> -no-cnv --single-turn` 

Timing and sampling
- Starts a `ResourceSampler`, runs the subprocess with a timeout, then stops sampling; on timeout it returns partial info with captured stdout and stderr 
- Wall clock time is recorded as `model_reply_time` 
- It removes the echoed prompt if present, falls back to stderr if stdout is empty, and counts tokens with a simple split  
- Inference time is parsed using `_parse_llm_timing(...)` and resource stats come from `sampler.summary()` 

Optional TTS
- If `tts_after=True` it calls `speak_text_timed(...)` on the generated string and records `tts_time` 

Returned dict fields
- `generated`, `model_reply_time`, `model_inference_time`, `tokens`, `tts_time`, `resource`, `raw_stdout`, `raw_stderr` 

---

## Class `StreamingLLM`
Accumulates recognized text until it makes sense to answer, then calls `llama110(...)` on a worker thread and returns a `Future`.

### Init
- Stores `llama_kwargs` and sets sensible defaults `n_predict=12`, `threads=os.cpu_count() or 4`, `temperature=0.6` 
- Creates a single worker `ThreadPoolExecutor` and an internal buffer list 

### `process_incremental(text_chunk, is_final=False) -> Optional[Future]`
- Trims the chunk, appends to the buffer, and decides whether to generate now  
- Triggers if the chunk is final or if `_should_respond()` says it looks complete  
- When generating, it joins the buffer into a prompt and submits `_generate_response(...)` to the executor; otherwise returns `None` 

### `_should_respond() -> bool`
- Returns `True` if the buffer contains a sentence end like `. ? !` or if the buffer has at least 6 words 

### `_generate_response(text) -> str`
- Prepends `Answer concisely:` to keep outputs short  
- Builds call kwargs from `llama_kwargs` including paths, `n_predict`, `threads`, `temperature`, and a timeout  
- Calls `llama110(...)` and handles errors with a friendly string  
- Cleans the model output and returns it 

### `_clean_response(response) -> str`
- Strips whitespace and leading quotes or stray question mark artifacts, then returns a neat sentence 

### `shutdown() -> None`
- Sets a closed flag and shuts down the executor without waiting 

---

## Typical usage
The orchestrator constructs `StreamingLLM(llama_kwargs=...)`. As transcripts arrive it calls:
1. `future = llm.process_incremental(text_chunk, is_final=...)`  
2. If `future` is not `None`, later `future.result()` returns the cleaned response for TTS.

---

## Notes
- Default model is **Llama 2 C Stories 110M pruned50 Q3_K_M GGUF** which is small and snappy on the Pi. You can override both model path and `llama-cli` path via `llama_kwargs` or CLI flags that the main entrypoint forwards into `llama_kwargs` 
