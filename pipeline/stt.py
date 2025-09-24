from typing import Optional, Any,  Dict, List, Optional
import shutil, threading, os, ctypes, numpy as np
import re,tempfile,subprocess,wave,math,contextlib
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor

from config import SAMPLE_RATE, WHISPER_EXE, WHISPER_MODEL



class WarmWhisperWorker:
    """Provide a persistent whisper.cpp worker for chunked transcription."""

    def __init__(self, binding: "_WarmBindingProtocol", sample_rate: int) -> None:
        self._binding = binding
        self._sample_rate = int(sample_rate)
        self._lock = threading.Lock()

    @classmethod
    def try_create(
        cls,
        whisper_exe: Path,
        whisper_model: Path,
        sample_rate: int,
        threads: Optional[int] = None,
    ) -> Optional["WarmWhisperWorker"]:
        """Attempt to construct a warm worker using python bindings or ctypes."""

        binding: Optional[_WarmBindingProtocol] = None

        

        if binding is None:
            binding = cls._create_ctypes_binding(whisper_exe, whisper_model, sample_rate, threads)

        if binding is None:
            return None

        return cls(binding, sample_rate)

    @staticmethod
    def _create_python_binding(
        whisper_model: Path, sample_rate: int, threads: Optional[int]
    ) -> Optional["_WarmBindingProtocol"]:
        try:
            from whispercpp import Whisper  # type: ignore[import]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("whispercpp python binding not available") from exc

        try:
            model = Whisper.from_pretrained(
                whisper_model.name,
                basedir=str(whisper_model.parent),
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("whispercpp model load failed") from exc

        return _WhisperPythonBinding(model, sample_rate, threads)

    @classmethod
    def _create_ctypes_binding(
        cls,
        whisper_exe: Path,
        whisper_model: Path,
        sample_rate: int,
        threads: Optional[int],
    ) -> Optional["_WarmBindingProtocol"]:
        lib_path = cls._resolve_library_path(whisper_exe)
        if lib_path is None:
            return None

        try:
            library = ctypes.CDLL(str(lib_path))
        except OSError as exc:
            if os.environ.get("WARM_WHISPER_DEBUG"):
                print(f"[WARM] Failed loading {lib_path}: {exc}")
            return None

        worker_obj: Any
        create_fn = getattr(library, "warm_whisper_create_worker", None)
        if callable(create_fn):
            try:
                worker_obj = create_fn(str(whisper_model), int(sample_rate), int(threads or 0))
            except Exception as exc:
                if os.environ.get("WARM_WHISPER_DEBUG"):
                    print(f"[WARM] warm_whisper_create_worker failed: {exc}")
                return None
        else:
            worker_obj = library

        

    @staticmethod
    def _resolve_library_path(whisper_exe: Path) -> Optional[Path]:
        env_vars = [
            os.environ.get("WHISPER_CPP_LIB"),
            os.environ.get("WHISPER_LIBWHISPER_PATH"),
            os.environ.get("WHISPER_LIB"),
        ]
        for candidate in env_vars:
            if not candidate:
                continue
            path = Path(candidate)
            if path.exists():
                return path

        exe_path = Path(whisper_exe)
        search_roots = [exe_path.parent, exe_path.parent.parent]
        if exe_path.parent.parent != exe_path.parent:
            search_roots.append(exe_path.parent.parent.parent)
        search_roots = [p for p in search_roots if p and p.exists()]

        lib_names = ["libwhisper.so", "libwhisper.dylib", "whisper.dll", "libwarmwhisper.so"]
        for root in search_roots:
            for variant in [root, root / "lib", root / "bin"]:
                if not variant.exists():
                    continue
                for name in lib_names:
                    candidate = variant / name
                    if candidate.exists():
                        return candidate
        return None

    def transcribe_chunk(self, audio_bytes: bytes, chunk_id: int) -> str:
        with self._lock:
            return self._binding.transcribe_chunk(audio_bytes, chunk_id)

    def finalize(self, audio_bytes: Optional[bytes]) -> str:
        with self._lock:
            return self._binding.finalize(audio_bytes)

    def reset(self) -> None:
        with self._lock:
            self._binding.reset()

    def shutdown(self) -> None:
        with self._lock:
            self._binding.shutdown()

class _WarmBindingProtocol:
    def transcribe_chunk(self, audio_bytes: bytes, chunk_id: int) -> str:  # pragma: no cover - protocol
        raise NotImplementedError

    def finalize(self, audio_bytes: Optional[bytes]) -> str:  # pragma: no cover - protocol
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover - protocol
        raise NotImplementedError

    def shutdown(self) -> None:  # pragma: no cover - protocol
        raise NotImplementedError


class _WhisperPythonBinding(_WarmBindingProtocol):
    def __init__(self, model: Any, sample_rate: int, threads: Optional[int]) -> None:
        self._model = model
        self._sample_rate = int(sample_rate)
        self._threads = max(1, int(threads or 1))
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_byte_length = 0

        self._aggregated_text = ""


    def _convert_pcm(self, audio_bytes: bytes) -> np.ndarray:
        if not audio_bytes:
            return np.empty(0, dtype=np.float32)
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        if pcm.size == 0:
            return np.empty(0, dtype=np.float32)
        return pcm.astype(np.float32) / 32768.0

    def _extract_text(self, segments: Any) -> str:
        if isinstance(segments, str):
            return segments
        try:
            return " ".join(getattr(seg, "text", "") for seg in segments).strip()
        except Exception:
            return ""

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _lexical_tokens(text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"[\w']+", text.lower())

    def _compute_incremental_text(self, previous: str, current: str) -> str:
        current = current or ""
        previous = previous or ""

        normalized_previous = self._normalize_whitespace(previous)
        normalized_current = self._normalize_whitespace(current)

        if not normalized_current:
            return ""

        previous_tokens = self._lexical_tokens(normalized_previous)
        current_tokens = self._lexical_tokens(normalized_current)

        common_len = 0
        for prev_token, curr_token in zip(previous_tokens, current_tokens):
            if prev_token != curr_token:
                break
            common_len += 1

        if common_len >= len(current_tokens):
            return ""

        matches = list(re.finditer(r"[\w']+", current))
        if common_len < len(matches):
            start = matches[common_len].start()
            return current[start:].lstrip()

        return current.strip()

    def _merge_with_increment(self, existing: str, addition: str) -> str:
        existing = (existing or "").strip()
        addition = (addition or "").strip()
        if not addition:
            return existing
        if not existing:
            return addition

        tentative = f"{existing} {addition}".strip()
        increment = self._compute_incremental_text(existing, tentative)
        if not increment:
            return existing
        return f"{existing} {increment}".strip()

    def transcribe_chunk(self, audio_bytes: bytes, chunk_id: int) -> str:
        pcm = self._convert_pcm(audio_bytes)
        if pcm.size == 0:
            return ""

        if self._buffer.size == 0:
            self._buffer = pcm.copy()
        else:
            self._buffer = np.concatenate([self._buffer, pcm])
        self._buffer_byte_length += len(audio_bytes)

        candidate_full_text = self._aggregated_text

        try:
            segments = self._model.transcribe(pcm, num_proc=self._threads)
        except Exception:
            segments = ""

        primary_text = self._extract_text(segments).strip()
        if primary_text:
            candidate_full_text = self._merge_with_increment(self._aggregated_text, primary_text)
        else:
            try:
                fallback_segments = self._model.transcribe(self._buffer, num_proc=self._threads)
            except Exception:
                fallback_segments = ""
            fallback_text = self._extract_text(fallback_segments).strip()
            if fallback_text:
                candidate_full_text = fallback_text

        new_text = self._compute_incremental_text(self._aggregated_text, candidate_full_text)
        self._aggregated_text = candidate_full_text
        return new_text

    def finalize(self, audio_bytes: Optional[bytes]) -> str:
        if audio_bytes:
            pcm = self._convert_pcm(audio_bytes)
            if pcm.size:
                total_bytes = len(audio_bytes)
                if self._buffer.size == 0 or total_bytes != self._buffer_byte_length:
                    self._buffer = pcm.copy()
                    self._buffer_byte_length = total_bytes
        if self._buffer.size == 0:
            self._aggregated_text = ""
            return ""
        try:
            segments = self._model.transcribe(self._buffer, num_proc=self._threads)
        except Exception:
            return ""
        final_text = self._extract_text(segments).strip()
        self._aggregated_text = final_text
        return final_text

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_byte_length = 0

        self._aggregated_text = ""


    def shutdown(self) -> None:
        close = getattr(self._model, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()


class _CtypesWarmBinding(_WarmBindingProtocol):
    def __init__(
        self,
        worker: Any,
        library: Any,
        sample_rate: int,
        threads: Optional[int],
    ) -> None:
        self._worker = worker
        self._library = library
        self._sample_rate = int(sample_rate)
        self._threads = int(threads or 0)

    def transcribe_chunk(self, audio_bytes: bytes, chunk_id: int) -> str:
        if hasattr(self._worker, "transcribe_chunk"):
            return str(
                self._worker.transcribe_chunk(audio_bytes, int(chunk_id), self._sample_rate)
            )
        if hasattr(self._worker, "transcribe"):
            return str(self._worker.transcribe(audio_bytes, int(chunk_id), self._sample_rate))
        raise RuntimeError("warm whisper binding lacks a transcribe function")

    def finalize(self, audio_bytes: Optional[bytes]) -> str:
        if hasattr(self._worker, "finalize"):
            return str(self._worker.finalize(audio_bytes, self._sample_rate))
        return ""

    def reset(self) -> None:
        if hasattr(self._worker, "reset"):
            self._worker.reset()

    def shutdown(self) -> None:
        destroy = getattr(self._library, "warm_whisper_destroy_worker", None)
        if callable(destroy):
            with contextlib.suppress(Exception):
                destroy(self._worker)
            return

        close = getattr(self._worker, "close", None) or getattr(self._worker, "destroy", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()



# Parallel Speech-to-Text (whisper.cpp)
class ParallelSTT:

    """Process audio chunks asynchronously using the whisper.cpp CLI."""

    def __init__(
        self,
        num_workers: int = 2,

        sample_rate: int = SAMPLE_RATE,
        whisper_exe: Path = WHISPER_EXE,
        whisper_model: Path = WHISPER_MODEL,
        whisper_threads: Optional[int] = None,
        emit_partials: bool = False,
    ) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max(1, num_workers))
        self.sample_rate = int(sample_rate)
        self.whisper_exe = Path(whisper_exe)
        self.whisper_model = Path(whisper_model)
        self.whisper_threads = int(whisper_threads) if whisper_threads else None
        self.emit_partials = bool(emit_partials)

        self._warm_worker: Optional[WarmWhisperWorker] = None

        self._chunks: List[bytes] = []
        self._transcript_lock = threading.Lock()
        self._emitted_transcript = ""
        self._last_partial = ""
        self._temp_dir = Path(tempfile.mkdtemp(prefix="pipeline_stt_"))
        
        self._persistent_proc: Optional[subprocess.Popen] = None
        self._persistent_lock = threading.Lock()


        if not self.whisper_exe.exists():
            raise FileNotFoundError(f"Whisper binary not found: {self.whisper_exe}")
        if not self.whisper_model.exists():
            raise FileNotFoundError(f"Whisper model not found: {self.whisper_model}")

        if self.emit_partials:
            self._warm_worker = WarmWhisperWorker.try_create(
                self.whisper_exe,
                self.whisper_model,
                self.sample_rate,
                self.whisper_threads,
            )
            

    # --------------------- Whisper helpers -----------------------

    def _write_wav(self, audio_bytes: bytes, prefix: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", prefix=f"{prefix}_", dir=self._temp_dir, delete=False
        )
        tmp_path = Path(tmp.name)
        tmp.close()
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_bytes)
        return tmp_path
    
    def _run_whisper(self, wav_path: Path, timeout: int = 60) -> str:
        """
        Dispatch to persistent whisper worker when partials are enabled,
        otherwise fall back to one-shot subprocess run.
        """
        if self.emit_partials:   
            text = self._run_whisper_persistent(wav_path, timeout=timeout)
            if text:
                return text

        
        return self._run_whisper_once(wav_path, timeout=timeout)


    def _run_whisper_once(self, wav_path: Path, timeout: int = 60) -> str:
        cmd = [
            str(self.whisper_exe),
            "-m",
            str(self.whisper_model),
            "-f",
            str(wav_path),
            "--no-prints",
            "--output-txt",
        ]
        if self.whisper_threads:
            cmd += ["-t", str(self.whisper_threads)]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print("[STT][Whisper] Transcription timed out")
            return ""
        except Exception as exc:
            print(f"[STT][Whisper] Error running whisper.cpp: {exc}")
            return ""

        text = ""
        candidates = [wav_path.with_suffix(".txt"), Path(str(wav_path) + ".txt")]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception as exc:
                print(f"[STT][Whisper] Failed reading {candidate.name}: {exc}")
                text = ""
            finally:
                candidate.unlink(missing_ok=True)
            if text:
                break

        return text
    
    def _ensure_persistent_whisper(self) -> Optional[subprocess.Popen]:
        if self._persistent_proc and self._persistent_proc.poll() is None:
            return self._persistent_proc

        cmd = [
            str(self.whisper_exe),
            "-m", str(self.whisper_model),
            "-f", "-",          # stdin mode
            "--no-prints",
            "--output-txt",
        ]
        if self.whisper_threads:
            cmd += ["-t", str(self.whisper_threads)]

        self._persistent_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        return self._persistent_proc
    
    def _run_whisper_persistent(self, wav_path: Path, timeout: int = 10) -> str:
        proc = self._ensure_persistent_whisper()
        if not proc or not proc.stdin or not proc.stdout:
            return ""

        # feed bytes
        try:
            proc.stdin.write(wav_path.read_bytes())
            proc.stdin.flush()
        except Exception as e:
            print("[STT] Persistent whisper write failed:", e)
            return ""

        # read back one line
        try:
            line = proc.stdout.readline().decode("utf-8").strip()
            return line
        except Exception as e:
            print("[STT] Persistent whisper read failed:", e)
            return ""



    def _process_chunk_whisper(self, audio_bytes: bytes, chunk_id: int) -> Dict[str, Any]:
        text: str = ""
        if self._warm_worker is not None:
            try:
                text = self._warm_worker.transcribe_chunk(audio_bytes, chunk_id)
            except Exception as exc:
                print(f"[STT][Whisper] Warm worker chunk {chunk_id} failed: {exc}")
                text = ""
        else:
            wav_path = self._write_wav(audio_bytes, f"chunk{chunk_id}")
            try:
                text = self._run_whisper(wav_path, timeout=45)
            finally:
                wav_path.unlink(missing_ok=True)

        text = (text or "").strip()
        if not text:
            return {"chunk_id": chunk_id, "text": "", "is_final": False}

        with self._transcript_lock:
            if text.lower() == self._last_partial.lower():
                return {"chunk_id": chunk_id, "text": "", "is_final": False}
            self._last_partial = text
            self._emitted_transcript = (f"{self._emitted_transcript} {text}").strip()

        return {"chunk_id": chunk_id, "text": text, "is_final": False}

    def _finalize_whisper(self, chunk_id: int, mark_final: bool) -> Dict[str, Any]:
        with self._transcript_lock:
            chunks = list(self._chunks)
            emitted = self._emitted_transcript.strip()
            self._last_partial = ""
            self._chunks.clear()

        if not chunks:
            return {"chunk_id": chunk_id, "text": "", "is_final": True}

        audio_bytes = b"".join(chunks)
        total_bytes = len(audio_bytes)
        approx_seconds = total_bytes / max(1, self.sample_rate * 2)
        adaptive_timeout = max(10, min(45, int(math.ceil(max(approx_seconds, 0.5) * 6))))

        full_text = ""
        if self._warm_worker is not None:
            try:
                full_text = (self._warm_worker.finalize(audio_bytes) or "").strip()
            except Exception as exc:
                print(f"[STT][Whisper] Warm worker finalize failed: {exc}")
                full_text = ""
        else:
            wav_path = self._write_wav(audio_bytes, f"session{chunk_id}")
            try:
                full_text = (
                    self._run_whisper(wav_path, timeout=adaptive_timeout) or ""
                ).strip()
            finally:
                wav_path.unlink(missing_ok=True)

        collapsed_emitted = re.sub(r"\s+", " ", emitted).strip()
        collapsed_full = re.sub(r"\s+", " ", full_text).strip()

        def _lexical_tokens(text: str) -> List[str]:
            if not text:
                return []
            return re.findall(r"[\w']+", text.lower())

        emitted_tokens = _lexical_tokens(collapsed_emitted)
        full_tokens = _lexical_tokens(collapsed_full)

        common_len = 0
        for emitted_token, full_token in zip(emitted_tokens, full_tokens):
            if emitted_token != full_token:
                break
            common_len += 1

        tokens_match_prefix = common_len == len(emitted_tokens)

        new_text = full_text.strip()

        if tokens_match_prefix and len(full_tokens) == len(emitted_tokens):
            new_text = ""
        elif tokens_match_prefix and len(full_tokens) > len(emitted_tokens):
            token_matches = list(re.finditer(r"[\w']+", full_text))
            if token_matches and common_len < len(token_matches):
                start = token_matches[common_len].start()
                new_text = full_text[start:].lstrip()
            else:
                new_text = full_text.strip()
        elif not full_text:
            new_text = ""

        with self._transcript_lock:
            self._emitted_transcript = full_text

        return {"chunk_id": chunk_id, "text": new_text, "is_final": mark_final}

    def _empty_chunk(self, chunk_id: int) -> Dict[str, Any]:
        return {"chunk_id": chunk_id, "text": "", "is_final": False}


    # -------------------------- Public API -----------------------

    def submit_chunk(
        self,
        audio_chunk: np.ndarray,
        chunk_id: int,
        *,
        skip_transcription: bool = False,
    ) -> Future:

        audio_bytes = np.ascontiguousarray(audio_chunk, dtype=np.int16).tobytes()

        if skip_transcription:
            future: "Future[Dict[str, Any]]" = Future()
            future.set_result({"chunk_id": chunk_id, "text": "", "is_final": False})
            return future

        with self._transcript_lock:
            self._chunks.append(audio_bytes)

        if not self.emit_partials:
            return self.executor.submit(self._empty_chunk, chunk_id)
        return self.executor.submit(self._process_chunk_whisper, audio_bytes, chunk_id)

    def finalize(self, chunk_id: int, mark_final: bool = True) -> Optional[Future]:
        return self.executor.submit(self._finalize_whisper, chunk_id, mark_final)

    def reset(self) -> None:
        with self._transcript_lock:
            self._chunks.clear()
            self._emitted_transcript = ""
            self._last_partial = ""
        if self._warm_worker is not None:
            self._warm_worker.reset()

    def shutdown(self) -> None:
        if self._warm_worker is not None:
            self._warm_worker.shutdown()
        self.executor.shutdown(wait=False)
        with contextlib.suppress(Exception):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

