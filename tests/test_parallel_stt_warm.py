import contextlib
import queue
import re
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from concurrent.futures import Future

from pipeline import ParallelVoiceAssistant, ParallelSTT, WarmWhisperWorker


class FakeRecorder:
    def __init__(self, chunks: Iterable[np.ndarray], sample_rate: int = 16000) -> None:
        prepared: List[np.ndarray] = [np.asarray(chunk, dtype=np.int16) for chunk in chunks]
        self._remaining = len(prepared)
        self.chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        for chunk in prepared:
            self.chunk_queue.put(chunk)
        self.sample_rate = sample_rate
        self.recording = False

    def start(self) -> None:
        self.recording = True

    def get_chunk(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        if not self.recording and self.chunk_queue.empty():
            return None
        try:
            chunk = self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        self._remaining = max(0, self._remaining - 1)
        if self._remaining <= 0:
            self.recording = False
        return chunk

    def clear_queue(self) -> None:
        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
            except queue.Empty:
                break
        self._remaining = 0

    def stop(self) -> None:
        self.recording = False


class DummyLLM:
    def __init__(self, response: str = "Acknowledged", delay: float = 0.02) -> None:
        self.response = response
        self.delay = delay

    def process_incremental(self, text: str, is_final: bool = False) -> Optional[Future]:
        clean = (text or "").strip()
        if not clean:
            return None
        future: "Future[str]" = Future()

        def _resolve() -> None:
            time.sleep(self.delay)
            future.set_result(self.response)

        threading.Thread(target=_resolve, daemon=True).start()
        return future

    def shutdown(self) -> None:
        pass


class DummyTTS:
    def __init__(self) -> None:
        self.speech_queue: "queue.Queue[object]" = queue.Queue()
        self.on_playback_start = None
        self.on_playback_error = None

    def start_playback(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def generate_and_queue(self, _text: str, _segment_id: int):
        return None


class FakeWarmBinding:
    def __init__(self, library: "FakeWarmLibrary") -> None:
        self.library = library
        self.transcribed: List[int] = []

    def transcribe_chunk(self, audio_bytes: bytes, chunk_id: int, sample_rate: int) -> str:
        self.library.chunk_calls.append(chunk_id)
        self.transcribed.append(chunk_id)
        return f"chunk-{chunk_id}"

    def finalize(self, audio_bytes: Optional[bytes], sample_rate: int) -> str:
        return " ".join(f"chunk-{cid}" for cid in self.transcribed)

    def reset(self) -> None:
        self.transcribed.clear()


class FakeWarmLibrary:
    def __init__(self) -> None:
        self.creations = 0
        self.chunk_calls: List[int] = []
        self.destroyed = 0

    def warm_whisper_create_worker(self, model_path: str, sample_rate: int, threads: int):
        self.creations += 1
        return FakeWarmBinding(self)

    def warm_whisper_destroy_worker(self, worker: FakeWarmBinding) -> None:
        self.destroyed += 1


class WarmFallbackTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

        self.whisper_exe = self.tmp_path / "whisper-cli"
        self.whisper_exe.write_text("#!/bin/sh\nexit 0\n")
        self.whisper_exe.chmod(0o755)

        self.whisper_model = self.tmp_path / "model.bin"
        self.whisper_model.write_bytes(b"0")

        self.piper_model = self.tmp_path / "voice.onnx"
        self.piper_model.write_bytes(b"0")

        self.fake_lib = self.tmp_path / "libwhisper_mock.so"
        self.fake_lib.write_bytes(b"0")

        chunk = np.full(int(0.05 * 16000), 1200, dtype=np.int16)
        self.chunks = (chunk.copy(), chunk.copy(), chunk.copy())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_assistant(self, use_warm: bool, cli_delay: float) -> float:
        recorder = FakeRecorder(self.chunks, sample_rate=16000)
        llm = DummyLLM(response="Roger", delay=0.02)
        tts = DummyTTS()

        def fake_run_whisper(stt_self: ParallelSTT, wav_path: Path, timeout: int = 60) -> str:
            name = wav_path.name
            if name.startswith("chunk"):
                match = re.search(r"chunk(\d+)", name)
                chunk_id = int(match.group(1)) if match else 0
                time.sleep(cli_delay)
                return f"chunk-{chunk_id}"
            if name.startswith("session"):
                time.sleep(cli_delay)
                count = len(stt_self._chunks)
                return " ".join(f"chunk-{idx}" for idx in range(count))
            return ""

        library = FakeWarmLibrary() if use_warm else None

        patches: List[Tuple[object, str, object]] = []
        patches.append((ParallelSTT, "_run_whisper", fake_run_whisper))
        patches.append((ParallelVoiceAssistant, "_is_silent_chunk", lambda _self, _chunk: False))

        if use_warm:
            patches.append((WarmWhisperWorker, "_resolve_library_path", lambda _exe: self.fake_lib))

        context_managers = []
        for target, attr, value in patches:
            cm = unittest.mock.patch.object(target, attr, value)
            context_managers.append(cm)

        if use_warm:
            context_managers.append(unittest.mock.patch("pipeline.ctypes.CDLL", return_value=library))
        else:
            context_managers.append(unittest.mock.patch.object(WarmWhisperWorker, "try_create", return_value=None))

        with contextlib.ExitStack() as stack:
            for cm in context_managers:
                stack.enter_context(cm)

            assistant = ParallelVoiceAssistant(
                chunk_duration=0.05,
                sample_rate=16000,
                stt_workers=1,
                whisper_exe=self.whisper_exe,
                whisper_model=self.whisper_model,
                whisper_threads=1,
                emit_stt_partials=True,
                piper_model_path=self.piper_model,
                use_subprocess_playback=False,
                silence_timeout=1.0,
                silence_threshold=100.0,
            )

            assistant.recorder = recorder
            assistant.stt.emit_partials = True
            assistant.llm = llm
            assistant.tts = tts
            assistant.tts.on_playback_start = assistant._on_tts_playback_start
            assistant.tts.on_playback_error = assistant._on_tts_playback_error

            if use_warm:
                self.assertIsNotNone(assistant.stt._warm_worker)

            assistant.run(duration=1.2)

        latency = assistant.stats.recording_to_first_llm_latency or 0.0
        return latency

    def _exercise_warm_stt(self) -> Tuple[FakeWarmLibrary, List[Dict[str, object]]]:
        chunk = np.full(int(0.05 * 16000), 1200, dtype=np.int16)
        library = FakeWarmLibrary()
        patches = [
            unittest.mock.patch.object(
                WarmWhisperWorker, "_resolve_library_path", lambda _exe: self.fake_lib
            ),
            unittest.mock.patch("pipeline.ctypes.CDLL", return_value=library),
        ]

        results: List[Dict[str, object]] = []
        with contextlib.ExitStack() as stack:
            for cm in patches:
                stack.enter_context(cm)

            stt = ParallelSTT(
                num_workers=1,
                sample_rate=16000,
                whisper_exe=self.whisper_exe,
                whisper_model=self.whisper_model,
                whisper_threads=1,
                emit_partials=True,
            )

            futures = [
                stt.submit_chunk(chunk, 0, skip_transcription=False),
                stt.submit_chunk(chunk, 1, skip_transcription=False),
            ]

            for future in futures:
                results.append(future.result(timeout=1.0))

            stt.shutdown()

        return library, results

    def test_ctypes_fallback_reuses_model_and_reduces_latency(self) -> None:
        library, results = self._exercise_warm_stt()
        self.assertEqual(library.creations, 1)
        self.assertEqual(library.chunk_calls, [0, 1])
        self.assertTrue(all((res.get("text") or "").startswith("chunk-") for res in results))

        warm_latency = self._run_assistant(use_warm=True, cli_delay=0.02)
        cli_latency = self._run_assistant(use_warm=False, cli_delay=0.12)

        self.assertGreater(cli_latency, warm_latency)
        self.assertGreater(cli_latency, 0.0)


if __name__ == "__main__":
    unittest.main()
