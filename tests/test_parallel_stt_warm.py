<<<<<<< HEAD
import queue
=======
import contextlib
import queue
import re
>>>>>>> a49d41338211c5b2566cbb668fce64e98358e935
import tempfile
import threading
import time
import unittest
<<<<<<< HEAD
from pathlib import Path
from typing import Optional

import numpy as np
from unittest import mock

from pipeline import ParallelSTT, ParallelVoiceAssistant


class FakeRecorder:
    def __init__(self, chunks: tuple[np.ndarray, ...], sample_rate: int = 16000) -> None:
        self.chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        for chunk in chunks:
            self.chunk_queue.put(chunk)
        self.sample_rate = sample_rate
        self.recording = False
        self._stopped_at: Optional[float] = None
=======
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
>>>>>>> a49d41338211c5b2566cbb668fce64e98358e935

    def start(self) -> None:
        self.recording = True

    def get_chunk(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        if not self.recording and self.chunk_queue.empty():
            return None
        try:
            chunk = self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None
<<<<<<< HEAD
        if self.chunk_queue.empty():
=======
        self._remaining = max(0, self._remaining - 1)
        if self._remaining <= 0:
>>>>>>> a49d41338211c5b2566cbb668fce64e98358e935
            self.recording = False
        return chunk

    def clear_queue(self) -> None:
<<<<<<< HEAD
        while True:
=======
        while not self.chunk_queue.empty():
>>>>>>> a49d41338211c5b2566cbb668fce64e98358e935
            try:
                self.chunk_queue.get_nowait()
            except queue.Empty:
                break
<<<<<<< HEAD

    def stop(self) -> None:
        self.recording = False
        if self._stopped_at is None:
            self._stopped_at = time.time()


class FakeTTS:
=======
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
>>>>>>> a49d41338211c5b2566cbb668fce64e98358e935
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

<<<<<<< HEAD
    def generate_and_queue(self, text: str, segment_id: int):
        from concurrent.futures import Future

        future: "Future[object]" = Future()
        future.set_result({"text": text, "segment_id": segment_id})
        callback = getattr(self, "on_playback_start", None)

        if callable(callback):

            def _fire_callback() -> None:
                time.sleep(0.01)
                callback(f"segment_{segment_id}.wav", time.time())

            threading.Thread(target=_fire_callback, daemon=True).start()
        return future


class RespondingLLM:
    def __init__(self, response: str, delay: float = 0.05) -> None:
        self._response = response
        self._delay = delay

    def process_incremental(self, text: str, is_final: bool = False):
        if not text.strip():
            return None

        from concurrent.futures import Future

        future: "Future[str]" = Future()

        def _resolve() -> None:
            time.sleep(self._delay)
            future.set_result(self._response)

        threading.Thread(target=_resolve, daemon=True).start()
        return future

    def shutdown(self) -> None:
        pass


class FakeWarmWorker:
    def __init__(self) -> None:
        self.transcribe_calls = 0
        self.finalize_calls = 0
        self._current = ""

    def transcribe_chunk(self, _audio_bytes: bytes) -> str:
        self.transcribe_calls += 1
        if self.transcribe_calls == 1:
            self._current = "hello there."
        else:
            self._current = "hello there."
        return self._current

    def finalize(self) -> str:
        self.finalize_calls += 1
        return self._current

    def reset(self) -> None:
        self._current = ""

    def shutdown(self) -> None:
        pass


class SlowSTT:
    def __init__(self, delay: float = 0.3) -> None:
        self.delay = delay
        self.emit_partials = True

    def submit_chunk(
        self,
        _audio_chunk: np.ndarray,
        chunk_id: int,
        *,
        skip_transcription: bool = False,
    ):
        from concurrent.futures import Future

        future: "Future[dict[str, object]]" = Future()
        future.set_running_or_notify_cancel()

        if skip_transcription:
            future.set_result({"chunk_id": chunk_id, "text": "", "is_final": False})
            return future

        def _resolve() -> None:
            time.sleep(self.delay)
            payload = {"chunk_id": chunk_id, "text": "", "is_final": False}
            if chunk_id == 0:
                payload = {"chunk_id": chunk_id, "text": "hello there.", "is_final": True}
            if not future.cancelled():
                future.set_result(payload)

        threading.Thread(target=_resolve, daemon=True).start()
        return future

    def finalize(self, *_args, **_kwargs):
        return None

    def reset(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class WarmWhisperIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        self.whisper_exe = tmp_path / "whisper-cli"
        self.whisper_model = tmp_path / "model.bin"
        self.whisper_exe.write_text("#!/bin/sh\nexit 0\n")
        self.whisper_exe.chmod(0o755)
        self.whisper_model.write_bytes(b"0")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build_assistant(
        self,
        recorder: FakeRecorder,
        stt_override: Optional[object] = None,
        *,
        silence_timeout: float = 0.5,
    ) -> ParallelVoiceAssistant:
        assistant = ParallelVoiceAssistant(
            chunk_duration=0.05,
            sample_rate=16000,
            stt_workers=1,
            whisper_exe=self.whisper_exe,
            whisper_model=self.whisper_model,
            emit_stt_partials=True,
            piper_model_path=Path(self._tmp.name) / "voice.onnx",
            use_subprocess_playback=False,
            silence_threshold=100.0,
            silence_timeout=silence_timeout,
        )

        assistant.recorder = recorder
        assistant.llm = RespondingLLM("Sure.", delay=0.02)
        assistant.tts = FakeTTS()
        setattr(assistant.tts, "on_playback_start", assistant._on_tts_playback_start)
        setattr(assistant.tts, "on_playback_error", assistant._on_tts_playback_error)
        if stt_override is not None:
            assistant.stt.shutdown()
            assistant.stt = stt_override
        setattr(assistant.stt, "emit_partials", True)
        return assistant

    def test_parallel_stt_uses_single_warm_worker_instance(self) -> None:
        worker = FakeWarmWorker()

        speech = (np.ones(int(0.05 * 16000), dtype=np.int16) * 2000).astype(np.int16)

        with mock.patch("pipeline.WarmWhisperWorker.try_create", return_value=worker), mock.patch.object(
            ParallelSTT, "_run_whisper", side_effect=AssertionError("should not call whisper-cli")
        ):
=======
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

>>>>>>> a49d41338211c5b2566cbb668fce64e98358e935
            stt = ParallelSTT(
                num_workers=1,
                sample_rate=16000,
                whisper_exe=self.whisper_exe,
                whisper_model=self.whisper_model,
<<<<<<< HEAD
                emit_partials=True,
            )

            first = stt.submit_chunk(speech, 0)
            second = stt.submit_chunk(speech, 1)

            first_result = first.result(timeout=1.0)
            second_result = second.result(timeout=1.0)

            finalize_future = stt.finalize(2, mark_final=True)
            assert finalize_future is not None
            finalize_result = finalize_future.result(timeout=1.0)

            stt.shutdown()

        self.assertEqual(worker.transcribe_calls, 2)
        self.assertEqual(worker.finalize_calls, 1)
        self.assertEqual(first_result.get("text"), "hello there.")
        self.assertEqual(second_result.get("text"), "")
        self.assertEqual(finalize_result.get("text"), "")

    def test_warm_worker_reduces_recording_to_first_llm_latency(self) -> None:
        speech = (np.ones(int(0.05 * 16000), dtype=np.int16) * 2500).astype(np.int16)
        warm_recorder = FakeRecorder((speech,))
        cold_recorder = FakeRecorder((speech,))

        worker = FakeWarmWorker()

        with mock.patch("pipeline.WarmWhisperWorker.try_create", return_value=worker), mock.patch.object(
            ParallelSTT, "_run_whisper", side_effect=AssertionError("should not invoke CLI when warm worker is active")
        ):
            warm_assistant = self._build_assistant(warm_recorder)
            start = time.time()
            warm_assistant.run(duration=1.5)
            warm_elapsed = time.time() - start
            warm_latency = warm_assistant.stats.recording_to_first_llm_latency

        self.assertIsNotNone(warm_latency)
        assert warm_latency is not None
        self.assertLess(warm_elapsed, 1.5)

        cold_stt = SlowSTT(delay=0.05)
        cold_assistant = self._build_assistant(
            cold_recorder, stt_override=cold_stt, silence_timeout=1.2
        )
        start = time.time()
        cold_assistant.run(duration=1.5)
        cold_elapsed = time.time() - start
        cold_latency = cold_assistant.stats.recording_to_first_llm_latency

        self.assertIsNotNone(cold_latency)
        assert cold_latency is not None
        self.assertLess(cold_elapsed, 1.6)
        self.assertGreater(cold_latency, warm_latency + 0.03)


if __name__ == "__main__":  # pragma: no cover
=======
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
>>>>>>> a49d41338211c5b2566cbb668fce64e98358e935
    unittest.main()
