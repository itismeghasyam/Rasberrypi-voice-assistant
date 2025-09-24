import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from pipeline.pipeline import ParallelVoiceAssistant


class FakeRecorder:
    def __init__(self, chunks: Tuple[np.ndarray, ...], sample_rate: int = 16000) -> None:
        self.chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        for chunk in chunks:
            self.chunk_queue.put(chunk)
        self.sample_rate = sample_rate
        self.recording = False
        self._stopped_at: Optional[float] = None

    def start(self) -> None:
        self.recording = True

    def get_chunk(self, timeout: float = 0.5) -> np.ndarray | None:
        if not self.recording and self.chunk_queue.empty():
            return None
        try:
            chunk = self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if self.chunk_queue.empty():
            self.recording = False
        return chunk

    def clear_queue(self) -> None:
        while True:
            try:
                self.chunk_queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        self.recording = False
        if self._stopped_at is None:
            self._stopped_at = time.time()


class FakeSTT:
    def __init__(
        self,
        responses: Dict[int, Tuple[float, Dict[str, object]]],
        emit_partials: bool = False,
    ) -> None:
        self._responses = responses
        self.emit_partials = emit_partials

    def submit_chunk(
        self,
        _audio_chunk: np.ndarray,
        chunk_id: int,
        *,
        skip_transcription: bool = False,
    ):
        from concurrent.futures import Future

        future: "Future[Dict[str, object]]" = Future()
        if skip_transcription:
            future.set_result({"chunk_id": chunk_id, "text": "", "is_final": False})
            return future

        delay, result = self._responses.get(chunk_id, (0.0, {}))
        if delay <= 0:
            future.set_result(result)
            return future

        future.set_running_or_notify_cancel()

        def _resolve() -> None:
            time.sleep(delay)
            future.set_result(result)

        threading.Thread(target=_resolve, daemon=True).start()
        return future

    def finalize(self, *_args, **_kwargs):
        return None

    def shutdown(self) -> None:
        pass


class FakeLLM:
    def process_incremental(self, _text: str, is_final: bool = False):
        return None

    def shutdown(self) -> None:
        pass


class FakeTTS:
    def __init__(self) -> None:
        self.speech_queue: "queue.Queue[object]" = queue.Queue()
        self.on_playback_start: Optional[Callable[[str, float], None]] = None
        self.on_playback_error: Optional[Callable[[], None]] = None

    def start_playback(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class RespondingLLM:
    def __init__(self, response: str, delay: float = 0.05) -> None:
        self.response = response
        self.delay = delay

    def process_incremental(self, text: str, is_final: bool = False):
        if not text.strip():
            return None

        from concurrent.futures import Future

        future: "Future[str]" = Future()

        def _resolve() -> None:
            time.sleep(self.delay)
            future.set_result(self.response)

        threading.Thread(target=_resolve, daemon=True).start()
        return future

    def shutdown(self) -> None:
        pass


class PendingTTS(FakeTTS):
    def __init__(self, delay: float = 0.3) -> None:
        super().__init__()
        self.delay = delay
        self.completed_times: List[float] = []

    def generate_and_queue(self, text: str, segment_id: int):
        from concurrent.futures import Future

        future: "Future[object]" = Future()

        def _resolve() -> None:
            time.sleep(self.delay)
            completion_time = time.time()
            self.completed_times.append(completion_time)
            future.set_result({"text": text, "segment_id": segment_id})
            callback = getattr(self, "on_playback_start", None)
            if callable(callback):
                callback(f"segment_{segment_id}.wav", completion_time)

        threading.Thread(target=_resolve, daemon=True).start()
        return future


class SilenceTimeoutTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

        self.whisper_exe = self.tmp_path / "whisper-cli"
        self.whisper_model = self.tmp_path / "model.bin"
        self.whisper_exe.write_text("#!/bin/sh\nexit 0\n")
        self.whisper_exe.chmod(0o755)
        self.whisper_model.write_bytes(b"0")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _create_assistant(self, **kwargs) -> ParallelVoiceAssistant:
        emit_partials = kwargs.get("emit_partials", False)
        assistant = ParallelVoiceAssistant(
            chunk_duration=kwargs.get("chunk_duration", 0.05),
            sample_rate=16000,
            stt_workers=1,
            whisper_exe=self.whisper_exe,
            whisper_model=self.whisper_model,
            emit_stt_partials=emit_partials,
            piper_model_path=self.tmp_path / "voice.onnx",
            use_subprocess_playback=False,
            silence_timeout=kwargs.get("silence_timeout", 0.35),
            silence_threshold=kwargs.get("silence_threshold", 200.0),
        )

        assistant.recorder = kwargs["recorder"]
        assistant.stt = kwargs["stt"]
        setattr(assistant.stt, "emit_partials", emit_partials)
        assistant.llm = kwargs.get("llm", FakeLLM())
        assistant.tts = kwargs.get("tts", FakeTTS())
        setattr(assistant.tts, "on_playback_start", assistant._on_tts_playback_start)
        setattr(assistant.tts, "on_playback_error", assistant._on_tts_playback_error)
        return assistant

    def test_emit_partials_false_stops_after_silence(self) -> None:
        speech = (np.ones(int(0.05 * 16000), dtype=np.int16) * 2000).astype(np.int16)
        recorder = FakeRecorder((speech,))
        stt = FakeSTT({
            0: (
                1.0,
                {"chunk_id": 0, "text": "hello there", "is_final": True},
            )
        })

        assistant = self._create_assistant(recorder=recorder, stt=stt)

        start = time.time()
        assistant.run(duration=3.0)
        elapsed = time.time() - start

        self.assertIsNotNone(assistant._recording_stop_time)
        assert assistant._recording_stop_time is not None
        stop_offset = assistant._recording_stop_time - assistant.stats.start_time
        self.assertLess(stop_offset, 0.75, msg=f"stop offset too large: {stop_offset:.3f}s")

        self.assertIsNotNone(assistant._first_voice_time)
        assert assistant._first_voice_time is not None
        first_voice_offset = assistant._first_voice_time - assistant.stats.start_time
        self.assertLess(first_voice_offset, 0.2, msg=f"first voice offset {first_voice_offset:.3f}s")

        self.assertIsNotNone(recorder._stopped_at)
        # Ensure the test does not stall excessively if logic regresses.
        self.assertLess(elapsed, 2.5)

    def test_noise_chunk_rolls_back_provisional_activity(self) -> None:
        speech = (np.ones(int(0.05 * 16000), dtype=np.int16) * 2000).astype(np.int16)
        recorder = FakeRecorder((speech,))
        stt = FakeSTT({
            0: (
                0.1,
                {"chunk_id": 0, "text": "wind blowing", "is_final": True},
            )
        })

        assistant = self._create_assistant(
            recorder=recorder,
            stt=stt,
            silence_threshold=100.0,
        )
        assistant._silent_chunks_before_stop = 1

        assistant.run(duration=2.0)

        self.assertTrue(assistant._consecutive_silent_chunks >= 1)
        self.assertIsNone(assistant._last_confirmed_voice_time)
        self.assertEqual(assistant._pending_activity, {})
        self.assertFalse(assistant._has_detected_speech)
        self.assertIsNone(assistant._first_voice_time)

    def _run_silence_with_pending_tts(self, emit_partials: bool) -> None:
        speech = (np.ones(int(0.05 * 16000), dtype=np.int16) * 2500).astype(np.int16)
        silence = np.zeros_like(speech)
        recorder = FakeRecorder((speech, silence, silence))

        stt = FakeSTT(
            {
                0: (
                    0.0,
                    {"chunk_id": 0, "text": "hello there", "is_final": True},
                )
            },
            emit_partials=emit_partials,
        )
        llm = RespondingLLM("Sure thing.", delay=0.05)
        tts = PendingTTS(delay=0.4)

        assistant = self._create_assistant(
            recorder=recorder,
            stt=stt,
            llm=llm,
            tts=tts,
            emit_partials=emit_partials,
            silence_threshold=150.0,
        )

        start = time.time()
        assistant.run(duration=3.0)
        elapsed = time.time() - start

        self.assertGreater(assistant.stats.tts_segments, 0)
        self.assertIsNotNone(assistant._recording_stop_time)
        assert assistant._recording_stop_time is not None
        stop_offset = assistant._recording_stop_time - assistant.stats.start_time
        self.assertLess(stop_offset, 0.8, msg=f"stop offset too large: {stop_offset:.3f}s")
        self.assertIsNotNone(assistant._stop_reason)
        assert assistant._stop_reason is not None
        self.assertIn("consecutive chunks", assistant._stop_reason)
        self.assertTrue(tts.completed_times)
        first_completion = min(tts.completed_times)
        self.assertLess(assistant._recording_stop_time, first_completion)
        self.assertLess(elapsed, 3.0)

    def test_silence_stops_recorder_with_pending_tts_no_partials(self) -> None:
        self._run_silence_with_pending_tts(emit_partials=False)

    def test_silence_stops_recorder_with_pending_tts_with_partials(self) -> None:
        self._run_silence_with_pending_tts(emit_partials=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
