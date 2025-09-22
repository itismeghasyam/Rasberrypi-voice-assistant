import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from pipeline import ParallelVoiceAssistant


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
    def __init__(self, responses: Dict[int, Tuple[float, Dict[str, object]]]) -> None:
        self._responses = responses

    def submit_chunk(self, _audio_chunk: np.ndarray, chunk_id: int):
        delay, result = self._responses.get(chunk_id, (0.0, {}))
        future: "Future[Dict[str, object]]"
        from concurrent.futures import Future

        future = Future()
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

    def start_playback(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


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
        assistant = ParallelVoiceAssistant(
            chunk_duration=kwargs.get("chunk_duration", 0.05),
            sample_rate=16000,
            stt_workers=1,
            whisper_exe=self.whisper_exe,
            whisper_model=self.whisper_model,
            emit_stt_partials=False,
            piper_model_path=self.tmp_path / "voice.onnx",
            use_subprocess_playback=False,
            silence_timeout=kwargs.get("silence_timeout", 0.35),
            silence_threshold=kwargs.get("silence_threshold", 200.0),
        )

        assistant.recorder = kwargs["recorder"]
        assistant.stt = kwargs["stt"]
        assistant.llm = FakeLLM()
        assistant.tts = FakeTTS()
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
