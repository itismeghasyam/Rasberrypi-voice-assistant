import json
import queue
import sys
import tempfile
import unittest
from pathlib import Path

from pipeline.pipeline import BufferedTTS


FAKE_SAMPLE_RATE = 16000


def _write_fake_piper_script(path: Path) -> None:
    script = """#!/usr/bin/env python3
import argparse
import math
import struct
import sys
import time

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-m", "--model")
parser.add_argument("--output-raw", action="store_true")
parser.add_argument("--speaker")
_, _ = parser.parse_known_args()

SAMPLE_RATE = {sample_rate}


def synthesize(text: str) -> None:
    duration = 0.05 + 0.01 * len(text)
    total = max(1, int(SAMPLE_RATE * duration))
    buf = bytearray()
    for i in range(total):
        angle = 2.0 * math.pi * 220.0 * i / SAMPLE_RATE
        sample = int(1200 * math.sin(angle))
        buf.extend(struct.pack("<h", sample))
    midpoint = len(buf) // 2
    sys.stdout.buffer.write(buf[:midpoint])
    sys.stdout.buffer.flush()
    time.sleep(0.02)
    sys.stdout.buffer.write(buf[midpoint:])
    sys.stdout.buffer.flush()


try:
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        synthesize(text)
except BrokenPipeError:
    pass
""".format(sample_rate=FAKE_SAMPLE_RATE)
    path.write_text(script)
    path.chmod(0o755)


class BufferedTTSTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _create_voice_files(self) -> Path:
        model_path = self.tmp_path / "voice.onnx"
        model_path.write_bytes(b"0")
        metadata = {
            "audio": {
                "sample_rate": FAKE_SAMPLE_RATE,
                "channels": 1,
            }
        }
        model_path.with_suffix(".onnx.json").write_text(json.dumps(metadata))
        return model_path

    def test_reuses_persistent_piper_process(self) -> None:
        model_path = self._create_voice_files()
        script_path = self.tmp_path / "fake_piper.py"
        _write_fake_piper_script(script_path)

        tts = BufferedTTS(
            model_path=model_path,
            playback_cmd=None,
            use_subprocess=False,
            piper_cmd=[sys.executable, str(script_path)],
        )

        segments = []
        try:
            first = tts._generate_speech("hello persistent world", 1)
            self.assertIsNotNone(first)
            assert first is not None  # narrow type for mypy-like tools
            segments.append(first)
            self.assertGreater(len(first.raw), 0)

            process = tts._piper_process
            self.assertIsNotNone(process)
            assert process is not None
            first_pid = process.pid

            second = tts._generate_speech("second utterance", 2)
            self.assertIsNotNone(second)
            assert second is not None
            segments.append(second)
            self.assertGreater(len(second.raw), 0)

            process = tts._piper_process
            self.assertIsNotNone(process)
            assert process is not None
            self.assertEqual(first_pid, process.pid)

            queued = []
            try:
                queued.append(tts.speech_queue.get(timeout=1.0))
                queued.append(tts.speech_queue.get(timeout=1.0))
            except queue.Empty as exc:  # pragma: no cover - should not happen
                self.fail(f"Speech queue did not receive segments: {exc}")

            self.assertEqual([s.text for s in segments], [s.text for s in queued])
        finally:
            tts.stop()
            for segment in segments:
                if segment.path:
                    Path(segment.path).unlink(missing_ok=True)

        self.assertIsNone(tts._piper_process)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
