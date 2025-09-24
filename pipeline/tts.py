from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional,Any,Iterable,Callable, Dict, List
from concurrent.futures import Future, ThreadPoolExecutor

import os, subprocess, threading, queue, json, selectors,wave,tempfile
import numpy as np, sounddevice as sd 
from time import time

from config import PIPER_MODEL_PATH



@dataclass
class SpeechSegment:
    path: str
    raw: bytes
    sample_rate: int
    channels: int = 1
    sampwidth: int = 2
    text: str = ""
    
    
@dataclass
class PiperVoiceInfo:
    sample_rate: int = 8000
    speaker_id: Optional[int] = None
    channels: int = 1
    metadata_path: Optional[Path] = None



class BufferedTTS:
    """Generate speech with Piper asynchronously and stream playback via a CLI player."""


    def __init__(
        self,
        model_path: Path = PIPER_MODEL_PATH,
        playback_cmd: Optional[Iterable[str]] = None,
        piper_cmd: Optional[Iterable[str]] = None,
        output_device: Optional[Any] = None,
        use_subprocess: bool = False,
        on_playback_start: Optional[Callable[[str, float], None]] = None,
        on_playback_error: Optional[Callable[[], None]] = None,
        timeout: int = 6,

    ) -> None:
        self.model_path = Path(model_path)
        self.timeout = int(timeout)
        self._voice_info = self._load_voice_info()
        self.speech_queue: "queue.Queue[SpeechSegment]" = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.playing = False
        self._playback_thread: Optional[threading.Thread] = None
        if playback_cmd:
            self.playback_cmd = list(playback_cmd)
        else:
            self.playback_cmd = ["aplay", "{file}"]
        self.use_subprocess = bool(use_subprocess)
        if output_device is None:
            self.output_device = None
        elif isinstance(output_device, int):
            self.output_device = output_device
        elif isinstance(output_device, str):
            try:
                self.output_device = int(output_device)
            except ValueError:
                self.output_device = output_device
        else:
            self.output_device = output_device
        self.on_playback_start = on_playback_start
        self.on_playback_error = on_playback_error

        self._playback_env = os.environ.copy()
        if isinstance(self.output_device, str):
            # Hint to PulseAudio-based players which sink to target.
            self._playback_env.setdefault("PULSE_SINK", self.output_device)

        self._piper_base_cmd = list(piper_cmd) if piper_cmd else ["piper"]
        self._piper_process: Optional[subprocess.Popen[bytes]] = None
        self._piper_stderr_thread: Optional[threading.Thread] = None
        self._piper_stderr_stop: Optional[threading.Event] = None
        self._piper_lock = threading.Lock()
        self._piper_read_interval = 0.05
        self._piper_idle_timeout = 0.35

        if self.model_path.exists():
            with self._piper_lock:
                self._start_piper_process()


    def _load_voice_info(self) -> PiperVoiceInfo:
        candidates = [
            self.model_path.with_suffix(self.model_path.suffix + ".json"),
            self.model_path.with_suffix(".json"),
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                metadata = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue

            audio = metadata.get("audio", {}) if isinstance(metadata, dict) else {}
            sample_rate = int(audio.get("sample_rate") or metadata.get("sample_rate", 22050))
            channels = int(audio.get("channels", 1) or 1)

            speaker_id: Optional[int] = None
            if "speaker_id" in metadata:
                try:
                    speaker_id = int(metadata.get("speaker_id"))
                except Exception:
                    speaker_id = None
            elif isinstance(metadata.get("speakers"), dict):
                speakers_dict: Dict[str, Any] = metadata.get("speakers", {})
                if speakers_dict:
                    first_key = next(iter(speakers_dict))
                    first_val = speakers_dict[first_key]
                    if isinstance(first_val, dict) and "id" in first_val:
                        try:
                            speaker_id = int(first_val["id"])
                        except Exception:
                            speaker_id = None
                    else:
                        try:
                            speaker_id = int(first_key)
                        except Exception:
                            speaker_id = None

            return PiperVoiceInfo(
                sample_rate=sample_rate or 22050,
                speaker_id=speaker_id,
                channels=channels or 1,
                metadata_path=candidate,
            )

        return PiperVoiceInfo()


    def start_playback(self) -> None:
        if self.playing:
            return
        self.playing = True
        self._playback_thread = threading.Thread(target=self._playback_loop, name="BufferedTTS", daemon=True)
        self._playback_thread.start()

    def _playback_loop(self) -> None:
        while self.playing:
            try:

                segment = self.speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not segment:
                continue

            played = False
            if not self.use_subprocess:
                played = self._play_via_sounddevice(segment)

            if not played and self.playback_cmd:
                played = self._play_via_subprocess(segment)

            if not played:
                print(f"[TTS] Playback failed for {segment.path}")
                self._notify_playback_error()

            try:
                if segment.path:
                    Path(segment.path).unlink(missing_ok=True)

            except OSError:
                pass

    def _notify_playback_error(self) -> None:
        if self.on_playback_error is None:
            return
        try:
            self.on_playback_error()
        except Exception as exc:
            print(f"[TTS] Playback error callback failed: {exc}")


    def _build_piper_command(self) -> List[str]:
        cmd = list(self._piper_base_cmd)
        cmd += ["-m", str(self.model_path), "--output-raw"]
        info = self._voice_info
        has_speaker_flag = any(part in {"-s", "--speaker"} for part in cmd)
        if info.speaker_id is not None and not has_speaker_flag:
            cmd += ["--speaker", str(info.speaker_id)]
        return cmd

    def _start_piper_process(self) -> None:
        self._stop_piper_process()
        cmd = self._build_piper_command()
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError:
            print(f"[TTS] Piper executable not found: {cmd[0]}")
            self._piper_process = None
            return
        except Exception as exc:
            print(f"[TTS] Failed to start Piper process: {exc}")
            self._piper_process = None
            return

        self._piper_process = process
        stop_event = threading.Event()
        self._piper_stderr_stop = stop_event
        if process.stderr is not None:
            self._piper_stderr_thread = threading.Thread(
                target=self._drain_piper_stderr,
                name="PiperStderr",
                args=(process.stderr, stop_event),
                daemon=True,
            )
            self._piper_stderr_thread.start()

    def _drain_piper_stderr(self, stream: Any, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                chunk = stream.readline()
            except Exception:
                break

            if not chunk:
                break

            try:
                message = chunk.decode("utf-8", errors="replace").strip()
            except Exception:
                message = ""

            if message:
                print(f"[TTS][Piper] {message}")

    def _stop_piper_process(self) -> None:
        process = self._piper_process
        self._piper_process = None

        if process is None:
            return

        stop_event = self._piper_stderr_stop
        if stop_event is not None:
            stop_event.set()

        try:
            if process.stdin:
                process.stdin.close()
        except Exception:
            pass

        try:
            if process.stdout:
                process.stdout.close()
        except Exception:
            pass

        try:
            if process.stderr:
                process.stderr.close()
        except Exception:
            pass

        try:
            process.terminate()
            process.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except Exception:
                pass
            try:
                process.wait(timeout=1.0)
            except Exception:
                pass
        except Exception:
            pass

        if self._piper_stderr_thread and self._piper_stderr_thread.is_alive():
            self._piper_stderr_thread.join(timeout=0.5)
        self._piper_stderr_thread = None
        self._piper_stderr_stop = None

    def _ensure_piper_process(self) -> Optional[subprocess.Popen[bytes]]:
        process = self._piper_process
        if process is None or process.poll() is not None:
            self._start_piper_process()
            process = self._piper_process
        return process

    def _write_to_piper(self, process: subprocess.Popen[bytes], utterance: str) -> None:
        if process.stdin is None:
            raise RuntimeError("Piper stdin is unavailable")

        payload = (utterance + "\n").encode("utf-8")
        process.stdin.write(payload)
        process.stdin.flush()

    def _read_from_piper(self, process: subprocess.Popen[bytes]) -> bytes:
        stdout = process.stdout
        if stdout is None:
            raise RuntimeError("Piper stdout is unavailable")

        buffer = bytearray()
        idle_started: Optional[float] = None
        deadline = time.time() + max(1.0, float(self.timeout))

        selector = selectors.DefaultSelector()
        try:
            selector.register(stdout, selectors.EVENT_READ)
        except Exception as exc:
            selector.close()
            raise RuntimeError(f"Failed to monitor Piper stdout: {exc}")

        try:
            while True:
                remaining = deadline - time.time()
                if remaining <= 0 and not buffer:
                    raise TimeoutError("Timed out waiting for Piper audio")

                wait_time = self._piper_read_interval
                if remaining > 0:
                    wait_time = min(wait_time, remaining)

                events = selector.select(wait_time)
                if events:
                    try:
                        if hasattr(stdout, "read1"):
                            chunk = stdout.read1(4096)
                        else:
                            chunk = stdout.read(4096)
                    except Exception as exc:
                        raise RuntimeError(f"Failed reading Piper audio: {exc}")

                    if not chunk:
                        break

                    buffer.extend(chunk)
                    idle_started = None
                    continue

                if process.poll() is not None:
                    break

                if buffer:
                    if idle_started is None:
                        idle_started = time.time()
                    elif time.time() - idle_started >= self._piper_idle_timeout:
                        break
                elif remaining <= 0:
                    raise TimeoutError("Timed out waiting for Piper audio")

        finally:
            selector.close()

        return bytes(buffer)

    def _play_via_sounddevice(self, segment: SpeechSegment) -> bool:
        if sd is None:
            return False
        try:
            if segment.raw:
                audio = np.frombuffer(segment.raw, dtype=np.int16)
                max_val = float(np.iinfo(np.int16).max)
                audio = audio.astype(np.float32) / max_val
                if segment.channels > 1:
                    audio = audio.reshape(-1, segment.channels)
                sample_rate = segment.sample_rate
            else:
                with wave.open(segment.path, "rb") as wf:
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    frames = wf.getnframes()
                    audio_bytes = wf.readframes(frames)

                dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
                dtype = dtype_map.get(sampwidth)
                if dtype is None:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")

                audio = np.frombuffer(audio_bytes, dtype=dtype)
                if dtype == np.uint8:
                    audio = audio.astype(np.float32)
                    audio = (audio - 128.0) / 128.0
                else:
                    max_val = float(np.iinfo(dtype).max)
                    if not max_val:
                        raise ValueError("Invalid max value for dtype")
                    audio = audio.astype(np.float32) / max_val

                if channels > 1:
                    audio = audio.reshape(-1, channels)

            sd.stop()
            sd.play(audio, sample_rate, device=self.output_device, blocking=False)
            if self.on_playback_start:
                self.on_playback_start(segment.path, time.time())
            sd.wait()
            return True
        except Exception as exc:
            print(f"[TTS] Direct playback failed for {segment.path}: {exc}")
            return False

    def _play_via_subprocess(self, segment: SpeechSegment) -> bool:
        try:
            if self.on_playback_start:
                self.on_playback_start(segment.path, time.time())
            cmd = list(self.playback_cmd)
            if isinstance(self.output_device, str) and cmd:
                if (
                    cmd[0] == "paplay"
                    and not any(str(arg).startswith("--device=") for arg in cmd[1:])
                ):
                    cmd = cmd + [f"--device={self.output_device}"]
                elif cmd[0] == "aplay" and "-D" not in cmd:
                    cmd = cmd[:1] + ["-D", self.output_device] + cmd[1:]
            if any("{file}" in str(part) for part in cmd):
                resolved = [str(part).replace("{file}", segment.path) for part in cmd]
                subprocess.run(resolved, check=True, env=self._playback_env)
            elif cmd and cmd[-1] == "-" and segment.raw is not None:
                subprocess.run(
                    cmd,
                    input=segment.raw,
                    check=True,
                    env=self._playback_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
            else:
                subprocess.run(cmd + [segment.path],
                               check=True,
                               env=self._playback_env,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL
                               )
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[TTS] Subprocess playback failed (exit {exc.returncode}) for {segment.path}: {exc}")
            return False
        except Exception as exc:
            print(f"[TTS] Subprocess playback failed for {segment.path}: {exc}")
            return False


    def generate_and_queue(self, text: str, segment_id: int) -> Optional[Future]:
        clean_text = " ".join((text or "").split())
        if not clean_text:
            return None
        return self.executor.submit(self._generate_speech, clean_text, segment_id)


    def _generate_speech(self, text: str, segment_id: int) -> Optional[SpeechSegment]:

        if not self.model_path.exists():
            print(f"[TTS] Piper model not found: {self.model_path}")
            return None

        utterance = " ".join((text or "").split())
        if not utterance:
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        keep_file = False
        audio_bytes: Optional[bytes] = None
        restart_needed = False
        try:
            with self._piper_lock:
                process = self._ensure_piper_process()
                if process is None or process.stdin is None or process.stdout is None:
                    raise RuntimeError("Piper process is unavailable")

                self._write_to_piper(process, utterance)
                audio_bytes = self._read_from_piper(process)

        except TimeoutError:
            restart_needed = True
            print("[TTS] Piper audio stream timed out")
        except BrokenPipeError:
            restart_needed = True
            print("[TTS] Piper audio stream closed unexpectedly")
        except Exception as exc:
            restart_needed = True
            print(f"[TTS] Piper streaming failed: {exc}")
        finally:
            if restart_needed and self.model_path.exists():
                with self._piper_lock:
                    self._start_piper_process()

        if not audio_bytes:
            print("[TTS] Piper returned no audio data")
            if not keep_file and tmp_path.exists():
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            return None

        if self._looks_like_text(audio_bytes):
            preview = audio_bytes[:120].decode("utf-8", errors="replace")
            print(
                "[TTS] Piper returned textual output instead of audio; "
                f"got: {preview!r}"
            )
            if not keep_file and tmp_path.exists():
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            return None

        info = self._voice_info
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(info.channels or 1)
            wf.setsampwidth(2)
            wf.setframerate(info.sample_rate or 22050)
            wf.writeframes(audio_bytes)
        keep_file = True

        sample_rate = info.sample_rate or 22050
        segment = SpeechSegment(
            path=str(tmp_path),
            raw=audio_bytes,
            sample_rate=sample_rate,
            channels=info.channels or 1,
            text=utterance,
        )
        self.speech_queue.put(segment)
        return segment

        return None

    @staticmethod
    def _looks_like_text(payload: bytes) -> bool:
        """Heuristic check to detect when Piper prints text instead of PCM."""

        if not payload:
            return False

        sample = payload[:64]
        printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in sample)
        # Random PCM rarely decodes into predominantly printable ASCII. Treat a
        # mostly printable prefix as an indication that Piper emitted text/logs.
        return printable >= max(10, len(sample) * 0.6)

    def stop(self) -> None:
        self.playing = False
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None
        with self._piper_lock:
            self._stop_piper_process()
        self.executor.shutdown(wait=False)

