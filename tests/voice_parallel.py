import os
import time
import threading
import queue
import subprocess
import sounddevice as sd
import wave
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Tuple, Dict, Any
import psutil
import shlex

# ================== Configuration ==================
PROJECT_DIR = Path.cwd()
RECORDED_WAV = PROJECT_DIR / "recorded.wav"
SAMPLE_RATE = 16000
CHUNK_DURATION = 2  # Process in 2-second chunks for streaming

# Model paths
WHISPER_EXE = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = Path.home() / "whisper.cpp" / "models" / "ggml-tiny.bin"
LLAMA_CLI = Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli"
QWEN_MODEL = Path.home() / "Downloads" / "qwen2.5-0.5b-instruct-q3_k_m.gguf"

# ================== Streaming Audio Recorder ==================
class StreamingRecorder:
    """Records audio in chunks for parallel processing"""
    
    def __init__(self, chunk_duration=2, sample_rate=16000):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_queue = queue.Queue()
        self.recording = False
        self._thread = None
        
    def start(self):
        """Start recording in background"""
        self.recording = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        
    def _record_loop(self):
        """Continuously record chunks"""
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16') as stream:
            while self.recording:
                audio_chunk, _ = stream.read(chunk_samples)
                self.chunk_queue.put(audio_chunk.copy())
                
    def get_chunk(self, timeout=1):
        """Get next audio chunk"""
        try:
            return self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def stop(self):
        """Stop recording"""
        self.recording = False
        if self._thread:
            self._thread.join(timeout=1)

# ================== Parallel STT Processor ==================
class ParallelSTT:
    """Processes audio chunks in parallel using multiple STT instances"""
    
    def __init__(self, num_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.result_queue = queue.Queue()
        
    def process_chunk_whisper(self, audio_chunk: np.ndarray, chunk_id: int) -> Tuple[int, str]:
        """Process single audio chunk with Whisper"""
        # Save chunk to temp file
        temp_wav = f"/tmp/chunk_{chunk_id}.wav"
        
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_chunk.tobytes())
            
        # Run Whisper
        cmd = [
            str(WHISPER_EXE),
            "-m", str(WHISPER_MODEL),
            "-f", temp_wav,
            "--no-prints",
            "--output-txt",
            "-t", "2"  # Use 2 threads per instance
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Read output
            txt_path = Path(temp_wav + ".txt")
            if txt_path.exists():
                text = txt_path.read_text().strip()
                txt_path.unlink()  # Clean up
            else:
                text = ""
                
            Path(temp_wav).unlink()  # Clean up WAV
            return chunk_id, text
            
        except Exception as e:
            print(f"[STT] Error processing chunk {chunk_id}: {e}")
            return chunk_id, ""
            
    def submit_chunk(self, audio_chunk: np.ndarray, chunk_id: int) -> Future:
        """Submit chunk for async processing"""
        return self.executor.submit(self.process_chunk_whisper, audio_chunk, chunk_id)
        
    def shutdown(self):
        """Clean up executor"""
        self.executor.shutdown(wait=False)

# ================== Streaming LLM Processor ==================
class StreamingLLM:
    """Processes text incrementally and generates responses in chunks"""
    
    def __init__(self):
        self.context_buffer = []
        self.processing = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    def process_incremental(self, text_chunk: str, is_final: bool = False) -> Optional[str]:
        """Process incoming text and generate response if appropriate"""
        if not text_chunk.strip():
            return None
            
        self.context_buffer.append(text_chunk)
        
        # Process when we have enough context or on final chunk
        if is_final or self._should_respond():
            full_text = " ".join(self.context_buffer).strip()
            self.context_buffer = []  # Reset for next utterance
            
            # Generate response asynchronously
            future = self.executor.submit(self._generate_response, full_text)
            return future
            
        return None
        
    def _should_respond(self) -> bool:
        """Determine if we have enough context to generate response"""
        # Simple heuristic: respond after sentence end markers
        current_text = " ".join(self.context_buffer)
        return any(marker in current_text for marker in ["?", ".", "!"])
        
    def _generate_response(self, text: str) -> str:
        """Generate LLM response"""
        prompt = f"Answer briefly: {text}"
        
        cmd = [
            str(LLAMA_CLI),
            "-m", str(QWEN_MODEL),
            "-p", prompt,
            "-n", "32",
            "-t", "4",
            "--temp", "0.2",
            "--simple-io",
            "--ctx-size", "512"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                stdin=subprocess.DEVNULL
            )
            
            response = result.stdout.strip()
            # Remove echoed prompt if present
            if text in response:
                response = response.split(text)[-1].strip()
                
            return response
            
        except Exception as e:
            print(f"[LLM] Error: {e}")
            return "I couldn't process that."

# ================== Parallel TTS with Buffering ==================
class BufferedTTS:
    """Plays TTS output with pre-generation and buffering"""
    
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.playing = False
        self._playback_thread = None
        
    def start_playback(self):
        """Start background playback thread"""
        self.playing = True
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()
        
    def _playback_loop(self):
        """Continuously play queued speech"""
        while self.playing:
            try:
                audio_file = self.speech_queue.get(timeout=0.5)
                if audio_file:
                    # Play through PulseAudio (for Bluetooth)
                    subprocess.run(["paplay", audio_file], check=False)
                    Path(audio_file).unlink()  # Clean up
            except queue.Empty:
                continue
                
    def generate_and_queue(self, text: str, segment_id: int):
        """Generate TTS and add to playback queue"""
        future = self.executor.submit(self._generate_speech, text, segment_id)
        return future
        
    def _generate_speech(self, text: str, segment_id: int) -> str:
        """Generate speech file"""
        if not text.strip():
            return ""
            
        output_file = f"/tmp/tts_segment_{segment_id}.wav"
        
        # Using espeak for speed (can replace with Piper)
        try:
            subprocess.run(
                ["espeak", "-w", output_file, text],
                check=True,
                timeout=5
            )
            self.speech_queue.put(output_file)
            return output_file
        except Exception as e:
            print(f"[TTS] Error: {e}")
            return ""
            
    def stop(self):
        """Stop playback"""
        self.playing = False
        if self._playback_thread:
            self._playback_thread.join(timeout=1)
        self.executor.shutdown(wait=False)

# ================== Main Parallel Pipeline ==================
class ParallelVoiceAssistant:
    """Orchestrates parallel processing pipeline"""
    
    def __init__(self):
        self.recorder = StreamingRecorder(chunk_duration=2)
        self.stt = ParallelSTT(num_workers=2)
        self.llm = StreamingLLM()
        self.tts = BufferedTTS()
        
        # Pipeline queues
        self.stt_futures = queue.Queue()
        self.llm_futures = queue.Queue()
        
        # Stats
        self.stats = {
            "stt_chunks": 0,
            "llm_responses": 0,
            "tts_segments": 0,
            "total_latency": 0
        }
        
    def run(self, duration=10):
        """Run the parallel pipeline for specified duration"""
        print(f"[MAIN] Starting parallel pipeline for {duration}s...")
        start_time = time.time()
        
        # Start all components
        self.recorder.start()
        self.tts.start_playback()
        
        # Pipeline coordination threads
        stt_thread = threading.Thread(target=self._stt_pipeline, daemon=True)
        llm_thread = threading.Thread(target=self._llm_pipeline, daemon=True)
        
        stt_thread.start()
        llm_thread.start()
        
        # Run for specified duration
        time.sleep(duration)
        
        # Shutdown
        print("[MAIN] Shutting down...")
        self.recorder.stop()
        
        # Wait for pipeline to drain
        time.sleep(2)
        
        # Stop components
        self.stt.shutdown()
        self.tts.stop()
        
        # Report stats
        elapsed = time.time() - start_time
        self._print_stats(elapsed)
        
    def _stt_pipeline(self):
        """STT processing pipeline"""
        chunk_id = 0
        
        while self.recorder.recording:
            # Get audio chunk
            audio_chunk = self.recorder.get_chunk(timeout=1)
            if audio_chunk is None:
                continue
                
            # Submit for STT processing
            future = self.stt.submit_chunk(audio_chunk, chunk_id)
            self.stt_futures.put((chunk_id, future))
            
            self.stats["stt_chunks"] += 1
            chunk_id += 1
            
            # Process completed STT results
            self._process_stt_results()
            
    def _process_stt_results(self):
        """Process completed STT results"""
        completed = []
        
        # Check for completed futures
        temp_queue = queue.Queue()
        
        while not self.stt_futures.empty():
            try:
                chunk_id, future = self.stt_futures.get_nowait()
                if future.done():
                    _, text = future.result()
                    if text:
                        print(f"[STT] Chunk {chunk_id}: {text}")
                        
                        # Submit to LLM
                        llm_future = self.llm.process_incremental(text, is_final=False)
                        if llm_future:
                            self.llm_futures.put(llm_future)
                else:
                    temp_queue.put((chunk_id, future))
            except queue.Empty:
                break
                
        # Put incomplete futures back
        while not temp_queue.empty():
            self.stt_futures.put(temp_queue.get())
            
    def _llm_pipeline(self):
        """LLM and TTS processing pipeline"""
        segment_id = 0
        
        while self.recorder.recording or not self.llm_futures.empty():
            try:
                # Get LLM future
                llm_future = self.llm_futures.get(timeout=0.5)
                
                if isinstance(llm_future, Future):
                    # Wait for LLM response
                    response = llm_future.result(timeout=10)
                    
                    if response:
                        print(f"[LLM] Response: {response[:100]}...")
                        self.stats["llm_responses"] += 1
                        
                        # Split response for incremental TTS
                        sentences = self._split_sentences(response)
                        
                        for sentence in sentences:
                            self.tts.generate_and_queue(sentence, segment_id)
                            self.stats["tts_segments"] += 1
                            segment_id += 1
                            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[LLM Pipeline] Error: {e}")
                
    def _split_sentences(self, text: str) -> list:
        """Split text into sentences for incremental TTS"""
        # Simple sentence splitter
        sentences = []
        current = []
        
        for word in text.split():
            current.append(word)
            if any(word.endswith(p) for p in ['.', '!', '?', ',']):
                sentences.append(" ".join(current))
                current = []
                
        if current:
            sentences.append(" ".join(current))
            
        return sentences
        
    def _print_stats(self, elapsed: float):
        """Print performance statistics"""
        print("\n--- PERFORMANCE STATS ---")
        print(f"Total runtime: {elapsed:.2f}s")
        print(f"STT chunks processed: {self.stats['stt_chunks']}")
        print(f"LLM responses generated: {self.stats['llm_responses']}")
        print(f"TTS segments played: {self.stats['tts_segments']}")
        
        if self.stats['stt_chunks'] > 0:
            avg_chunk_time = elapsed / self.stats['stt_chunks']
            print(f"Avg time per STT chunk: {avg_chunk_time:.2f}s")
            
        # Memory stats
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Memory usage: {mem_mb:.1f} MB")
        print("------------------------\n")

# ================== Model Preloader ==================
class ModelPreloader:
    """Preload and warm up models for faster inference"""
    
    @staticmethod
    def warmup_whisper():
        """Warm up Whisper model"""
        print("[WARMUP] Loading Whisper model...")
        
        # Create dummy audio
        dummy_audio = np.zeros(16000, dtype=np.int16)
        temp_wav = "/tmp/warmup.wav"
        
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(dummy_audio.tobytes())
            
        # Run Whisper once
        cmd = [
            str(WHISPER_EXE),
            "-m", str(WHISPER_MODEL),
            "-f", temp_wav,
            "--no-prints",
            "-t", "1"
        ]
        
        subprocess.run(cmd, capture_output=True, timeout=30)
        Path(temp_wav).unlink()
        print("[WARMUP] Whisper ready")
        
    @staticmethod
    def warmup_llm():
        """Warm up LLM model"""
        print("[WARMUP] Loading LLM model...")
        
        cmd = [
            str(LLAMA_CLI),
            "-m", str(QWEN_MODEL),
            "-p", "test",
            "-n", "1",
            "--simple-io"
        ]
        
        subprocess.run(cmd, capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
        print("[WARMUP] LLM ready")

# ================== Usage Example ==================
def main():
    """Run the parallel voice assistant"""
    
    # Optional: Preload models
    print("[MAIN] Warming up models...")
    ModelPreloader.warmup_whisper()
    ModelPreloader.warmup_llm()
    
    # Run assistant
    assistant = ParallelVoiceAssistant()
    
    try:
        # Run for 30 seconds
        assistant.run(duration=30)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
        
if __name__ == "__main__":
    main()
