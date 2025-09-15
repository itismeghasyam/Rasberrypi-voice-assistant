import sounddevice as sd 
import wave 






RECORDED_WAV = "recorded.wav"
DURATION = 5
SAMPLE_RATE = 16000
WHISPER_EXE_PATH = r"C:\\Users\\Kushal\\Documents\\Release\\whisper-cli.exe"
WHISPER_MODEL = r"C:\\Users\\Kushal\\Documents\\Release\\models\\ggml-tiny.bin"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

def record_wav(path = RECORDED_WAV, duration = DURATION, s_r = SAMPLE_RATE):
    print(f"Recording audio for {duration} seconds at {s_r} Hz")
    audio = sd.rec(int(duration * s_r), samplerate=s_r, channels=1)
    sd.wait()
    audio = audio.flatten().astype('int16')
    with wave.open(path, 'wb') as wf: 
        wf.setnchannels(1)
        wf.setsamplewidth(2)
        
        wf.setframerate(s_r)
        wf.writeframes(audio.tobytes())
    print(f"Audio recorded and saved to {path}")
        

def transcribe_audio(wav_path):
    
    print("hello")

def generate_response(prompt, timeout=30):
    print("hello")
    
def speak_text():
    print("hello")

def main():
    record_wav()
    transcribe = transcribe_audio(RECORDED_WAV)
    
    if not transcribe:
        print("No transcription found.")
        return
    
    response = generate_response(transcribe)
    print(" The Model replied", response)
    speak_text(response)

if __name__ == "__main__":
    main()     



    
    