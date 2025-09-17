import sounddevice as sd 
import wave 
import subprocess
import requests
import json
import pyttsx3





RECORDED_WAV = "recorded.wav"
DURATION = 5
SAMPLE_RATE = 16000
WHISPER_EXE_PATH = r"C:\\Users\\Kushal\\Documents\\Release\\whisper-cli.exe"
WHISPER_MODEL = r"C:\\Users\\Kushal\\Documents\\Release\\models\\ggml-tiny.bin"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

def record_wav(path = RECORDED_WAV, duration = DURATION, s_r = SAMPLE_RATE):
    print(f"Recording audio for {duration} seconds at {s_r} Hz")
    audio = sd.rec(int(duration * s_r), samplerate=s_r, channels=1, dtype='int16')
    sd.wait()
    audio = audio.flatten().astype('int16')
    with wave.open(path, 'wb') as wf: 
        wf.setnchannels(1)
        wf.setsampwidth(2)
        
        wf.setframerate(s_r)
        wf.writeframes(audio.tobytes())
    print(f"Audio recorded and saved to {path}")
        

def transcribe_audio(wav_path):
    cmd = [WHISPER_EXE_PATH, "-m", WHISPER_MODEL, "-f", wav_path]
    print("Running the command : ","" .join(cmd))
    try:
        proces = subprocess.run(cmd,capture_output = True, text = True, timeout = 120 )
    except subprocess.TimeoutExpired:
        print ("Transcription process has timed out")
        return ""
    raw = proces.stdout.strip() or proces.stderr.strip()
    
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return "Nothing transcribed"
    
    for line in reversed(lines):
        if line.startswith('[') and '-->' in line:
            continue
        if ']' in line: 
            text = line.split(']')[-1].strip()
            if text:
                return text 
        else:
            return line
    
    return lines[-1]

def generate_response(prompt, timeout=30):
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}
    
    try:
        r = requests.post(OLLAMA_URL, json = payload, headers = headers , timeout = timeout)
        r.raise_for_status()
        data = r.json()
        
    except Exception as e:
        print(f"Ollama error: {e}")
        return "Sorry, couldn't send request to the model"
    if isinstance(data,dict):
        if "response" in data and isinstance(data["response"], str):
            return data["response"].strip()

        if "completion"  in data and isinstance(data["completion"], str):
            return data["completition"].strip()
        
        if "choices" in data and isinstance(data["choices"], list) and len(data["choices"])>0:
            c = data["choices"][0]
            if isinstance(c, dict) and "text" in c:
                return c["text"].strip()   
    return json.dumps(data)[:1000]


def speak_text(text):
    print(f"Speak: {text}")
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    

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



    
    