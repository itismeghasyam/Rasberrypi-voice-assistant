from openai import OpenAI
import os 

client = OpenAI(api_key="cant-be-empty", base_url="http://localhost:8000/v1/")

audio_file = open(os.path.expanduser("~/Rasberrypi-voice-assistant/recorded.wav"), "rb")
transcript = client.audio.transcriptions.create(
    model="Systran/faster-distil-whisper-tiny-v3", file=audio_file
)
print(transcript.text)