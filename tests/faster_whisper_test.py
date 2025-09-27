from openai import OpenAI

client = OpenAI(api_key="cant-be-empty", base_url="http://localhost:8000/v1/")

audio_file = open("~/Rasberrypi-voice-assistant/recorded.wav", "rb")
transcript = client.audio.transcriptions.create(
    model="Systran/faster-distil-whisper-large-v3", file=audio_file
)
print(transcript.text)