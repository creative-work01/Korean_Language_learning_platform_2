import whisper

# Load the Whisper model (you can choose 'base' or 'small' for faster processing)
model = whisper.load_model("base")

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result['text']

# Test with a sample audio file
audio_path = "1_1018.wav"  # Replace with the path to your audio file
transcription = transcribe_audio(audio_path)
print("Transcription:", transcription)
