from flask import Flask, request, render_template
import whisper
from transformers import BertTokenizer, BertForMaskedLM

app = Flask(__name__)

# Load Whisper model for speech recognition
whisper_model = whisper.load_model("base")

# Load KoBERT model for language analysis
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
bert_model = BertForMaskedLM.from_pretrained("monologg/kobert")

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result['text']

def analyze_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = bert_model(**inputs)
    return outputs.logits

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    # Get the uploaded audio file
    audio = request.files['audio']
    audio.save("uploaded_audio.wav")
    
    # Transcribe and analyze the audio
    transcription = transcribe_audio("uploaded_audio.wav")
    analysis = analyze_text(transcription)
    
    return render_template('feedback.html', transcription=transcription, analysis=analysis)

if __name__ == '__main__':
    app.run(debug=True)
