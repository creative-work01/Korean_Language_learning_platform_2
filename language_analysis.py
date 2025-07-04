from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load pre-trained KoBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
model = BertForMaskedLM.from_pretrained("monologg/kobert")

def analyze_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    # Print logits (can be used for further analysis or feedback)
    return outputs.logits

# Example text from Whisper transcription
text = "안녕하세요. 저는 한국어를 배우고 있습니다."
analysis = analyze_text(text)
print("Analysis:", analysis)
