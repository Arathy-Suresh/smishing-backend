from flask import Flask, request, jsonify
from googletrans import Translator
from langdetect import detect

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from tensorflow.keras.models import load_model
import json

# Setup
app = Flask(__name__)
translator = Translator()
model = load_model("sms_phishing_model.h5")

# Load tokenizer
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)


# Prediction route
@app.route('/detect-smishing', methods=['POST'])
def detect_smishing():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        lang = detect(message)
    except Exception:
        lang = "unknown"

    translated_msg = message
    if lang != "en" and lang != "unknown":
        translated_msg = translator.translate(message, dest='en').text

    seq = tokenizer.texts_to_sequences([translated_msg])
    padded = pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded)[0][0]
    label = "Likely Smishing" if prediction > 0.5 else "Not Smishing"

    return jsonify({
        "language": lang,
        "translated_if_needed": translated_msg,
        "smishing": label,
        "confidence": round(float(prediction) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
