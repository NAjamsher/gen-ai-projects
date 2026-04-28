# MoodDetector — detects emotion/sentiment from any sentence
# Day 2 | Pillar: Python + APIs | Model: HuggingFace Sentiment Pipeline

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

# load the sentiment analysis pipeline once when server starts
print("Loading sentiment model... please wait")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Model loaded and ready!")

def detect_mood(text):
    result = sentiment_pipeline(text)[0]

    label = result["label"]
    score = result["score"]

    # make the label more human friendly
    if label == "POSITIVE":
        emoji = "😊"
        mood = "Positive"
    else:
        emoji = "😔"
        mood = "Negative"

    # convert score to percentage
    confidence = round(score * 100, 2)

    return {
        "mood": mood,
        "emoji": emoji,
        "confidence": confidence,
        "raw_label": label
    }

@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    user_text = data["text"]

    if not user_text.strip():
        return jsonify({"error": "Please enter some text"})

    result = detect_mood(user_text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)