# Summarizer — condenses any long text into 2-3 lines
# Day 3 | Pillar: Python + APIs | Model: Llama via HuggingFace

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)
CORS(app)

API_KEY = "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"  # your key here

client = InferenceClient(
    provider="novita",
    api_key=API_KEY,
)

print("Summarizer ready!")

def summarize_text(text):
    word_count = len(text.split())

    if word_count < 30:
        return {
            "error": "Text too short. Please paste at least 30 words."
        }

    prompt = f"""Please summarize the following text in 2-3 clear sentences. 
Only give the summary, and give a suitable emoji  .

Text to summarize:
{text}

Summary:"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a summarization assistant. You only output clean, concise summaries. Never explain yourself. Never add extra commentary . end with a relatble joke."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.3
    )

    summary       = response.choices[0].message.content.strip()
    summary_words = len(summary.split())
    reduced_by    = round((1 - summary_words / word_count) * 100)

    return {
        "summary"       : summary,
        "original_words": word_count,
        "summary_words" : summary_words,
        "reduced_by"    : reduced_by
    }

@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data["text"]

    if not text.strip():
        return jsonify({"error": "Please enter some text"})

    result = summarize_text(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)