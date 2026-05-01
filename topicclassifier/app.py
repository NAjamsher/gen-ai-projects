# TopicClassifier — classifies any text into a topic category
# Day 5 | Pillar: Python + APIs | Zero-Shot Classification

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json

app = Flask(__name__)
CORS(app)

load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

client = InferenceClient(
    provider="novita",
    api_key=API_KEY,
)

print("TopicClassifier ready!")

# these are the default categories
# the power of zero-shot — you can change these anytime
DEFAULT_CATEGORIES = [
    "Technology",
    "Health & Medicine",
    "Sports",
    "Politics",
    "Business & Finance",
    "Science",
    "Entertainment",
    "Education",
    "Environment",
    "Crime & Law"
]

def classify_topic(text, categories):
    word_count = len(text.split())

    if word_count < 5:
        return {"error": "Text too short. Please enter at least 5 words."}

    # format categories as a numbered list for the prompt
    category_list = "\n".join(
        [f"{i+1}. {cat}" for i, cat in enumerate(categories)]
    )

    prompt = f"""Classify the following text into exactly one of these categories:

{category_list}

Return ONLY a valid JSON object with these exact keys:
- category: the single best matching category name
- confidence: a percentage from 0 to 100 showing how confident you are
- reason: one sentence explaining why you chose this category

Text to classify:
{text}

JSON:"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a text classification assistant. You only return valid JSON. Never add explanation or markdown. Always pick exactly one category from the provided list."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()

    # clean markdown if model adds it
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
        result["word_count"] = word_count
        result["all_categories"] = categories
        return result
    except json.JSONDecodeError:
        return {"error": "Could not parse result. Please try again."}

@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/classify", methods=["POST"])
def classify():
    data        = request.get_json()
    text        = data.get("text", "")
    categories  = data.get("categories", DEFAULT_CATEGORIES)

    if not text.strip():
        return jsonify({"error": "Please enter some text"})

    result = classify_topic(text, categories)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)