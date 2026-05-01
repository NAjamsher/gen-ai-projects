# EntityExtractor — extracts named entities from any text
# Day 4 | Pillar: Python + APIs | Model: Llama 3.1 via HuggingFace

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

print("EntityExtractor ready!")

def extract_entities(text):
    word_count = len(text.split())

    if word_count < 10:
        return {"error": "Text too short. Please enter at least 10 words."}

    prompt = f"""Extract all named entities from the following text.
Return ONLY a valid JSON object with these exact keys:
- persons: list of people's names
- organizations: list of company, institution, or organization names
- locations: list of countries, cities, or places
- dates: list of dates or time periods
- skills: list of technical skills, tools, or technologies mentioned
- awards: list of any awards or recognitions mentioned


If none found for a category return an empty list.
Do not add any explanation. Return only the JSON.

Text:
{text}

JSON:"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are an entity extraction assistant. You only return valid JSON. Never add explanation or markdown formatting. Never wrap in code blocks."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()

    # clean markdown code blocks if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        entities = json.loads(raw)
        total = sum(len(v) for v in entities.values() if isinstance(v, list))
        entities["total"] = total
        entities["word_count"] = word_count
        return entities
    except json.JSONDecodeError:
        return {"error": "Could not parse entities. Please try again."}

@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/extract", methods=["POST"])
def extract():
    data = request.get_json()
    text = data["text"]

    if not text.strip():
        return jsonify({"error": "Please enter some text"})

    result = extract_entities(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)