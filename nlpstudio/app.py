# NLP Studio — Week 1 Multi-Tool AI Platform
# Day 7 | 6 Tools: Chat, Sentiment, Summarizer, NER, Topic, QA

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from transformers import pipeline
from dotenv import load_dotenv
import os
import json

app = Flask(__name__)
CORS(app)

load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ── CLIENT — exact same as TextSpark ──────────────────────────
client = InferenceClient(
    provider="novita",
    api_key=API_KEY,
)

# ── SENTIMENT MODEL — exact same as MoodDetector ──────────────
print("Loading sentiment model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("All systems ready!")

# ── CORE AI CALL — exact same as TextSpark ────────────────────
def call_llama(prompt, system="You are a helpful assistant.", temp=0.7, tokens=300):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=tokens,
        temperature=temp
    )
    return response.choices[0].message.content.strip()

# ── SERVE HTML ────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

# ── ROUTE 1 — CHAT — exact TextSpark pattern ──────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json()
    message = data.get("message", "")
    if not message.strip():
        return jsonify({"reply": "Please type something."})
    reply = call_llama(message)
    return jsonify({"reply": reply})

# ── ROUTE 2 — SENTIMENT — exact MoodDetector pattern ──────────
@app.route("/sentiment", methods=["POST"])
def sentiment():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Please enter some text."})
    result     = sentiment_pipeline(text)[0]
    label      = result["label"]
    confidence = round(result["score"] * 100, 2)
    emoji      = "😊" if label == "POSITIVE" else "😔"
    mood       = "Positive" if label == "POSITIVE" else "Negative"
    return jsonify({"mood": mood, "emoji": emoji, "confidence": confidence})

# ── ROUTE 3 — SUMMARIZER — exact Summarizer pattern ───────────
@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Please enter some text."})
    if len(text.split()) < 30:
        return jsonify({"error": "Text too short. Please enter at least 30 words."})

    prompt = f"""Summarize the following text in 2-3 clear sentences.
Only give the summary, nothing else.

Text:
{text}

Summary:"""

    summary = call_llama(prompt, "You are a summarization assistant. Only output clean concise summaries.", temp=0.3)
    return jsonify({
        "summary"       : summary,
        "original_words": len(text.split()),
        "summary_words" : len(summary.split()),
        "reduced_by"    : round((1 - len(summary.split()) / len(text.split())) * 100)
    })

# ── ROUTE 4 — ENTITIES — exact EntityExtractor pattern ────────
@app.route("/entities", methods=["POST"])
def entities():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Please enter some text."})
    if len(text.split()) < 10:
        return jsonify({"error": "Text too short. Please enter at least 10 words."})

    prompt = f"""Extract all named entities from the following text.
Return ONLY a valid JSON object with these exact keys:
- persons: list of people names
- organizations: list of company or institution names
- locations: list of countries cities or places
- dates: list of dates or time periods
- skills: list of technical skills tools or technologies

If none found return empty list. Return only JSON no explanation.

Text:
{text}

JSON:"""

    raw = call_llama(prompt, "You are an entity extraction assistant. Only return valid JSON. Never add markdown or explanation.", temp=0.1, tokens=400)

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        entities = json.loads(raw)
        entities["total"] = sum(len(v) for v in entities.values() if isinstance(v, list))
        return jsonify(entities)
    except json.JSONDecodeError:
        return jsonify({"error": "Could not parse entities. Please try again."})

# ── ROUTE 5 — CLASSIFIER — exact TopicClassifier pattern ──────
@app.route("/classify", methods=["POST"])
def classify():
    data       = request.get_json()
    text       = data.get("text", "")
    categories = data.get("categories", [
        "Technology", "Health", "Sports",
        "Politics", "Business", "Science",
        "Entertainment", "Education"
    ])
    if not text.strip():
        return jsonify({"error": "Please enter some text."})
    if len(text.split()) < 5:
        return jsonify({"error": "Text too short. Please enter at least 5 words."})

    category_list = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])

    prompt = f"""Classify the following text into exactly one of these categories:

{category_list}

Return ONLY a valid JSON object with these exact keys:
- category: the single best matching category name
- confidence: a percentage from 0 to 100
- reason: one sentence explaining why

Text:
{text}

JSON:"""

    raw = call_llama(prompt, "You are a text classification assistant. Only return valid JSON. Never add markdown.", temp=0.1, tokens=200)

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return jsonify(json.loads(raw))
    except json.JSONDecodeError:
        return jsonify({"error": "Could not classify. Please try again."})

# ── ROUTE 6 — QA — exact QABot pattern ────────────────────────
GREETINGS = [
    "hi", "hello", "hey", "thanks", "thank you", "thankyou",
    "ok", "okay", "great", "awesome", "cool", "bye", "goodbye",
    "nice", "perfect", "got it", "understood", "sure", "alright"
]

def is_greeting(message):
    msg = message.lower().strip()
    return any(msg == g or msg.startswith(g + " ") for g in GREETINGS)

@app.route("/qa", methods=["POST"])
def qa():
    data         = request.get_json()
    context      = data.get("context", "")
    question     = data.get("question", "")
    chat_history = data.get("chat_history", [])

    if not context.strip():
        return jsonify({"answer": "Please provide a context first."})
    if not question.strip():
        return jsonify({"answer": "Please ask a question."})
    if len(context.split()) < 20:
        return jsonify({"answer": "Context too short. Please paste at least 20 words."})

    if is_greeting(question):
        reply = call_llama(
            f'The user said: "{question}". Reply naturally in one short sentence.',
            "You are a friendly assistant. Keep replies short and warm.",
            temp=0.7, tokens=60
        )
        return jsonify({"answer": reply})

    history_text = ""
    if chat_history:
        history_text = "\n".join([
            f"Q: {item['question']}\nA: {item['answer']}"
            for item in chat_history[-3:]
        ])
        history_text = f"\nPrevious conversation:\n{history_text}\n"

    prompt = f"""Answer the question using ONLY the context below.
If not found say exactly: "I could not find the answer in the provided text."
Never use outside knowledge.

Context:
{context}
{history_text}
Question: {question}

Answer:"""

    answer = call_llama(prompt, "You are a precise QA assistant. Only answer from the provided context.", temp=0.1, tokens=300)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)