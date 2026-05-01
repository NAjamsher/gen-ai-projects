# QABot — answers questions from a given text context
# Day 6 | Pillar: Python + APIs | Model: Llama 3.1 via HuggingFace

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

client = InferenceClient(
    provider="novita",
    api_key=API_KEY,
)

print("QABot ready!")

# list of casual messages and greetings
GREETINGS = [
    "hi", "hello", "hey", "thanks", "thank you", "thankyou",
    "ok", "okay", "great", "awesome", "cool", "bye", "goodbye",
    "good morning", "good evening", "good night", "welcome",
    "nice", "perfect", "got it", "understood", "sure", "alright",
    "ok thanks", "ok thank you", "sounds good", "makes sense",
    "interesting", "wow", "amazing", "fantastic", "wonderful","fine"
]

def is_greeting(message):
    message_lower = message.lower().strip()
    return any(
        message_lower == g or message_lower.startswith(g + " ")
        for g in GREETINGS
    )

def handle_greeting(question):
    prompt = f"""The user said: "{question}"
This is a casual message or greeting — not a question about a document.
Reply naturally and warmly in one short sentence.
Do not mention the document or context."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a friendly and helpful assistant. Keep replies short and warm."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=60,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def answer_question(context, question, chat_history):

    # validation
    if len(context.split()) < 20:
        return {"error": "Context too short. Please paste at least 20 words."}

    if not question.strip():
        return {"error": "Please ask a question."}

    # greeting detection — handle before touching context
    if is_greeting(question):
        reply = handle_greeting(question)
        return {
            "answer"  : reply,
            "question": question
        }

    # build conversation history for multi turn
    history_text = ""
    if chat_history:
        history_text = "\n".join([
            f"Q: {item['question']}\nA: {item['answer']}"
            for item in chat_history[-3:]
        ])
        history_text = f"\nPrevious conversation:\n{history_text}\n"

    # main QA prompt — grounded to context only
    prompt = f"""You are a question answering assistant.
Answer the question using ONLY the information provided in the context below.
If the answer is not found in the context say exactly:
"I could not find the answer in the provided text."
Never use your general knowledge. Only use the context.

Context:
{context}
{history_text}
Current Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a precise question answering assistant. You only answer from the provided context. You never use outside knowledge. If the answer is not in the context you say so clearly."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=300,
        temperature=0.1
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer"  : answer,
        "question": question
    }

@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data         = request.get_json()
    context      = data.get("context", "")
    question     = data.get("question", "")
    chat_history = data.get("chat_history", [])

    if not context.strip():
        return jsonify({"error": "Please provide a context first."})

    result = answer_question(context, question, chat_history)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)