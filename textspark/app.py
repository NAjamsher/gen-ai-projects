# TextSpark Web — Flask backend + serves HTML
# Day 1 Extended | Pillar: Python + APIs + Flask

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)
CORS(app)

API_KEY = "your_huggingface_api_key_here"  # your key here

client = InferenceClient(
    provider="novita",
    api_key=API_KEY,
)

def ask_ai(prompt):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[

          {"role": "system", "content": "You are a funny assistant who answers every question with a joke first, then gives the real answer."},
          {"role": "user", "content": prompt}

          ],
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content

# this route serves your index.html when you visit /
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

# this route handles the AI chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data["message"]
    ai_reply = ask_ai(user_message)
    return jsonify({"reply": ai_reply})

if __name__ == "__main__":
    app.run(debug=True)