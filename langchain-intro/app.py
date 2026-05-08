# LangChain Intro — Summarizer with LangChain + proven InferenceClient
# Day 8 | Pillar: LangChain

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)
CORS(app)

load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

print("Setting up LangChain...")

# ── PROVEN CLIENT — same as every working project ──────────────
client = InferenceClient(
    provider="novita",
    api_key=API_KEY,
)

# ── WRAP CLIENT AS LANGCHAIN RUNNABLE ─────────────────────────
# RunnableLambda turns any function into a LangChain component
# This is the cleanest way to use InferenceClient with LangChain
def call_llama(prompt_value):
    prompt_text = prompt_value.text if hasattr(prompt_value, 'text') else str(prompt_value)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=200,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

llm = RunnableLambda(call_llama)

# ── PROMPT TEMPLATE ───────────────────────────────────────────
summarize_template = PromptTemplate(
    input_variables=["text"],
    template="""Summarize the following text in 2-3 clear sentences.
Only give the summary, nothing else.

Text:
{text}

Summary:"""
)

# ── CHAIN — prompt | llm | output parser ─────────────────────
summarize_chain = summarize_template | llm | StrOutputParser()

print("LangChain ready!")

# ── FUNCTION ──────────────────────────────────────────────────
def summarize_text(text):
    if len(text.split()) < 30:
        return {"error": "Text too short. Please enter at least 30 words."}

    summary        = summarize_chain.invoke({"text": text})
    summary        = summary.strip()
    original_words = len(text.split())
    summary_words  = len(summary.split())
    reduced_by     = round((1 - summary_words / original_words) * 100)

    return {
        "summary"       : summary,
        "original_words": original_words,
        "summary_words" : summary_words,
        "reduced_by"    : reduced_by
    }

# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Please enter some text."})
    try:
        result = summarize_text(text)
        return jsonify(result)
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)