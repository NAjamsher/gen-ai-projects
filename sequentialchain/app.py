# Sequential Chain — News Analyzer
# Day 10 | Pillar: LangChain | Sequential Chains

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)
CORS(app)

load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
print(f"API KEY LOADED: {API_KEY}")

print("Setting up Sequential Chain...")

# ── PROVEN CLIENT ──────────────────────────────────────────────
client = InferenceClient(
    provider="novita",
    api_key=API_KEY,
)

# ── CORE LLM FUNCTION ─────────────────────────────────────────
def call_llama(prompt_value, temp=0.3, tokens=300):
    prompt_text = prompt_value.text if hasattr(prompt_value, 'text') else str(prompt_value)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=tokens,
        temperature=temp
    )
    return response.choices[0].message.content.strip()

parser = StrOutputParser()

# ── STEP 1 — SUMMARIZE THE ARTICLE ────────────────────────────
summarize_template = PromptTemplate(
    input_variables=["article"],
    template="""Summarize the following news article in 2-3 clear sentences.
Only give the summary, nothing else.

Article:
{article}

Summary:"""
)

summarize_chain = summarize_template | RunnableLambda(call_llama) | parser

# ── STEP 2 — ANALYSE SENTIMENT OF SUMMARY ─────────────────────
sentiment_template = PromptTemplate(
    input_variables=["summary"],
    template="""Analyse the sentiment of the following text.
Return ONLY one word: Positive, Negative, or Neutral.

Text:
{summary}

Sentiment:"""
)

sentiment_chain = sentiment_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.1, tokens=10)
) | parser

# ── STEP 3 — SUGGEST A TWEET ──────────────────────────────────
tweet_template = PromptTemplate(
    input_variables=["summary", "sentiment"],
    template="""Based on this news summary and its sentiment, write an engaging tweet.

Summary: {summary}
Sentiment: {sentiment}

Write a tweet under 280 characters with 2-3 relevant hashtags.
Only write the tweet, nothing else.

Tweet:"""
)

tweet_chain = tweet_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.7, tokens=150)
) | parser

# ── SEQUENTIAL PIPELINE ───────────────────────────────────────
def analyze_news(article):
    if len(article.split()) < 30:
        return {"error": "Article too short. Please paste at least 30 words."}

    # step 1 — summarize
    summary = summarize_chain.invoke({"article": article})

    # step 2 — analyse sentiment of the summary
    sentiment = sentiment_chain.invoke({"summary": summary})
    sentiment = sentiment.strip().capitalize()

    # step 3 — suggest tweet using both summary and sentiment
    tweet = tweet_chain.invoke({
        "summary"  : summary,
        "sentiment": sentiment
    })

    return {
        "summary"  : summary,
        "sentiment": sentiment,
        "tweet"    : tweet,
        "word_count": len(article.split())
    }

# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data    = request.get_json()
    article = data.get("article", "")
    if not article.strip():
        return jsonify({"error": "Please paste a news article."})
    try:
        result = analyze_news(article)
        return jsonify(result)
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)