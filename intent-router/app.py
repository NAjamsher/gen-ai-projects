# Intent Router — detects what the user wants and routes to the right chain
# Day 14 | Pillar: LangChain | Concept: Intent Detection + Routing

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

# ── CONFIGURE ─────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
print(f"API KEY LOADED: {API_KEY}")

# ── SETUP ─────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

client = InferenceClient(
    provider="novita",
    api_key=API_KEY,
)

# ── CORE LLM FUNCTION ─────────────────────────────────────────
def call_llama(prompt_value, temp=0.1, tokens=500):
    prompt_text = prompt_value.text if hasattr(prompt_value, "text") else str(prompt_value)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=tokens,
        temperature=temp
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        return content[0]["text"]
    return str(content).strip()

# ── LANGCHAIN COMPONENTS ───────────────────────────────────────
llm    = RunnableLambda(call_llama)
parser = StrOutputParser()

# ── INTENT DETECTION CHAIN ────────────────────────────────────
intent_template = PromptTemplate(
    input_variables=["user_input"],
    template="""You are an intent classifier. Read the user message and classify it into exactly one of these intents:

SUMMARIZE    — user wants to summarize text
SENTIMENT    — user wants to detect emotion or mood
QA           — user wants to ask a question about something
GENERATE     — user wants to generate content like blog, tweet, tagline
CHAT         — user just wants to have a general conversation

Rules:
- Reply with ONLY the intent word. Nothing else.
- No explanation. No punctuation. Just one word.

User message: {user_input}

Intent:"""
)

intent_chain = intent_template | llm | parser

# ── SUMMARIZE CHAIN ───────────────────────────────────────────
summarize_template = PromptTemplate(
    input_variables=["user_input"],
    template="""Summarize the following text in 2-3 clear sentences.
Only give the summary, nothing else.

Text: {user_input}

Summary:"""
)
summarize_chain = summarize_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.3, tokens=200)
) | parser

# ── SENTIMENT CHAIN ───────────────────────────────────────────
sentiment_template = PromptTemplate(
    input_variables=["user_input"],
    template="""Analyze the sentiment of the following text.
Return a JSON object with:
- sentiment: Positive, Negative, or Neutral
- confidence: a percentage like 92%
- reason: one sentence explaining why

Only return the JSON. Nothing else.

Text: {user_input}

JSON:"""
)
sentiment_chain = sentiment_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.1, tokens=150)
) | parser

# ── QA CHAIN ──────────────────────────────────────────────────
qa_template = PromptTemplate(
    input_variables=["user_input"],
    template="""Answer the following question clearly and concisely.
Give a helpful, accurate answer in 2-4 sentences.

Question: {user_input}

Answer:"""
)
qa_chain = qa_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.3, tokens=300)
) | parser

# ── GENERATE CHAIN ────────────────────────────────────────────
generate_template = PromptTemplate(
    input_variables=["user_input"],
    template="""The user wants to generate content about: {user_input}

Write the following three things:
1. A blog introduction (3-4 sentences)
2. A tweet (under 280 characters with hashtags)
3. A tagline (maximum 10 words)

Format your response exactly like this:
BLOG: [blog intro here]
TWEET: [tweet here]
TAGLINE: [tagline here]"""
)
generate_chain = generate_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.7, tokens=400)
) | parser

# ── CHAT CHAIN ────────────────────────────────────────────────
chat_template = PromptTemplate(
    input_variables=["user_input"],
    template="""You are a friendly and helpful AI assistant.
Reply naturally and conversationally to the following message.

Message: {user_input}

Reply:"""
)
chat_chain = chat_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.7, tokens=300)
) | parser

# ── ROUTER FUNCTION ───────────────────────────────────────────
def route(user_input):
    raw_intent = intent_chain.invoke({"user_input": user_input})
    intent     = raw_intent.strip().upper()

    # clean up in case model adds extra words
    for keyword in ["SUMMARIZE", "SENTIMENT", "QA", "GENERATE", "CHAT"]:
        if keyword in intent:
            intent = keyword
            break
    else:
        intent = "CHAT"

    print(f"DETECTED INTENT: {intent}")

    if intent == "SUMMARIZE":
        result = summarize_chain.invoke({"user_input": user_input})
        return {"intent": "SUMMARIZE", "result": result}

    elif intent == "SENTIMENT":
        raw = sentiment_chain.invoke({"user_input": user_input})
        # clean JSON if model wraps in markdown
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        try:
            import json
            parsed = json.loads(raw)
            return {"intent": "SENTIMENT", "result": parsed}
        except Exception:
            return {"intent": "SENTIMENT", "result": {"sentiment": raw, "confidence": "N/A", "reason": ""}}

    elif intent == "QA":
        result = qa_chain.invoke({"user_input": user_input})
        return {"intent": "QA", "result": result}

    elif intent == "GENERATE":
        result = generate_chain.invoke({"user_input": user_input})
        return {"intent": "GENERATE", "result": result}

    else:
        result = chat_chain.invoke({"user_input": user_input})
        return {"intent": "CHAT", "result": result}

# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data       = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Please enter a message."})
    try:
        result = route(user_input)
        return jsonify(result)
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)