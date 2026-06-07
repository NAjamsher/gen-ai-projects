# Content Moderation API
# Day 26 | Industry Project | Concept: Text classification, confidence scoring, action routing
# Detects hate speech, spam, abuse, adult content, misinformation, and safe content

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from datetime import datetime
import json
import os

# ── CONFIGURE ─────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
print(f"API KEY LOADED: {API_KEY}")

# ── SETUP ─────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

client = InferenceClient(provider="novita", api_key=API_KEY)

# ── MODERATION LOG ────────────────────────────────────────────
moderation_log = []

# ── CORE LLM FUNCTION ─────────────────────────────────────────
def call_llama(prompt_value, temp=0.1, tokens=300):
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

# ── MODERATION PROMPT ─────────────────────────────────────────
moderation_template = PromptTemplate(
    input_variables=["content"],
    template="""You are a content moderation AI. Analyze the following content and classify it.

Return ONLY a valid JSON object with these exact keys:
- "verdict": one of SAFE, HATE_SPEECH, SPAM, ABUSE, ADULT, MISINFORMATION, TOXIC
- "confidence": a number from 0 to 100 representing how confident you are
- "reason": one sentence explaining why
- "action": one of APPROVE, FLAG_FOR_REVIEW, AUTO_REMOVE
- "severity": one of LOW, MEDIUM, HIGH

Rules for action:
- APPROVE if verdict is SAFE
- FLAG_FOR_REVIEW if confidence is below 80 or severity is LOW or MEDIUM
- AUTO_REMOVE if confidence is above 80 AND severity is HIGH

Content to analyze:
{content}

JSON:"""
)

moderation_chain = moderation_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.1, tokens=200)
) | parser

# ── BATCH MODERATION PROMPT ───────────────────────────────────
batch_template = PromptTemplate(
    input_variables=["items"],
    template="""You are a content moderation AI. Analyze each piece of content below.

For each item return a JSON object in an array with these keys:
- "index": the item number starting from 0
- "verdict": one of SAFE, HATE_SPEECH, SPAM, ABUSE, ADULT, MISINFORMATION, TOXIC
- "confidence": number from 0 to 100
- "action": one of APPROVE, FLAG_FOR_REVIEW, AUTO_REMOVE
- "severity": one of LOW, MEDIUM, HIGH

Return ONLY a valid JSON array. Nothing else.

Items to analyze:
{items}

JSON array:"""
)

batch_chain = batch_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.1, tokens=600)
) | parser

# ── CLEAN JSON ────────────────────────────────────────────────
def clean_json(raw):
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()

# ── MODERATE SINGLE CONTENT ────────────────────────────────────
def moderate(content):
    raw    = moderation_chain.invoke({"content": content})
    raw    = clean_json(raw)
    result = json.loads(raw)

    # ensure all keys exist
    result.setdefault("verdict",    "SAFE")
    result.setdefault("confidence", 50)
    result.setdefault("reason",     "No reason provided")
    result.setdefault("action",     "FLAG_FOR_REVIEW")
    result.setdefault("severity",   "LOW")

    # log it
    log_entry = {
        "id"         : len(moderation_log) + 1,
        "content"    : content[:100] + ("..." if len(content) > 100 else ""),
        "verdict"    : result["verdict"],
        "confidence" : result["confidence"],
        "action"     : result["action"],
        "severity"   : result["severity"],
        "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    moderation_log.append(log_entry)

    return result

# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

# single content moderation
@app.route("/moderate", methods=["POST"])
def moderate_single():
    data    = request.get_json()
    content = data.get("content", "").strip()

    if not content:
        return jsonify({"error": "Please provide content to moderate."})

    if len(content) > 2000:
        return jsonify({"error": "Content too long. Maximum 2000 characters."})

    try:
        result = moderate(content)
        return jsonify(result)
    except json.JSONDecodeError:
        return jsonify({"error": "Could not parse moderation result. Try again."})
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

# batch moderation — multiple items at once
@app.route("/moderate/batch", methods=["POST"])
def moderate_batch():
    data  = request.get_json()
    items = data.get("items", [])

    if not items:
        return jsonify({"error": "Please provide items to moderate."})

    if len(items) > 10:
        return jsonify({"error": "Maximum 10 items per batch."})

    try:
        # format items for prompt
        items_text = "\n".join([
            f"Item {i}: {item}" for i, item in enumerate(items)
        ])

        raw    = batch_chain.invoke({"items": items_text})
        raw    = clean_json(raw)
        results = json.loads(raw)

        # log each result
        for i, result in enumerate(results):
            log_entry = {
                "id"         : len(moderation_log) + 1,
                "content"    : items[i][:100] if i < len(items) else "",
                "verdict"    : result.get("verdict", "SAFE"),
                "confidence" : result.get("confidence", 50),
                "action"     : result.get("action", "FLAG_FOR_REVIEW"),
                "severity"   : result.get("severity", "LOW"),
                "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            moderation_log.append(log_entry)

        return jsonify({"results": results, "total": len(results)})

    except json.JSONDecodeError:
        return jsonify({"error": "Could not parse batch results. Try again."})
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

# get moderation log
@app.route("/log", methods=["GET"])
def get_log():
    # stats
    total     = len(moderation_log)
    approved  = sum(1 for l in moderation_log if l["action"] == "APPROVE")
    flagged   = sum(1 for l in moderation_log if l["action"] == "FLAG_FOR_REVIEW")
    removed   = sum(1 for l in moderation_log if l["action"] == "AUTO_REMOVE")

    return jsonify({
        "log"     : moderation_log[-20:],
        "stats"   : {
            "total"   : total,
            "approved": approved,
            "flagged" : flagged,
            "removed" : removed
        }
    })

# clear log
@app.route("/log/clear", methods=["POST"])
def clear_log():
    moderation_log.clear()
    return jsonify({"message": "Log cleared."})

if __name__ == "__main__":
    app.run(debug=True, port=5001)