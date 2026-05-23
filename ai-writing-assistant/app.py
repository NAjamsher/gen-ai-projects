# AI Writing Assistant — Phase 2 LangChain Capstone
# Day 15 | Combines: PromptTemplate + Sequential Chain + Memory + Routing
# One tool that improves, rewrites, expands, shortens, and fixes any text

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

# ── MANUAL MEMORY — plain Python list, no imports needed ───────
chat_history_store = []

# ── CORE LLM FUNCTION ─────────────────────────────────────────
def call_llama(prompt_value, temp=0.7, tokens=600):
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

# ── CHAIN 1 — IMPROVE ─────────────────────────────────────────
improve_template = PromptTemplate(
    input_variables=["text"],
    template="""You are a professional editor. Improve the following text.
Fix grammar, clarity, and flow. Keep the original meaning intact.
Only return the improved text. Nothing else.

Text: {text}

Improved:"""
)
improve_chain = improve_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.4, tokens=600)
) | parser

# ── CHAIN 2 — REWRITE ─────────────────────────────────────────
rewrite_template = PromptTemplate(
    input_variables=["text", "tone"],
    template="""Rewrite the following text in a {tone} tone.
Keep the same meaning but change the style completely.
Only return the rewritten text. Nothing else.

Text: {text}

Rewritten ({tone}):"""
)
rewrite_chain = rewrite_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.7, tokens=600)
) | parser

# ── CHAIN 3 — EXPAND ──────────────────────────────────────────
expand_template = PromptTemplate(
    input_variables=["text"],
    template="""Expand the following text. Add more detail, examples, and depth.
Make it approximately 2-3 times longer while keeping the same message.
Only return the expanded text. Nothing else.

Text: {text}

Expanded:"""
)
expand_chain = expand_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.6, tokens=800)
) | parser

# ── CHAIN 4 — SHORTEN ─────────────────────────────────────────
shorten_template = PromptTemplate(
    input_variables=["text"],
    template="""Shorten the following text. Remove unnecessary words and redundancy.
Keep all the key points but make it at least 50 percent shorter.
Only return the shortened text. Nothing else.

Text: {text}

Shortened:"""
)
shorten_chain = shorten_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.3, tokens=400)
) | parser

# ── CHAIN 5 — FIX GRAMMAR ─────────────────────────────────────
grammar_template = PromptTemplate(
    input_variables=["text"],
    template="""Fix all grammar, spelling, and punctuation errors in the following text.
Do not change the meaning or style. Only fix errors.
Only return the corrected text. Nothing else.

Text: {text}

Corrected:"""
)
grammar_chain = grammar_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.1, tokens=600)
) | parser

# ── CHAIN 6 — CHAT WITH AI ABOUT THE TEXT ─────────────────────
chat_template = PromptTemplate(
    input_variables=["history", "text", "question"],
    template="""You are a writing coach helping improve the following text.

Text being discussed:
{text}

Conversation so far:
{history}

User question: {question}

Your response:"""
)
chat_chain = chat_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.7, tokens=400)
) | parser

# ── SEQUENTIAL CHAIN — IMPROVE THEN ANALYSE ───────────────────
def improve_and_analyse(text):
    improved = improve_chain.invoke({"text": text})

    analyse_template = PromptTemplate(
        input_variables=["original", "improved"],
        template="""Compare these two versions of a text.

Original: {original}

Improved: {improved}

List exactly 3 specific changes that were made. Be brief.
Format as:
1. [change]
2. [change]
3. [change]

Changes:"""
    )
    analyse_chain = analyse_template | RunnableLambda(
        lambda x: call_llama(x, temp=0.3, tokens=200)
    ) | parser

    changes = analyse_chain.invoke({
        "original": text,
        "improved": improved
    })

    return {"improved": improved, "changes": changes}

# ── WORD COUNT HELPER ─────────────────────────────────────────
def word_stats(original, result):
    orig_words   = len(original.split())
    result_words = len(result.split())
    diff         = result_words - orig_words
    sign         = "+" if diff > 0 else ""
    return {
        "original_words": orig_words,
        "result_words"  : result_words,
        "diff"          : f"{sign}{diff}"
    }

# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/improve", methods=["POST"])
def improve():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please enter some text."})
    try:
        result = improve_and_analyse(text)
        stats  = word_stats(text, result["improved"])
        return jsonify({
            "result" : result["improved"],
            "changes": result["changes"],
            "stats"  : stats
        })
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

@app.route("/rewrite", methods=["POST"])
def rewrite():
    data = request.get_json()
    text = data.get("text", "").strip()
    tone = data.get("tone", "professional").strip()
    if not text:
        return jsonify({"error": "Please enter some text."})
    try:
        result = rewrite_chain.invoke({"text": text, "tone": tone})
        stats  = word_stats(text, result)
        return jsonify({"result": result, "stats": stats})
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

@app.route("/expand", methods=["POST"])
def expand():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please enter some text."})
    try:
        result = expand_chain.invoke({"text": text})
        stats  = word_stats(text, result)
        return jsonify({"result": result, "stats": stats})
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

@app.route("/shorten", methods=["POST"])
def shorten():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please enter some text."})
    try:
        result = shorten_chain.invoke({"text": text})
        stats  = word_stats(text, result)
        return jsonify({"result": result, "stats": stats})
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

@app.route("/grammar", methods=["POST"])
def grammar():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please enter some text."})
    try:
        result = grammar_chain.invoke({"text": text})
        stats  = word_stats(text, result)
        return jsonify({"result": result, "stats": stats})
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json()
    text     = data.get("text", "").strip()
    question = data.get("question", "").strip()
    if not text or not question:
        return jsonify({"error": "Please provide both text and a question."})
    try:
        history_text = "\n".join([
            f"User: {item['question']}\nAI: {item['answer']}"
            for item in chat_history_store[-5:]
        ])
        response = chat_chain.invoke({
            "history" : history_text,
            "text"    : text,
            "question": question
        })
        chat_history_store.append({"question": question, "answer": response})
        return jsonify({"reply": response})
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

@app.route("/clear", methods=["POST"])
def clear():
    chat_history_store.clear()
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True)