# Prompt Templates — generates blog intro, tweet, tagline from one topic
# Day 9 | Pillar: LangChain | PromptTemplate deep dive

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
print(f"API KEY LOADED: {API_KEY}") 

print("Setting up LangChain Prompt Templates...")

# ── PROVEN CLIENT ──────────────────────────────────────────────
client = InferenceClient(
    provider="novita",
    api_key=API_KEY,
)

# ── CORE LLM FUNCTION ─────────────────────────────────────────
def call_llama(prompt_value):
    prompt_text = prompt_value.text if hasattr(prompt_value, 'text') else str(prompt_value)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

llm    = RunnableLambda(call_llama)
parser = StrOutputParser()

# ── PROMPT TEMPLATE 1 — BLOG INTRO ────────────────────────────
blog_template = PromptTemplate(
    input_variables=["topic"],
    template="""Write an engaging blog introduction paragraph about {topic}.
Make it interesting and hook the reader in 3-4 sentences.
Only write the introduction paragraph, nothing else.

Blog Introduction:"""
)

# ── PROMPT TEMPLATE 2 — TWEET ─────────────────────────────────
tweet_template = PromptTemplate(
    input_variables=["topic"],
    template="""Write a single engaging tweet about {topic}.
Keep it under 280 characters. Make it punchy and interesting.
Add 2-3 relevant hashtags at the end.
Only write the tweet, nothing else.

Tweet:"""
)

# ── PROMPT TEMPLATE 3 — TAGLINE ───────────────────────────────
tagline_template = PromptTemplate(
    input_variables=["topic"],
    template="""Create a short memorable tagline for {topic}.
Maximum 10 words. Make it catchy and professional.
Only write the tagline, nothing else.

Tagline:"""
)

# ── THREE SEPARATE CHAINS ─────────────────────────────────────
blog_chain    = blog_template    | llm | parser
tweet_chain   = tweet_template   | llm | parser
tagline_chain = tagline_template | llm | parser

print("All chains ready!")

# ── FUNCTION — runs all 3 chains for one topic ─────────────────
def generate_content(topic):
    if not topic.strip():
        return {"error": "Please enter a topic."}
    if len(topic.split()) > 20:
        return {"error": "Topic too long. Keep it under 20 words."}

    blog    = blog_chain.invoke({"topic": topic})
    tweet   = tweet_chain.invoke({"topic": topic})
    tagline = tagline_chain.invoke({"topic": topic})

    return {
        "topic"  : topic,
        "blog"   : blog.strip(),
        "tweet"  : tweet.strip(),
        "tagline": tagline.strip()
    }

# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data  = request.get_json()
    topic = data.get("topic", "")
    if not topic.strip():
        return jsonify({"error": "Please enter a topic."})
    try:
        result = generate_content(topic)
        return jsonify(result)
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)