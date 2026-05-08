# Memory Chatbot — Day 11 | Pillar: LangChain | Conversation Memory

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_classic.memory import ConversationBufferMemory
from huggingface_hub import InferenceClient
import os

# ── FLASK SETUP ───────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── LOAD ENV VARIABLES ───────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ── HUGGINGFACE CLIENT ───────────────────────────────────────
client = InferenceClient(
    provider="novita",
    api_key=API_KEY
)

# ── MEMORY — stores the full conversation ────────────────────
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=False
)

# ── CORE LLM FUNCTION ────────────────────────────────────────
def call_llama(prompt_value):

    prompt_text = (
        prompt_value.text
        if hasattr(prompt_value, "text")
        else str(prompt_value)
    )

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        max_tokens=400,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# ── OUTPUT PARSER ────────────────────────────────────────────
parser = StrOutputParser()

# ── PROMPT TEMPLATE ──────────────────────────────────────────
chat_template = PromptTemplate(
    input_variables=["history", "human_input"],
    template="""
You are a helpful and friendly AI assistant with memory.
You remember everything said earlier in this conversation.

Conversation so far:
{history}

Human:
{human_input}

Your reply (be friendly and reference earlier context when relevant):
"""
)

# ── LANGCHAIN PIPELINE ───────────────────────────────────────
chat_chain = (
    chat_template
    | RunnableLambda(call_llama)
    | parser
)

# ── CHAT FUNCTION ────────────────────────────────────────────
def chat(user_message):

    # Load previous history
    history = memory.load_memory_variables({})["history"]

    # Generate response
    response = chat_chain.invoke({
        "history": history,
        "human_input": user_message
    })

    # Save conversation to memory
    memory.save_context(
        {"input": user_message},
        {"output": response}
    )

    return {
        "reply": response,
        "turn_count": len(memory.chat_memory.messages) // 2
    }

# ── ROUTES ───────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

# ── CHAT ROUTE ───────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat_route():

    data = request.get_json()

    message = data.get("message", "").strip()

    if not message:
        return jsonify({
            "error": "Please type a message."
        })

    try:
        result = chat(message)
        return jsonify(result)

    except Exception as e:
        print(f"ERROR: {e}")

        return jsonify({
            "error": str(e)
        })

# ── CLEAR MEMORY ROUTE ───────────────────────────────────────
@app.route("/clear", methods=["POST"])
def clear_memory():

    memory.clear()

    return jsonify({
        "status": "Memory cleared"
    })

# ── RUN SERVER ───────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)