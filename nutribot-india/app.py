# NutriBot India — RAG-based Indian Nutrition Assistant
# Day 22 | Pillar: Deployment | Platform: HuggingFace Spaces
# Deployed with Gradio — Indian food specific RAG chatbot

import gradio as gr
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from knowledge import KNOWLEDGE_BASE
import os

# ── CONFIGURE ─────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
print(f"API KEY LOADED: {API_KEY}")

# ── SETUP CLIENT ──────────────────────────────────────────────
client = InferenceClient(provider="novita", api_key=API_KEY)

# ── EMBEDDING MODEL ───────────────────────────────────────────
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model ready.")

# ── BUILD KNOWLEDGE CHUNKS ────────────────────────────────────
knowledge_chunks = [
    f"Topic: {item['topic']}\nInformation: {item['content']}"
    for item in KNOWLEDGE_BASE
]

# ── BUILD VECTOR STORE AT STARTUP ─────────────────────────────
print("Building nutrition vector store...")
vector_store = FAISS.from_texts(knowledge_chunks, embeddings)
print(f"NutriBot India ready — {len(knowledge_chunks)} knowledge entries loaded.")

# ── CORE LLM FUNCTION ─────────────────────────────────────────
def call_llama(prompt_value, temp=0.3, tokens=500):
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

# ── NUTRIBOT PROMPT ───────────────────────────────────────────
nutri_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are NutriBot India — an expert Indian nutrition assistant specializing in fitness and health for Indians.
Answer using ONLY the knowledge base information provided below.
Be specific with numbers — grams, calories, protein amounts.
Always give Indian food examples and context.
Keep answers clear, practical, and actionable.
If the information is not in the knowledge base say: I don't have that specific information. Please consult a certified nutritionist for personalized advice.

Knowledge Base:
{context}

User Question: {question}

NutriBot India:"""
)

nutri_chain = nutri_template | llm | parser

# ── RETRIEVE ──────────────────────────────────────────────────
def retrieve(query, top_k=3):
    docs = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]

# ── GENERATE ──────────────────────────────────────────────────
def generate(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    return nutri_chain.invoke({
        "context" : context,
        "question": query
    })

# ── GRADIO CHAT FUNCTION ──────────────────────────────────────
def chat(user_message, history):
    if not user_message.strip():
        return "Please ask a nutrition question."
    try:
        # build conversation history as text
        history_text = ""
        if history:
            for msg in history[-4:]:  # last 4 exchanges only
                if isinstance(msg, dict):
                    role    = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        history_text += f"User: {content}\n"
                    elif role == "assistant":
                        history_text += f"NutriBot: {content}\n"

        # combine history with current question for retrieval
        search_query = user_message
        if history_text:
            search_query = history_text + "\nUser: " + user_message

        retrieved = retrieve(search_query, top_k=3)
        context   = "\n\n".join(retrieved)

        # build prompt with memory
        full_prompt = f"""You are NutriBot India — an expert Indian nutrition assistant.
Answer using ONLY the knowledge base information provided below.
Be specific with numbers — grams, calories, protein amounts.
Always give Indian food examples and context.
If the information is not in the knowledge base say: I don't have that specific information. Please consult a certified nutritionist.

Knowledge Base:
{context}

Conversation so far:
{history_text}
User: {user_message}

NutriBot India:"""

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=500,
            temperature=0.3
        )
        content = response.choices[0].message.content
        if isinstance(content, list):
            return content[0]["text"]
        return str(content).strip()

    except Exception as e:
        return f"Error: {str(e)}"

# ── GRADIO UI ─────────────────────────────────────────────────
# ── GRADIO UI ─────────────────────────────────────────────────
# ── GRADIO UI ─────────────────────────────────────────────────
custom_css = """
/* ── GLOBAL ── */
body, .gradio-container {
    background: #0a0a0f !important;
    font-family: 'Segoe UI', sans-serif !important;
}

/* ── HEADER ── */
.header-box {
    background: linear-gradient(135deg, #1a0a2e 0%, #0a0a1f 50%, #0a0a2e 100%);
    border: 1px solid #2a1a4e;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-bottom: 8px;
}

/* ── STATS BAR ── */
.stats-bar {
    display: flex;
    gap: 12px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 12px 0;
}

.stat-chip {
    background: #0f0f1f;
    border: 1px solid #2a2a4e;
    border-radius: 99px;
    padding: 6px 16px;
    font-size: 12px;
    color: #a89df5;
}

/* ── CHAT WINDOW ── */
.chatbot {
    background: #0f0f17 !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 16px !important;
}

/* ── USER BUBBLE ── */
.message.user, [data-testid="user"] {
    background: #1a1a3e !important;
    color: #c0b8ff !important;
    border-radius: 12px 12px 4px 12px !important;
    border: 1px solid #2a2a5e !important;
}

/* ── BOT BUBBLE ── */
.message.bot, [data-testid="bot"] {
    background: #0a0f1f !important;
    color: #e0e0e0 !important;
    border-radius: 12px 12px 12px 4px !important;
    border: 1px solid #1a3a1a !important;
}

/* ── INPUT BOX ── */
textarea, input[type="text"] {
    background: #0a0a0f !important;
    border: 1.5px solid #2a2a4e !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
    font-size: 14px !important;
}

textarea:focus, input:focus {
    border-color: #7c6ff7 !important;
    box-shadow: 0 0 0 2px #7c6ff720 !important;
}

/* ── SEND BUTTON ── */
button.primary {
    background: linear-gradient(135deg, #7c6ff7, #5a4fd4) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #6a5fe0, #4a3fc4) !important;
    transform: translateY(-1px) !important;
}

/* ── EXAMPLE BUTTONS ── */
.example-set button {
    background: #0f0f1f !important;
    border: 1px solid #2a2a4e !important;
    border-radius: 99px !important;
    color: #a89df5 !important;
    font-size: 12px !important;
    transition: all 0.15s !important;
}

.example-set button:hover {
    border-color: #7c6ff7 !important;
    background: #1a1a3a !important;
    color: #fff !important;
}

/* ── TIPS BOX ── */
.tips-box {
    background: #0f0f17;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 10px;
}

.tip-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 6px 0;
    font-size: 13px;
    color: #888;
    border-bottom: 1px solid #1a1a2a;
}

.tip-item:last-child { border-bottom: none; }
.tip-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }

/* ── FOOTER ── */
.footer-box {
    text-align: center;
    padding: 14px;
    color: #333;
    font-size: 11px;
    margin-top: 8px;
}

/* ── HIDE DEFAULT GRADIO FOOTER ── */
footer { display: none !important; }
.built-with { display: none !important; }
"""

with gr.Blocks(css=custom_css) as demo:

    # ── HEADER ────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-box">
        <div style="font-size:3rem; margin-bottom:8px;">🇮🇳</div>
        <h1 style="font-size:2rem; font-weight:800; color:#fff; margin:0 0 8px 0;
                   background: linear-gradient(135deg, #a89df5, #7c6ff7, #378ADD);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            NutriBot India
        </h1>
        <p style="color:#888; font-size:0.9rem; margin:0;">
            AI Nutrition Assistant for Indian Diets · Powered by Llama 3.1 + RAG + FAISS
        </p>
        <div class="stats-bar">
            <div class="stat-chip">🧬 34 Knowledge Entries</div>
            <div class="stat-chip">🥗 5 Food Categories</div>
            <div class="stat-chip">🍽️ 6 Meal Plans</div>
            <div class="stat-chip">⚡ Semantic Search</div>
        </div>
    </div>
    """)

    # ── CHAT INTERFACE ─────────────────────────────────────────
    gr.ChatInterface(
        fn          = chat,
        chatbot     = gr.Chatbot(
            height          = 460,
            placeholder     = "<div style='text-align:center; color:#333; padding:40px;'>👋 Ask me anything about Indian nutrition, proteins, meal plans, and fitness diet</div>",
            show_label      = False,
        ),
        textbox     = gr.Textbox(
            placeholder = "e.g. How much protein in 100g paneer?",
            container   = False,
            scale       = 7,
            show_label  = False,
        ),
        examples    = [
            "How much protein in 100g paneer?",
            "Give me a vegetarian muscle building meal plan",
            "Is idli good for weight loss?",
            "Best Indian post-workout meal?",
            "How many calories to lose fat at 70kg?",
            "What are the best plant protein sources in India?",
            "How much protein do I need at 65kg?",
            "Pre-workout meal suggestions for Indian diet",
        ],
        title       = "",
        description = "",
    )

    # ── TIPS BOX ───────────────────────────────────────────────
    gr.HTML("""
    <div class="tips-box">
        <div style="font-size:12px; font-weight:700; color:#7c6ff7;
                    text-transform:uppercase; letter-spacing:0.5px; margin-bottom:10px;">
            💡 What You Can Ask
        </div>
        <div class="tip-item">
            <span class="tip-icon">🥩</span>
            <span>Macros of any Indian food — <em style="color:#7c6ff7">
            "How much protein in 100g chicken breast?"</em></span>
        </div>
        <div class="tip-item">
            <span class="tip-icon">🍽️</span>
            <span>Full meal plans — <em style="color:#7c6ff7">
            "Give me a non-veg muscle building meal plan"</em></span>
        </div>
        <div class="tip-item">
            <span class="tip-icon">⚖️</span>
            <span>Weight goals — <em style="color:#7c6ff7">
            "How many calories to lose weight at 75kg?"</em></span>
        </div>
        <div class="tip-item">
            <span class="tip-icon">🏋️</span>
            <span>Workout nutrition — <em style="color:#7c6ff7">
            "Best Indian pre-workout and post-workout meals"</em></span>
        </div>
        <div class="tip-item">
            <span class="tip-icon">🌱</span>
            <span>Vegetarian protein — <em style="color:#7c6ff7">
            "Best plant protein sources for Indian vegetarians"</em></span>
        </div>
    </div>
    """)

    # ── FOOTER ─────────────────────────────────────────────────
    gr.HTML("""
    <div class="footer-box">
        Built with LangChain · FAISS · Llama 3.1 · HuggingFace Spaces
        &nbsp;|&nbsp;
        30-Day GenAI Challenge by Jamsher NA
        &nbsp;|&nbsp;
        <a href="https://github.com/NAjamsher/gen-ai-projects"
           style="color:#7c6ff7; text-decoration:none;">
           GitHub ↗
        </a>
        <br><br>
        <span style="color:#222; font-size:10px;">
            ⚕️ NutriBot India provides general nutrition information only.
            For personalized medical advice consult a certified nutritionist.
        </span>
    </div>
    """)

# ── LAUNCH ────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()