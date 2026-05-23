# FAQ Bot — semantic question answering from a business knowledge base
# Day 21 | Pillar: RAG | Industry use case: Customer Support
# Same RAG pipeline as Day 19/20 — applied to FAQ knowledge base

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
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

client = InferenceClient(provider="novita", api_key=API_KEY)

# ── EMBEDDING MODEL ───────────────────────────────────────────
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model ready.")

# ── FAQ KNOWLEDGE BASE ────────────────────────────────────────
# Format: each item has a question and answer
# The question + answer are combined into one chunk for embedding
# This way semantic search matches both question phrasing and answer content

FAQ = [
    {
        "question": "What are your business hours?",
        "answer"  : "We are open Monday to Friday from 9 AM to 6 PM and Saturday from 10 AM to 4 PM. We are closed on Sundays and public holidays."
    },
    {
        "question": "What is your return policy?",
        "answer"  : "We accept returns within 30 days of purchase. Items must be unused and in original packaging. To initiate a return, contact our support team with your order number."
    },
    {
        "question": "How long does shipping take?",
        "answer"  : "Standard shipping takes 5 to 7 business days. Express shipping takes 2 to 3 business days. International shipping takes 10 to 15 business days."
    },
    {
        "question": "Do you offer free shipping?",
        "answer"  : "Yes, we offer free standard shipping on all orders above 500 rupees. Orders below 500 rupees have a flat shipping fee of 50 rupees."
    },
    {
        "question": "How do I track my order?",
        "answer"  : "Once your order is shipped you will receive a tracking number via email. You can use this number on our website or the courier's website to track your package."
    },
    {
        "question": "What payment methods do you accept?",
        "answer"  : "We accept UPI, credit cards, debit cards, net banking, and cash on delivery. All online payments are secured with 256-bit encryption."
    },
    {
        "question": "Can I cancel my order?",
        "answer"  : "Orders can be cancelled within 24 hours of placement. After 24 hours the order enters processing and cannot be cancelled. Contact support immediately if you need to cancel."
    },
    {
        "question": "How do I contact customer support?",
        "answer"  : "You can reach our customer support team by email at support@example.com, by phone at 1800-123-4567 (toll free), or through the live chat on our website. Support is available Monday to Friday 9 AM to 6 PM."
    },
    {
        "question": "Do you have a loyalty program?",
        "answer"  : "Yes, we have a rewards program. You earn 1 point for every 10 rupees spent. Points can be redeemed for discounts on future purchases. Sign up on our website to join."
    },
    {
        "question": "Is my personal data safe with you?",
        "answer"  : "Yes, we take data privacy very seriously. We never sell your personal data to third parties. All data is stored securely and encrypted. You can request deletion of your data at any time by contacting support."
    },
    {
        "question": "Do you offer discounts or coupons?",
        "answer"  : "Yes, we regularly offer discount codes through our newsletter and social media channels. Subscribe to our newsletter to receive exclusive offers and seasonal promotions."
    },
    {
        "question": "What happens if my order is damaged?",
        "answer"  : "If your order arrives damaged, take photos immediately and contact support within 48 hours. We will arrange a free replacement or full refund depending on your preference."
    },
    {
        "question": "Can I exchange a product?",
        "answer"  : "Yes, exchanges are accepted within 15 days of delivery for unused items in original condition. Contact support to initiate an exchange and we will guide you through the process."
    },
    {
        "question": "Do you ship internationally?",
        "answer"  : "Yes, we ship to over 30 countries. International orders may be subject to customs duties and taxes which are the responsibility of the buyer. Check our shipping page for the full country list."
    },
    {
        "question": "How do I reset my password?",
        "answer"  : "Click Forgot Password on the login page and enter your registered email address. You will receive a password reset link within 5 minutes. Check your spam folder if you do not see it."
    },
]

# ── BUILD FAQ CHUNKS ──────────────────────────────────────────
# combine question and answer into one searchable text chunk
faq_chunks = [
    f"Question: {item['question']}\nAnswer: {item['answer']}"
    for item in FAQ
]

# ── BUILD VECTOR STORE AT STARTUP ─────────────────────────────
print("Building FAQ vector store...")
vector_store = FAISS.from_texts(faq_chunks, embeddings)
print(f"FAQ vector store ready with {len(faq_chunks)} entries.")

# ── CORE LLM FUNCTION ─────────────────────────────────────────
def call_llama(prompt_value, temp=0.3, tokens=300):
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

# ── FAQ PROMPT ────────────────────────────────────────────────
faq_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful customer support assistant.
Answer the customer's question using ONLY the FAQ information provided below.
Be friendly, concise, and professional.
If the answer is not in the FAQ say: I'm sorry, I don't have information about that. Please contact our support team directly.

FAQ Information:
{context}

Customer Question: {question}

Answer:"""
)

faq_chain = faq_template | llm | parser

# ── RETRIEVE ──────────────────────────────────────────────────
def retrieve(query, top_k=2):
    docs = vector_store.similarity_search(query, k=top_k)
    return [{"text": doc.page_content, "rank": i + 1} for i, doc in enumerate(docs)]

# ── GENERATE ──────────────────────────────────────────────────
def generate(query, retrieved):
    context = "\n\n".join([r["text"] for r in retrieved])
    return faq_chain.invoke({"context": context, "question": query})

# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data  = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Please enter a question."})
    try:
        retrieved = retrieve(query, top_k=2)
        answer    = generate(query, retrieved)
        return jsonify({
            "answer" : answer,
            "sources": retrieved
        })
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

@app.route("/faqs", methods=["GET"])
def get_faqs():
    return jsonify({"faqs": FAQ, "total": len(FAQ)})

if __name__ == "__main__":
    app.run(debug=True, port=5001)