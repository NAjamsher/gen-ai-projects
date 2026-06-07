# Customer Support Bot — Production grade support system
# Day 25 | Industry Project | Concept: Support flows, ticket system, escalation
# Full conversation memory + order lookup + ticket creation + escalation

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from datetime import datetime
import random
import string
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

# ── EMBEDDING MODEL ───────────────────────────────────────────
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model ready.")

# ── KNOWLEDGE BASE ────────────────────────────────────────────
KNOWLEDGE_BASE = [
    {
        "topic"  : "Return policy",
        "content": "We accept returns within 30 days of purchase. Items must be unused and in original packaging with all tags intact. To initiate a return contact support with your order number. Refunds are processed within 5 to 7 business days after we receive the returned item. Damaged or used items are not eligible for return."
    },
    {
        "topic"  : "Shipping and delivery",
        "content": "Standard shipping takes 5 to 7 business days. Express shipping takes 2 to 3 business days and costs extra. Free shipping is available on orders above 500 rupees. International shipping takes 10 to 15 business days. Once shipped you will receive a tracking number via email and SMS."
    },
    {
        "topic"  : "Order cancellation",
        "content": "Orders can be cancelled within 24 hours of placement at no charge. After 24 hours the order enters processing and cannot be cancelled. If you need to cancel contact support immediately with your order number. Cancelled order refunds are processed within 3 to 5 business days."
    },
    {
        "topic"  : "Payment methods",
        "content": "We accept UPI payments, credit cards, debit cards, net banking, and cash on delivery. All online payments are secured with 256-bit SSL encryption. EMI options are available on credit cards for orders above 3000 rupees. Payment issues should be reported immediately as orders may be held."
    },
    {
        "topic"  : "Exchange policy",
        "content": "Exchanges are accepted within 15 days of delivery for unused items in original condition. Size exchanges are free of charge. To request an exchange contact support with order number and the item you want to exchange for. Exchange items are shipped within 2 to 3 business days after receiving the original item."
    },
    {
        "topic"  : "Damaged or wrong item received",
        "content": "If you receive a damaged or wrong item take clear photos immediately and contact support within 48 hours of delivery. We will arrange a free pickup and send a replacement within 3 to 5 business days. In some cases a full refund will be offered instead. Evidence photos are mandatory for processing damaged item claims."
    },
    {
        "topic"  : "Account and login issues",
        "content": "If you cannot login click Forgot Password on the login page and enter your registered email. A reset link will be sent within 5 minutes. Check spam folder if not received. For account lockout after multiple failed attempts wait 30 minutes before trying again. For persistent issues contact support with your registered email address."
    },
    {
        "topic"  : "Discount codes and offers",
        "content": "Discount codes can be applied at checkout in the promo code field. Only one code can be used per order. Codes are case sensitive. Expired codes will show an error message. Subscribe to our newsletter for exclusive discount offers. Referral codes give 10 percent off your next purchase."
    },
    {
        "topic"  : "Loyalty points and rewards",
        "content": "You earn 1 loyalty point for every 10 rupees spent. Points can be redeemed for discounts at a rate of 100 points equals 10 rupees off. Points expire after 12 months of account inactivity. Points cannot be transferred between accounts. Check your point balance in the My Account section."
    },
    {
        "topic"  : "Product availability",
        "content": "If an item shows as out of stock you can click Notify Me to receive an email when it is back in stock. Restocking typically takes 7 to 14 days. We cannot guarantee restock dates. Pre-order items show expected delivery dates at checkout. Limited edition items are available on a first come first served basis."
    },
    {
        "topic"  : "Warranty and product issues",
        "content": "Electronics and appliances come with a manufacturer warranty of 1 year. Warranty claims require proof of purchase and product must show manufacturing defects not physical damage. To claim warranty contact support with order number and photos of the issue. We will coordinate with the manufacturer for repair or replacement."
    },
    {
        "topic"  : "Contact and support hours",
        "content": "Customer support is available Monday to Friday 9 AM to 6 PM and Saturday 10 AM to 4 PM. Email support at support@example.com receives a response within 24 hours. Phone support at 1800-123-4567 is toll free. Live chat on website is available during support hours. Emergency escalations are handled within 2 hours."
    },
]

# ── BUILD FAQ CHUNKS ──────────────────────────────────────────
faq_chunks = [
    f"Topic: {item['topic']}\nPolicy: {item['content']}"
    for item in KNOWLEDGE_BASE
]

print("Building knowledge base vector store...")
vector_store = FAISS.from_texts(faq_chunks, embeddings)
print(f"Knowledge base ready with {len(faq_chunks)} entries.")

# ── SIMULATED ORDER DATABASE ───────────────────────────────────
ORDERS = {
    "ORD001": {"status": "Delivered",   "item": "Blue Denim Jacket",      "date": "2024-01-10", "amount": 1299, "tracking": "TRK789456"},
    "ORD002": {"status": "In Transit",  "item": "Running Shoes Size 9",   "date": "2024-01-14", "amount": 2499, "tracking": "TRK123789"},
    "ORD003": {"status": "Processing",  "item": "Wireless Earbuds",       "date": "2024-01-15", "amount": 1899, "tracking": None},
    "ORD004": {"status": "Cancelled",   "item": "Cotton T-Shirt Pack",    "date": "2024-01-08", "amount": 599,  "tracking": None},
    "ORD005": {"status": "Out for Delivery", "item": "Yoga Mat Premium",  "date": "2024-01-15", "amount": 899,  "tracking": "TRK456123"},
}

# ── TICKET STORE ──────────────────────────────────────────────
tickets = {}

# ── SESSION STORE ─────────────────────────────────────────────
sessions = {}

# ── CORE LLM FUNCTION ─────────────────────────────────────────
def call_llama(prompt_value, temp=0.4, tokens=500):
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

# ── SUPPORT PROMPT ────────────────────────────────────────────
support_template = PromptTemplate(
    input_variables=["context", "history", "message"],
    template="""You are a professional and empathetic customer support agent for an Indian e-commerce company.
Your name is Aria. You are helpful, polite, and solution-focused.
Answer using the policy information provided. Be concise and clear.
If you cannot resolve the issue from the provided information, acknowledge the issue and suggest creating a support ticket.
Never make up information not in the provided context.

Company Policies:
{context}

Conversation History:
{history}

Customer: {message}

Aria:"""
)

support_chain = support_template | llm | parser

# ── INTENT DETECTION ──────────────────────────────────────────
intent_template = PromptTemplate(
    input_variables=["message"],
    template="""Classify this customer support message into exactly one category.
Reply with ONLY the category word. Nothing else.

Categories:
ORDER_STATUS   - customer asking about order status or tracking
RETURN         - customer wants to return an item
CANCEL         - customer wants to cancel an order
COMPLAINT      - customer is complaining about a problem
PAYMENT        - customer has payment related issue
GENERAL        - general question about policy or product
GREETING       - hello hi or general greeting
ESCALATE       - customer is very angry or demanding human agent

Message: {message}

Category:"""
)

intent_chain = intent_template | RunnableLambda(
    lambda x: call_llama(x, temp=0.1, tokens=10)
) | parser

# ── GENERATE TICKET ID ─────────────────────────────────────────
def generate_ticket_id():
    return "TKT" + "".join(random.choices(string.digits, k=6))

# ── CREATE TICKET ──────────────────────────────────────────────
def create_ticket(session_id, issue, priority="Normal"):
    ticket_id = generate_ticket_id()
    tickets[ticket_id] = {
        "id"        : ticket_id,
        "session_id": session_id,
        "issue"     : issue,
        "priority"  : priority,
        "status"    : "Open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "agent"     : None
    }
    return ticket_id

# ── ORDER LOOKUP ───────────────────────────────────────────────
def lookup_order(order_id):
    order_id = order_id.upper().strip()
    if order_id in ORDERS:
        order = ORDERS[order_id]
        response = f"Order {order_id} details:\n"
        response += f"Item: {order['item']}\n"
        response += f"Status: {order['status']}\n"
        response += f"Order Date: {order['date']}\n"
        response += f"Amount: ₹{order['amount']}\n"
        if order['tracking']:
            response += f"Tracking Number: {order['tracking']}"
        else:
            response += "Tracking: Not yet available"
        return response
    return None

# ── RETRIEVE POLICY ────────────────────────────────────────────
def retrieve_policy(query, top_k=2):
    docs = vector_store.similarity_search(query, k=top_k)
    return "\n\n".join([doc.page_content for doc in docs])

# ── GENERATE SUPPORT RESPONSE ──────────────────────────────────
def generate_response(message, history_text, context):
    return support_chain.invoke({
        "context": context,
        "history": history_text,
        "message": message
    })

# ── EXTRACT ORDER ID FROM MESSAGE ──────────────────────────────
def extract_order_id(message):
    words = message.upper().split()
    for word in words:
        if word.startswith("ORD") and len(word) == 6:
            return word
    return None

# ── MAIN CONVERSATION HANDLER ──────────────────────────────────
def handle_message(session_id, message):
    # get or create session
    if session_id not in sessions:
        sessions[session_id] = {
            "history"   : [],
            "ticket_id" : None,
            "escalated" : False
        }

    session = sessions[session_id]

    # detect intent
    raw_intent = intent_chain.invoke({"message": message})
    intent     = raw_intent.strip().upper()

    for keyword in ["ORDER_STATUS", "RETURN", "CANCEL", "COMPLAINT", "PAYMENT", "GENERAL", "GREETING", "ESCALATE"]:
        if keyword in intent:
            intent = keyword
            break
    else:
        intent = "GENERAL"

    print(f"INTENT: {intent}")

    # build history text
    history_text = ""
    for item in session["history"][-6:]:
        history_text += f"Customer: {item['user']}\nAria: {item['bot']}\n\n"

    # handle by intent
    response     = ""
    ticket_id    = None
    escalated    = False
    order_info   = None

    if intent == "GREETING":
        response = "Hello! Welcome to our customer support. I am Aria, your support assistant. How can I help you today?"

    elif intent == "ORDER_STATUS":
        order_id = extract_order_id(message)
        if order_id:
            order_details = lookup_order(order_id)
            if order_details:
                order_info = order_details
                context    = retrieve_policy("order tracking shipping delivery")
                response   = generate_response(
                    f"Customer asking about order {order_id}. Order details: {order_details}",
                    history_text,
                    context
                )
            else:
                response = f"I could not find order {order_id} in our system. Please check the order ID and try again. You can find your order ID in your confirmation email."
        else:
            response = "I can help you check your order status. Please provide your order ID. It looks like ORD001 and can be found in your order confirmation email."

    elif intent == "RETURN":
        context  = retrieve_policy("return policy refund")
        response = generate_response(message, history_text, context)

    elif intent == "CANCEL":
        context  = retrieve_policy("cancellation policy cancel order")
        response = generate_response(message, history_text, context)

    elif intent == "PAYMENT":
        context  = retrieve_policy("payment methods EMI")
        response = generate_response(message, history_text, context)

    elif intent == "ESCALATE" or session["escalated"]:
        escalated  = True
        ticket_id  = create_ticket(session_id, message, priority="High")
        session["escalated"] = True
        response   = f"I completely understand your frustration and I sincerely apologize for the inconvenience. I have escalated your case to our senior support team. Your ticket ID is {ticket_id}. A human agent will contact you within 2 hours. Is there anything else I can note for them?"

    elif intent == "COMPLAINT":
        context  = retrieve_policy(message)
        response = generate_response(message, history_text, context)
        # auto create ticket for complaints
        ticket_id = create_ticket(session_id, message, priority="Normal")
        response += f"\n\nI have also created a support ticket {ticket_id} for your complaint. Our team will follow up within 24 hours."

    else:
        context  = retrieve_policy(message)
        response = generate_response(message, history_text, context)

    # save to history
    session["history"].append({
        "user": message,
        "bot" : response
    })

    return {
        "response"  : response,
        "intent"    : intent,
        "ticket_id" : ticket_id,
        "order_info": order_info,
        "escalated" : escalated,
        "session_id": session_id
    }

# ── ROUTES ────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data       = request.get_json()
    session_id = data.get("session_id", "default")
    message    = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Please enter a message."})
    try:
        result = handle_message(session_id, message)
        return jsonify(result)
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)})

@app.route("/ticket/<ticket_id>", methods=["GET"])
def get_ticket(ticket_id):
    if ticket_id in tickets:
        return jsonify(tickets[ticket_id])
    return jsonify({"error": "Ticket not found."})

@app.route("/tickets", methods=["GET"])
def get_all_tickets():
    return jsonify({"tickets": list(tickets.values()), "total": len(tickets)})

@app.route("/orders/<order_id>", methods=["GET"])
def get_order(order_id):
    result = lookup_order(order_id)
    if result:
        return jsonify({"order": ORDERS[order_id.upper()]})
    return jsonify({"error": "Order not found."})

@app.route("/clear", methods=["POST"])
def clear():
    data       = request.get_json()
    session_id = data.get("session_id", "default")
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({"status": "Session cleared."})

if __name__ == "__main__":
    app.run(debug=True, port=5001)