# Day 13 — Gradio Chat UI
# LangChain + Gradio + HuggingFace

import os
import gradio as gr

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# ── LOAD ENV ───────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("HUGGINGFACE_API_KEY")

print("API KEY LOADED")

# ── HUGGINGFACE CLIENT ────────────────────────────────────
client = InferenceClient(
    provider="novita",
    api_key=API_KEY
)

# ── LLM FUNCTION ──────────────────────────────────────────
# ── LLM FUNCTION ──────────────────────────────────────────
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
        max_tokens=500,
        temperature=0.7
    )

    content = response.choices[0].message.content

    if isinstance(content, list):
        return content[0]["text"]

    return str(content).strip()
# ── LANGCHAIN COMPONENTS ──────────────────────────────────
llm = RunnableLambda(call_llama)

parser = StrOutputParser()

# ── PROMPT TEMPLATE ───────────────────────────────────────
prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template="""
You are a helpful AI assistant.

Conversation History:
{history}

User:
{human_input}

Assistant:
"""
)

# ── CHAIN ─────────────────────────────────────────────────
chain = prompt | llm | parser

# ── CHAT FUNCTION ─────────────────────────────────────────
# ── CHAT FUNCTION ─────────────────────────────────────────
def chatbot(user_message, history):

    conversation = ""

    for message in history:

        if message["role"] == "user":
            conversation += f"User: {message['content']}\n"

        elif message["role"] == "assistant":
            conversation += f"Assistant: {message['content']}\n"

    response = chain.invoke({
        "history": conversation,
        "human_input": user_message
    })

    return response
# ── GRADIO UI ─────────────────────────────────────────────
app = gr.ChatInterface(
    fn=chatbot,
    title="Day 13 - AI Chatbot",
    description="LangChain + Gradio + HuggingFace"
)

# ── RUN ───────────────────────────────────────────────────
app.launch()