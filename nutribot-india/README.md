# NutriBot India 🇮🇳

AI Nutrition Assistant for Indian Diets — powered by RAG, LangChain, FAISS, and Llama 3.1.
Ask anything about Indian foods, macros, meal plans, and fitness nutrition.

🔴 **Live Demo:** [huggingface.co/spaces/NAjamsher/nutribot-india](https://huggingface.co/spaces/NAjamsher/nutribot-india)

---

## What it does

- Answers nutrition questions specific to Indian foods
- Provides exact macros for dal, paneer, rice, roti, idli, dosa, eggs, chicken, and more
- Gives complete Indian meal plans for muscle building and weight loss
- Covers vegetarian and non-vegetarian Indian fitness diets
- Explains nutrition concepts — protein per kg, calorie deficit, surplus, sleep, hydration
- Answers questions semantically — finds the right answer even if phrasing differs

---

## Example Questions
"How much protein in 100g paneer?"
"What to eat for muscle gain as a vegetarian Indian?"
"Is idli good for weight loss?"
"Best Indian post-workout meal?"
"Give me a South Indian weight loss meal plan"
"How many calories to lose weight at 70kg?"
"What are the best plant protein sources in India?"

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.1 8B Instruct via HuggingFace |
| Framework | LangChain (PromptTemplate, LCEL pipe operator) |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| UI | Gradio |
| Deployment | HuggingFace Spaces |

---

## How it works
User asks a nutrition question
↓
Question embedded into 384-dimensional vector
↓
FAISS similarity search across 25+ Indian nutrition knowledge entries
↓
Top 3 most relevant knowledge chunks retrieved
↓
LangChain PromptTemplate fills context + question
↓
Llama 3.1 generates grounded answer from retrieved context only
↓
NutriBot India responds with specific numbers and Indian examples

---

## Knowledge Base Coverage

- **Protein Sources** — Paneer, chicken, eggs, dal, rajma, soya chunks, fish, whey, tofu, curd
- **Carbohydrate Sources** — Rice, roti, idli, dosa, oats, banana, sweet potato, poha
- **Fat Sources** — Ghee, nuts, peanut butter
- **Meal Plans** — Vegetarian muscle building, non-vegetarian muscle building, South Indian weight loss, North Indian weight loss, pre-workout, post-workout
- **Nutrition Concepts** — Protein per kg rule, calorie deficit, calorie surplus, sleep, hydration, creatine

---

## Project Structure
nutribot-india/
├── app.py          — Gradio UI + RAG pipeline
├── knowledge.py    — Indian nutrition knowledge base (25+ entries)
├── requirements.txt— dependencies
├── .env            — API key (local only, never pushed)
└── README.md

---

## Run Locally

```bash
git clone https://github.com/NAjamsher/gen-ai-projects
cd gen-ai-projects/nutribot-india
pip install -r requirements.txt
```

Create `.env`:
HUGGINGFACE_API_KEY=hf_your_key_here

```bash
python app.py
```

---

## Deploy to HuggingFace Spaces

1. Create a new Space at huggingface.co/new-space
2. Select Gradio as the SDK
3. Push these files: `app.py`, `knowledge.py`, `requirements.txt`, `README.md`
4. Add `HUGGINGFACE_API_KEY` as a Space Secret in Settings
5. Space builds and deploys automatically

---

## Part of 30-Day GenAI Challenge

This is Day 22 of a 30-day challenge building one GenAI project per day.
Full portfolio: [github.com/NAjamsher/gen-ai-projects](https://github.com/NAjamsher/gen-ai-projects)

---

## License

MIT