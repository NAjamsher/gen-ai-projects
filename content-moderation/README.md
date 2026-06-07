# Content Moderation API

An AI-powered content moderation system that classifies text into violation categories, assigns severity scores, and returns structured moderation decisions — built with LangChain and Llama 3.1.

---

## What it does

- Classifies any input text into moderation categories: SAFE, HATE_SPEECH, VIOLENCE, SPAM, ADULT_CONTENT, MISINFORMATION, HARASSMENT
- Returns a severity score (0.0 to 1.0) alongside each classification
- Provides a moderation decision: APPROVE, FLAG, or REJECT
- Gives a brief explanation of why the content was flagged
- Returns structured JSON output — ready to integrate into any application or pipeline
- Handles edge cases with the JSON cleaning pattern to ensure reliable parsing

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI Model | Llama 3.1 8B Instruct |
| Provider | Novita via HuggingFace |
| Framework | LangChain (LCEL) |
| Output | Structured JSON |
| Frontend | HTML, CSS, JavaScript |
| Auth | python-dotenv |

---

## How it works

```
User submits text for moderation
        |
        ▼
LangChain chain formats the moderation prompt
        |
        ▼
Llama 3.1 classifies content at temperature 0.1
(low temperature for reliable, consistent classification)
        |
        ▼
JSON cleaning pattern strips markdown if present
        |
        ▼
Safe JSON parsing with try/except
        |
        ▼
Returns structured decision:
  {
    "category": "HATE_SPEECH",
    "severity": 0.85,
    "decision": "REJECT",
    "reason": "Content contains discriminatory language"
  }
```

---

## Project Structure

```
content-moderation/
├── app.py        ← Flask server, moderation chain, JSON parsing
├── index.html    ← Dark themed UI with moderation results display
├── .env          ← API key (not pushed to GitHub)
└── README.md
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/NAjamsher/gen-ai-projects
cd gen-ai-projects/content-moderation

# Install dependencies
pip install flask flask-cors huggingface-hub langchain-core python-dotenv

# Add your API key
echo "HUGGINGFACE_API_KEY=your_key_here" > .env

# Run the app
python app.py
```

Open browser at `http://127.0.0.1:5001`

---

## Key Implementation Details

- Temperature is set to `0.1` — content moderation must be deterministic and consistent, not creative
- JSON cleaning pattern handles cases where the model wraps output in markdown code blocks
- All seven categories are defined explicitly in the system prompt to prevent hallucinated categories
- Severity score (0.0–1.0) allows downstream systems to apply custom thresholds
- Decision logic: severity below 0.3 → APPROVE, 0.3–0.6 → FLAG, above 0.6 → REJECT

---

## Moderation Categories

| Category | Description |
|---|---|
| SAFE | No violations detected |
| HATE_SPEECH | Discriminatory or hateful language |
| VIOLENCE | Violent threats or graphic content |
| SPAM | Repetitive or promotional spam |
| ADULT_CONTENT | Explicit or inappropriate content |
| MISINFORMATION | False or misleading claims |
| HARASSMENT | Personal attacks or bullying |

---

## Real World Use Cases

Content moderation APIs power every major social platform. This project demonstrates the same pattern used by companies like Meta, Twitter/X, and YouTube to automatically review user-generated content before it reaches other users.

---

## Environment Variables

| Variable | Description |
|---|---|
| HUGGINGFACE_API_KEY | HuggingFace user access token |

---

## License

MIT