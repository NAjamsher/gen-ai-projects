AI Writing Assistant
An AI-powered writing tool that generates blogs, tweets, and taglines from a single topic — built with LangChain and Llama 3.1.

What it does

Paste any topic — get a full blog post, tweet, and tagline instantly
Three independent LangChain chains run in parallel — each optimized for its output type
Blog posts are detailed and structured
Tweets are punchy and under 280 characters
Taglines are sharp one-liners
Dark themed responsive UI built with HTML and CSS


Tech Stack
LayerTechnologyBackendPython, FlaskAI ModelLlama 3.1 8B InstructProviderNovita via HuggingFaceFrameworkLangChain (LCEL)FrontendHTML, CSS, JavaScriptAuthpython-dotenv

How it works
User inputs topic
      |
      ▼
Three independent LangChain chains fire simultaneously
      |
      ├── Blog chain    (temperature 0.7 — creative, detailed)
      ├── Tweet chain   (temperature 0.7 — punchy, concise)
      └── Tagline chain (temperature 0.7 — sharp, memorable)
      |
      ▼
Flask returns all three outputs in one response
      |
      ▼
UI displays blog, tweet, and tagline side by side

Project Structure
ai-writing-assistant/
├── app.py        ← Flask server with three LangChain chains
├── index.html    ← Dark themed single page UI
├── .env          ← API key (not pushed to GitHub)
└── README.md

Getting Started
bash# Clone the repository
git clone https://github.com/NAjamsher/gen-ai-projects
cd gen-ai-projects/ai-writing-assistant

# Install dependencies
pip install flask flask-cors huggingface-hub langchain-core python-dotenv

# Add your API key
echo "HUGGINGFACE_API_KEY=your_key_here" > .env

# Run the app
python app.py
Open browser at http://127.0.0.1:5000

Key Implementation Details

Uses LCEL pipe operator — prompt | llm | parser — for clean chain composition
Each chain has its own PromptTemplate tuned for that content type
RunnableLambda wraps the HuggingFace InferenceClient to make it LangChain compatible
StrOutputParser ensures clean string output without metadata


Environment Variables
VariableDescriptionHUGGINGFACE_API_KEYHuggingFace user access token

License
MIT