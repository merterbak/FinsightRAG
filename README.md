# FinsightRAG
Helper LLM app with RAG and OCR for very long reports like 10-Q, 10-K and earnings reports filed with the SEC. App is powered by mistral models and sec-api.


## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create .env then run
streamlit run app.py
```

## Usage
1. Open the app → sidebar “Configuration” → paste keys (or rely on .env)
2. Fetch & Index: enter `Ticker`, optionally enable Year/Quarter → Fetch
3. Chat: ask questions view “Sources for latest question” on the right
4. Generate Report: choose sections, set target pages → Generate → Download
