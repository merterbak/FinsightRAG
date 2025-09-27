# Finsight RAG

It is a Streamlit application that pulls SEC filings (10-Q, 10-K) straight from EDGAR with no api, performs Mistral OCR on HTML/PDF documents, lets you chat with a Mistral powered LLM with streaming and RAG, and generates exportable reports.

---

## 1. Prerequisites

- Valid Mistral API key

### Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```
### Launch Playwright install:

```bash
playwright install
```

---

## 2. Environment configuration

Place a `.env` file beside `app.py` or set variables in your UI:

```env
MISTRAL_API_KEY=your-mistral-key
```

---

## 3. Run the application

```bash
streamlit run app.py
```


---

## 4. Feature overview

- **Fetch & Index** – Quickly grab filings by ticker, form, year, or quarter, or just upload a PDF/HTML yourself. The app breaks them into chunks and builds a smart search index.
- **Chat** – Ask questions. Mistral model streams back answers with extra reranking to make sure you get the most relevant info.
- **Generate Report** – Turn filings to markdown reports. You can export it straight to PDF.
- **OCR Preview** – Clean up messy HTML/PDF filings with Mistral OCR and get an easy to read markdown view with extracted images.


---

## 5. Compliance reminders

- Respect SEC guidance (~10 requests/second) and include your identity in every request.

---

## 6. Disclaimer

This application does not provide investment advice. The content is for informational and educational purposes only, aiming to assist users in exploring and understanding SEC filings.

---

## 7. Future Work

 - Better reports
 - Fix issue where exported images were not showing in Streamlit



