# IntelliDocs-AI

**Description**  
IntelliDocs-AI is a document understanding tool powered by AI. It allows a user to upload a document and interact with its contents using natural language queries.
So, instead of going through lengthy files, users can directly ask questions, and IntelliDocs will extract and summarize relevant answers.

---

## Features

- Extract text from one or more PDF documents.
- Ask questions in natural language and receive accurate, context-grounded responses.
- Semantic search across document segments using a FAISS vector store.
- Safe management of API keys through a `.env` file.
- Built with Google's Gemini models for both embeddings and chat.

---

## Tech Stack

- **Frontend & UI**: Streamlit
- **Backend & Processing**: Python 3.10+
- **AI/LLM Framework**: LangChain 1.x (`langchain-core`, `langchain-community`, `langchain-text-splitters`)
- **Vector Database**: FAISS
- **LLM Provider**: Google Gemini (`gemini-flash-latest` for chat, `gemini-embedding-001` for embeddings)
- **Document Parsing**: PyPDF2

---

## Getting Started

### Prerequisites

- Python **3.10 or newer** (current LangChain releases don't support 3.9)
- A Google Gemini API key. Get one for free at [Google AI Studio](https://aistudio.google.com/apikey)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd IntelliDocs-AI

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure your API key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_api_key_here
```

`.env` is listed in `.gitignore`, so your key won't be committed.

### Run the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. Upload one or more PDF files from the sidebar.
2. Click **Submit & Process**. The text is extracted, split into chunks, embedded, and stored in a local FAISS index (`faiss_index/`).
3. Type a question in the main input box. The most relevant chunks are retrieved and sent to Gemini, which answers using only the document's content.

---

## Configuration Notes

- **Chat model**: `gemini-flash-latest` is used by default because it works on Gemini's free tier. If billing is enabled on your Google AI account, you can switch to a Pro model (for example `gemini-pro-latest`) in `get_conversational_chain()` in `app.py`.
- **Embedding model**: `models/gemini-embedding-001`. If you change it, delete `faiss_index/` and re-process your PDFs, since indexes built with different embedding models aren't compatible.

---

## Architecture Diagram

<img width="1600" height="704" alt="image" src="https://github.com/user-attachments/assets/fcf963dc-bc4a-469b-8fb0-2ea0210f395a" />

---
