# IntelliDocs-AI

**Description**  
IntelliDocs-AI is a document understanding tool powered by AI. It allows a user to upload a document and interact with its contents using natural language queries.
So, instead of going through lengthy files, users can directly ask questions, and IntelliDocs will extract and summarize relevant answers.

---

## Features

- **Chat interface**: ask follow-up questions in a conversation, with answers streamed as they're written.
- **Source citations**: every answer shows which file and page it came from, with an excerpt.
- **Multiple PDFs**: upload several documents and query them together.
- **Suggested questions** to get started quickly, plus one-click *Clear chat* and *Start over*.
- **Clean, modern theme** with automatic light and dark mode.
- **Helpful errors** for a missing API key, scanned PDFs without selectable text, and API failures.
- Semantic search across document segments using a FAISS vector store.
- Built with Google's Gemini models for both embeddings and chat.

---

## Tech Stack

- **Frontend & UI**: Streamlit
- **Backend & Processing**: Python 3.10+
- **AI/LLM Framework**: LangChain 1.x (`langchain-core`, `langchain-community`, `langchain-text-splitters`)
- **Vector Database**: FAISS
- **LLM Provider**: Google Gemini (`gemini-flash-lite-latest` for chat, `gemini-embedding-001` for embeddings)
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
2. Click **Process documents**. The text is extracted page by page, split into chunks, embedded, and stored in an in-memory FAISS index for your session.
3. Ask a question in the chat box, or pick a suggested one. The most relevant chunks are retrieved and sent to Gemini, which answers using only the documents' content.
4. Expand **Sources** under any answer to see the file, page and excerpt it was based on.

Documents are kept only for your browser session. Refreshing the page or clicking **Start over** clears them.

---

## Configuration Notes

- **Chat model**: `gemini-flash-lite-latest` is used by default because it is fast (answers usually start within a few seconds). Free-tier keys have a daily request limit per model; see [ai.dev/rate-limit](https://ai.dev/rate-limit). If billing is enabled on your Google AI account, you can switch to a Pro model (for example `gemini-pro-latest`) by changing `CHAT_MODEL` in `app.py`.
- **Embedding model**: `models/gemini-embedding-001`.
- Both model names are set at the top of `app.py` (`CHAT_MODEL`, `EMBEDDING_MODEL`).
- **Theme**: colors, font and corner radius live in `.streamlit/config.toml`.

---

## Architecture Diagram

<img width="1600" height="704" alt="image" src="https://github.com/user-attachments/assets/fcf963dc-bc4a-469b-8fb0-2ea0210f395a" />

---
