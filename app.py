import itertools
import os
import uuid
from collections import OrderedDict
from types import SimpleNamespace

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-flash-lite-latest"

AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}

# Number of browser sessions whose documents are kept in memory.
MAX_WORKSPACES = 20

# Streamlit's frontend remembers the chosen theme in localStorage under this key and
# only reads it on page load. If the toggle stops working after a Streamlit upgrade,
# check this key against the frontend bundle (search for "stActiveTheme").
THEME_STORAGE_KEY = "stActiveTheme-{path}-v2"

SUGGESTED_QUESTIONS = [
    "Summarize this document",
    "What are the key points?",
    "List any important dates or numbers",
    "Explain the main conclusion",
]

# Small style touches the theme config can't express. Targets containers by their
# `key`, which Streamlit exposes as an `st-key-<key>` class.
CUSTOM_CSS = """
<style>
.st-key-brand h2 {
    font-size: 2.25rem !important;
    line-height: 1.1;
    padding: 0;
}
[class*="st-key-step-"] {
    background: rgba(128, 128, 128, 0.06);
    padding: 1.25rem 1.1rem 1.1rem;
    transition: border-color 0.15s ease;
}
[class*="st-key-step-"]:hover {
    border-color: rgba(180, 83, 42, 0.55);
}
[class*="st-key-step-"] h4 {
    font-size: 1.6rem;
    padding: 0.4rem 0 0;
}
</style>
"""

PROMPT_TEMPLATE = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer.
Format the answer in clear Markdown, using short paragraphs, bullet points or tables where they help.
If you use headings, use level-4 headings (####) only.

Context:
{context}

Question:
{question}

Answer:
"""


# ---------- Document processing ----------

def get_pdf_pages(pdf_docs):
    """Extract text page by page, keeping the file name and page number for citations."""
    pages, doc_info = [], []
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        readable = 0
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                readable += 1
                pages.append((text, {"source": pdf.name, "page": number}))
        doc_info.append({"name": pdf.name, "pages": len(reader.pages), "readable": readable})
    return pages, doc_info


def get_text_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    texts, metadatas = zip(*pages)
    return text_splitter.create_documents(list(texts), metadatas=list(metadatas))


@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store(chunks):
    return FAISS.from_documents(chunks, embedding=get_embeddings())


@st.cache_resource
def get_conversational_chain():
    # Few retries so quota errors surface quickly instead of leaving the spinner running.
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3, max_retries=1)
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    return prompt | model


def stream_answer(question, docs):
    context = "\n\n".join(doc.page_content for doc in docs)
    for chunk in get_conversational_chain().stream({"context": context, "question": question}):
        yield chunk.text


# ---------- State ----------

@st.cache_resource
def workspaces():
    return OrderedDict()


def init_state():
    """Attach this browser tab to a workspace identified by a URL parameter.

    Documents and chat live in the workspace rather than in st.session_state, so they
    survive a page reload (which the theme toggle needs).
    """
    st.session_state.setdefault("uploader_key", 0)
    store = workspaces()
    workspace_id = st.query_params.get("w")
    if workspace_id not in store:
        workspace_id = uuid.uuid4().hex[:12]
        store[workspace_id] = SimpleNamespace(
            messages=[], vector_store=None, documents=[], unreadable=[]
        )
        st.query_params["w"] = workspace_id
    store.move_to_end(workspace_id)
    while len(store) > MAX_WORKSPACES:
        store.popitem(last=False)
    st.session_state.workspace = store[workspace_id]


def workspace():
    return st.session_state.workspace


# ---------- UI helpers ----------

def render_theme_toggle():
    is_dark = st.context.theme.type == "dark"
    if st.button(
        "",
        icon=":material/light_mode:" if is_dark else ":material/dark_mode:",
        type="tertiary",
        help="Switch to light mode" if is_dark else "Switch to dark mode",
        key="theme_toggle",
    ):
        st.html(
            f"""<script>
            localStorage.setItem(
                "{THEME_STORAGE_KEY}".replace("{{path}}", window.location.pathname),
                JSON.stringify("{'Light' if is_dark else 'Dark'}")
            );
            window.location.reload();
            </script>""",
            unsafe_allow_javascript=True,
        )


def reset_documents():
    workspace().vector_store = None
    workspace().documents = []
    workspace().messages = []
    workspace().unreadable = []
    st.session_state.uploader_key += 1


def ask_suggestion():
    st.session_state.pending_question = st.session_state.suggestion
    st.session_state.suggestion = None


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"**{source['source']}** · page {source['page']}")
            st.caption(source["excerpt"])


def collect_sources(docs):
    sources, seen = [], set()
    for doc in docs:
        key = (doc.metadata["source"], doc.metadata["page"])
        if key in seen:
            continue
        seen.add(key)
        excerpt = " ".join(doc.page_content.split())
        sources.append({
            "source": key[0],
            "page": key[1],
            "excerpt": excerpt[:280] + ("…" if len(excerpt) > 280 else ""),
        })
    return sources


def render_sidebar():
    with st.sidebar:
        brand, toggle = st.columns([5, 1], vertical_alignment="center")
        with brand.container(key="brand"):
            st.markdown("## IntelliDocs")
        with toggle:
            render_theme_toggle()
        st.caption("Ask questions about your PDFs and get answers with page references.")

        if not os.getenv("GOOGLE_API_KEY"):
            st.error(
                "No `GOOGLE_API_KEY` found. Add it to a `.env` file in the project folder.",
                icon=":material/key_off:",
            )

        st.space("small")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )

        processed_names = {doc["name"] for doc in workspace().documents}
        has_new_files = bool(pdf_docs) and {f.name for f in pdf_docs} != processed_names

        if st.button(
            "Process documents",
            type="primary",
            disabled=not has_new_files,
            width="stretch",
        ):
            process_documents(pdf_docs)

        if workspace().documents:
            st.space("small")
            st.markdown("**Your documents**")
            with st.container(border=True):
                for doc in workspace().documents:
                    pages = f"{doc['pages']} page{'s' if doc['pages'] != 1 else ''}"
                    st.markdown(f"**{doc['name']}**  \n:gray[{pages}]")

            col1, col2 = st.columns(2)
            col1.button(
                "Clear chat",
                width="stretch",
                disabled=not workspace().messages,
                on_click=lambda: workspace().messages.clear(),
            )
            col2.button(
                "Start over",
                width="stretch",
                on_click=reset_documents,
            )

        st.space("medium")
        with st.expander("How it works"):
            st.markdown(
                "1. Text is extracted from each page of your PDFs.\n"
                "2. It's split into chunks and embedded with Gemini.\n"
                "3. Your question finds the most relevant chunks in a FAISS index.\n"
                "4. Gemini answers using only those chunks, and shows its sources."
            )


def process_documents(pdf_docs):
    with st.status("Processing documents…", expanded=True) as status:
        try:
            st.write("Extracting text")
            pages, doc_info = get_pdf_pages(pdf_docs)

            unreadable = [doc["name"] for doc in doc_info if doc["readable"] == 0]
            if not pages:
                status.update(label="No readable text found", state="error")
                st.warning(
                    "These PDFs don't contain selectable text. They may be scanned images."
                )
                return

            st.write("Splitting into chunks")
            chunks = get_text_chunks(pages)

            st.write(f"Embedding {len(chunks)} chunks")
            vector_store = get_vector_store(chunks)
        except Exception as error:
            status.update(label="Processing failed", state="error")
            st.error(f"Something went wrong while processing: {error}")
            return

        workspace().vector_store = vector_store
        workspace().documents = doc_info
        workspace().messages = []
        workspace().unreadable = unreadable
        status.update(label="Documents ready", state="complete", expanded=False)

    # Rerun so the sidebar and main area reflect the newly processed documents.
    st.session_state.just_processed = True
    st.rerun()


def render_welcome():
    st.title("Read less, ask more.")
    st.markdown(
        "Upload a document and ask questions in plain language. "
        "IntelliDocs finds the relevant passages and answers with sources."
    )
    st.space("medium")

    steps = [
        ("Upload", "Add one or more PDFs from the sidebar."),
        ("Process", "Index them with **Process documents**."),
        ("Ask", "Get answers with page references."),
    ]
    for number, (column, (title, text)) in enumerate(zip(st.columns(3), steps), start=1):
        with column.container(border=True, height="stretch", key=f"step-{number}"):
            st.badge(f"Step {number:02d}", color="primary")
            st.markdown(f"#### {title}")
            st.caption(text)


def render_chat_header():
    docs = workspace().documents
    total_pages = sum(doc["pages"] for doc in docs)
    st.title("Your documents")
    st.caption(
        f"{len(docs)} document{'s' if len(docs) != 1 else ''} · "
        f"{total_pages} page{'s' if total_pages != 1 else ''} indexed"
    )


def answer_question(question):
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            # Keep the spinner visible until Gemini sends its first words.
            with st.spinner("Reading your documents…"):
                docs = workspace().vector_store.similarity_search(question)
                stream = stream_answer(question, docs)
                first = next(stream, "")
            answer = st.write_stream(itertools.chain([first], stream))
            sources = collect_sources(docs)
            render_sources(sources)
        except Exception as error:
            if "RESOURCE_EXHAUSTED" in str(error):
                st.error(
                    "Gemini's usage limit for your API key has been reached. "
                    "Free-tier limits reset daily. You can check your usage at "
                    "[ai.dev/rate-limit](https://ai.dev/rate-limit).",
                    icon=":material/hourglass_empty:",
                )
            else:
                st.error("Sorry, I couldn't get an answer from Gemini.", icon=":material/error:")
            with st.expander("Error details"):
                st.code(str(error), language=None)
            return

    # Saved only once the answer is complete, so an interrupted question isn't left unanswered.
    workspace().messages.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "sources": sources},
    ])


def main():
    st.set_page_config(
        page_title="IntelliDocs AI",
        page_icon=":material/auto_stories:",
        layout="centered",
    )
    st.html(CUSTOM_CSS)
    init_state()
    if st.session_state.pop("just_processed", False):
        st.toast("Your documents are ready.")
    render_sidebar()

    ready = workspace().vector_store is not None

    # The chat input is always pinned to the bottom, so read it first to know
    # whether this run will answer a question before drawing the rest of the page.
    question = st.chat_input(
        "Ask a question about your documents…" if ready else "Upload and process a PDF to start",
        disabled=not ready,
    )
    question = question or st.session_state.pop("pending_question", None)

    if not ready:
        render_welcome()
    else:
        render_chat_header()
        if workspace().unreadable:
            st.warning(
                f"No readable text in: {', '.join(workspace().unreadable)}. They may be scanned images.",
                icon=":material/warning:",
            )

        for message in workspace().messages:
            with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
                st.markdown(message["content"])
                render_sources(message.get("sources"))

        if not workspace().messages and not question:
            st.space("small")
            st.pills(
                "Try asking",
                SUGGESTED_QUESTIONS,
                key="suggestion",
                on_change=ask_suggestion,
            )

        if question:
            answer_question(question)


if __name__ == "__main__":
    main()
