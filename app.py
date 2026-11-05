import os
import re
from pathlib import Path
from operator import itemgetter

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Zyro Dynamics | HR Help Desk",
    page_icon="🏢",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
PDF_DIR = APP_DIR / "data" / "hr_policies"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ==================================================
# 2. DOCUMENT FORMATTING
# ==================================================

def format_docs(docs):

    formatted = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1

        formatted.append(
            f"[Source: {source}, Page: {page}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted)


def clean_answer_markdown(answer):
    """Remove HTML artifacts that otherwise appear literally in Streamlit."""
    answer = str(answer or "")
    answer = re.sub(r"(?i)<br\s*/?>", " · ", answer)
    answer = re.sub(r"(?i)</?p\s*>", "\n\n", answer)
    answer = re.sub(r"(?i)</?div\b[^>]*>", "\n", answer)
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


# ==================================================
# 3. LOAD DOCUMENTS AND BUILD RAG PIPELINE
# ==================================================

@st.cache_resource(show_spinner="Building HR knowledge base...")

def load_rag_system(groq_api_key, langchain_api_key=""):

    # Configure API keys
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Enable LangSmith tracing when configured
    if langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = (
            "https://api.smith.langchain.com"
        )
        os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

    # Load HR PDF files
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF documents found in: {PDF_DIR}"
        )

    documents = []

    for pdf_file in pdf_files:

        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_file.name

        documents.extend(pages)

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    # Generate HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Build FAISS vector database
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Configure similarity retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Initialize Groq LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        reasoning_effort="low",
        reasoning_format="hidden",
        max_tokens=512,
        max_retries=2
    )

    # ==================================================
    # 4. LCEL RAG CHAIN
    # ==================================================

    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an HR Help Desk assistant for
Zyro Dynamics Pvt. Ltd.

Answer employee questions using ONLY the provided
HR policy context.

Instructions:

1. Give accurate and concise answers.
2. Preserve important numbers, dates and conditions.
3. Never confuse different leave categories.
4. Specific policy rules take precedence over general rules.
5. Do not invent or assume missing information.
6. Consider all relevant retrieved context.
7. If the context does not support an answer,
   state that the information is unavailable.
8. Include relevant source document and page number.
9. Use clean Markdown only; never output HTML tags such as <br>.
10. Avoid Markdown tables. Use short headings and bullet points instead.
11. Keep answers concise while preserving important conditions, limits and exceptions.

HR POLICY CONTEXT:

{context}
"""
        ),
        ("human", "{question}")
    ])

    answer_chain = (
        answer_prompt
        | llm
        | StrOutputParser()
    )

    # Build LCEL retrieval and context chain
    rag_chain = (
        {
            "question": itemgetter("question"),
            "source_documents": itemgetter("question") | retriever
        }
        | RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda x: format_docs(x["source_documents"])
            )
        )
    )

    # ==================================================
    # 5. ANSWERABILITY GUARDRAIL
    # ==================================================

    guardrail_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an answerability evaluator for
the Zyro Dynamics HR Help Desk.

Determine whether the retrieved context contains enough
evidence to answer the employee's question.

Rules:

1. Read all retrieved context.
2. ANSWERABLE when relevant evidence directly supports
   the requested answer.
3. Combine information from multiple chunks if needed.
4. Ignore unrelated retrieved chunks.
5. Never confuse different leave categories.
6. Do not use external knowledge.
7. OUT_OF_SCOPE only when sufficient evidence
   is genuinely absent.

Examples:

Question: How much Earned Leave can I carry forward?
Context: Earned Leave can be carried forward up to 45 days.
Decision: ANSWERABLE

Question: What is Zoho's leave policy?
Context: Only Zyro Dynamics policies are available.
Decision: OUT_OF_SCOPE

Return exactly one label:

ANSWERABLE
or
OUT_OF_SCOPE
"""
        ),
        (
            "human",
            """Question:
{question}

Retrieved HR Policy Context:
{context}

Decision:"""
        )
    ])

    guardrail_llm = ChatGroq(
        model=LLM_MODEL,
        reasoning_effort="low",
        reasoning_format="hidden",
        max_tokens=512,
        max_retries=2
    )

    guardrail_chain = (
        guardrail_prompt
        | guardrail_llm
        | StrOutputParser()
    )

    return {
        "rag_chain": rag_chain,
        "answer_chain": answer_chain,
        "guardrail_chain": guardrail_chain,
        "pdf_count": len(pdf_files),
        "page_count": len(documents),
        "chunk_count": len(chunks)
    }


# ==================================================
# 6. CHATBOT FUNCTION
# ==================================================

def ask_bot(question, rag_system):

    # Retrieve policy context
    result = rag_system["rag_chain"].invoke({
        "question": question
    })

    # Evaluate answerability
    verdict = rag_system["guardrail_chain"].invoke({
        "question": question,
        "context": result["context"]
    })

    verdict = verdict.strip().upper()

    # Parse guardrail response
    if re.search(r"\bOUT_OF_SCOPE\b", verdict):
        is_answerable = False

    elif re.search(r"\bANSWERABLE\b", verdict):
        is_answerable = True

    else:
        is_answerable = False

    # Refuse unsupported questions
    if not is_answerable:

        return {
            "answer": (
                "I'm sorry, but the provided HR policy documents "
                "do not contain sufficient information to answer "
                "this question. Please contact the HR team "
                "for clarification."
            ),
            "status": "OUT_OF_SCOPE",
            "sources": []
        }

    # Generate grounded answer
    answer = rag_system["answer_chain"].invoke({
        "question": question,
        "context": result["context"]
    })
    answer = clean_answer_markdown(answer)

    # Collect unique source citations
    sources = []
    seen = set()

    for doc in result["source_documents"]:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1

        key = (source, page)

        if key not in seen:

            sources.append({
                "source": source,
                "page": page
            })

            seen.add(key)

    return {
        "answer": answer,
        "status": "ANSWERABLE",
        "sources": sources
    }




# ==================================================
# 7. BAM BEE-INSPIRED EMPLOYEE PORTAL UI
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root { --navy:#173B55; --teal:#087E72; --bg:#F5F7F8; --muted:#64748B; --line:#E3E9ED; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--bg); color:#203343; }
[data-testid="stHeader"] { background:rgba(245,247,248,.94); }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:1.35rem; }
.zyro-brand { display:flex; align-items:center; gap:11px; }
.brand-mark { width:42px;height:42px;border-radius:13px;background:#E5F2EF;color:var(--teal);display:flex;align-items:center;justify-content:center;font-size:21px; }
.brand-name { color:var(--navy);font-weight:800;font-size:17px;line-height:1.2; }
.brand-sub { color:#82909B;font-size:10px;font-weight:700;letter-spacing:1.25px;margin-top:4px; }
.eyebrow { color:var(--teal);font-size:11px;font-weight:800;letter-spacing:1.8px;margin-bottom:10px; }
.hero-title { color:var(--navy);font-size:clamp(32px,4vw,45px);font-weight:800;letter-spacing:-1.8px;line-height:1.12; }
.hero-sub { color:var(--muted);font-size:15px;line-height:1.65;max-width:700px;margin-top:10px; }
.section-title { color:var(--navy);font-weight:800;font-size:19px;letter-spacing:-.4px; }
.section-sub { color:var(--muted);font-size:13px;margin-top:2px;margin-bottom:13px; }
.metric-card { background:#fff;border:1px solid var(--line);border-radius:15px;padding:16px 18px;min-height:100px;box-shadow:0 2px 8px rgba(23,59,85,.025); }
.metric-label { color:var(--muted);font-size:12px;font-weight:600; }
.metric-value { color:var(--navy);font-size:29px;font-weight:800;letter-spacing:-1px;margin-top:5px; }
.service-card { background:#fff;border:1px solid var(--line);border-radius:16px;padding:17px;min-height:155px;box-shadow:0 3px 12px rgba(23,59,85,.035); }
.service-icon { width:39px;height:39px;border-radius:12px;background:#E8F3F0;display:flex;align-items:center;justify-content:center;font-size:19px;margin-bottom:11px; }
.service-title { color:var(--navy);font-size:15px;font-weight:800;margin-bottom:5px; }
.service-desc { color:var(--muted);font-size:12px;line-height:1.5;min-height:38px; }
.online-pill { display:inline-flex;align-items:center;gap:7px;background:#E8F5EF;color:#18794E;border-radius:20px;padding:7px 11px;font-size:11px;font-weight:700; }
.online-dot { width:7px;height:7px;border-radius:50%;background:#22A06B; }
.side-label { color:#82909B;font-size:10px;font-weight:800;letter-spacing:1.3px;text-transform:uppercase;margin:20px 0 8px; }
.side-note { color:#7A8994;font-size:11px;line-height:1.5; }
.footer-note { color:#87939D;font-size:11px;text-align:center;padding:15px 0 8px; }
div.stButton > button { border-radius:10px;border:1px solid #D5E2E3;background:#F8FBFA;color:var(--teal);font-weight:700;min-height:39px; }
div.stButton > button:hover { border-color:var(--teal);background:#EAF4F1;color:var(--navy); }
div[data-testid="stChatInput"] { border:1px solid #DCE5E8;border-radius:15px;background:#fff;box-shadow:0 5px 18px rgba(23,59,85,.06); }
div[data-testid="stChatMessage"] { border:1px solid var(--line);border-radius:15px;background:#fff;padding:12px 15px;margin-bottom:11px; }
div[data-testid="stExpander"] { border:1px solid var(--line);border-radius:12px;background:#fff; }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { color:var(--navy); }
hr { border-color:var(--line); }
@media(max-width:800px) { .hero-title{font-size:32px}.hero-sub{font-size:14px}.service-card{min-height:145px} }

/* ACCESSIBILITY / TEXT CONTRAST FIXES */
div[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    color: #173B55 !important;
}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *,
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] strong,
div[data-testid="stChatMessage"] span {
    color: #173B55 !important;
    opacity: 1 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #526779 !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] b,
section[data-testid="stSidebar"] strong {
    color: #173B55 !important;
}

/* Chat input */
div[data-testid="stChatInput"] {
    background: #FFFFFF !important;
}
div[data-testid="stChatInput"] textarea {
    color: #173B55 !important;
    -webkit-text-fill-color: #173B55 !important;
    caret-color: #087E72 !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
    color: #8795A1 !important;
    -webkit-text-fill-color: #8795A1 !important;
    opacity: 1 !important;
}

/* Ensure normal app text uses the light-theme palette */
.stApp {
    color: #203343;
}


/* FINAL UI POLISH: light chat dock and readable service buttons */

/* Light background for Streamlit's fixed bottom chat area. */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"],
[data-testid="stBottomBlockContainer"] > div,
.stChatFloatingInputContainer,
div[class*="stChatInput"],
section[data-testid="stChatInput"] {
    background: #F5F7F8 !important;
    background-color: #F5F7F8 !important;
}

/* White input surface with readable text. */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div {
    background: #FFFFFF !important;
    border: 1px solid #DCE5E8 !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 14px rgba(23, 59, 85, 0.05) !important;
}

[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: #173B55 !important;
    -webkit-text-fill-color: #173B55 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #8795A1 !important;
    -webkit-text-fill-color: #8795A1 !important;
    opacity: 1 !important;
}

.stChatFloatingInputContainer {
    border-top: 1px solid #E3E9ED !important;
    box-shadow: none !important;
    padding-top: 10px !important;
}

/* Prevent long policy tables from overflowing chat bubbles. */
div[data-testid="stChatMessage"] {
    max-width: 100% !important;
    overflow-wrap: anywhere !important;
}
div[data-testid="stChatMessage"] table {
    width: 100% !important;
    table-layout: fixed !important;
}
div[data-testid="stChatMessage"] th,
div[data-testid="stChatMessage"] td {
    overflow-wrap: anywhere !important;
    white-space: normal !important;
}

/* Keep service-card buttons fully readable. */
div.stButton > button {
    width: 100% !important;
    min-height: 42px !important;
    height: auto !important;
    padding: 8px 7px !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    line-height: 1.25 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
}
div.stButton > button p {
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    line-height: 1.25 !important;
}

</style>

""", unsafe_allow_html=True)

# Read credentials from Streamlit secrets; never hard-code API keys.
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("GROQ_API_KEY is missing. Add it to .streamlit/secrets.toml.")
    st.stop()

LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY", "")

try:
    rag_system = load_rag_system(GROQ_API_KEY, LANGCHAIN_API_KEY)
except Exception as error:
    st.error(f"Unable to initialize the HR knowledge base: {error}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Sidebar: compact employee self-service navigation and system information.
with st.sidebar:
    st.markdown("""
    <div class="zyro-brand">
      <div class="brand-mark">🏢</div>
      <div><div class="brand-name">Zyro Dynamics</div>
      <div class="brand-sub">PEOPLE OPERATIONS</div></div>
    </div><br>
    <div class="online-pill"><span class="online-dot"></span> HR knowledge base online</div>
    <div class="side-label">YOUR WORKPLACE</div>
    <b>🏠 &nbsp; HR Help Desk</b>
    <div class="side-note">Ask questions about company policies</div>
    <div class="side-label">KNOWLEDGE BASE</div>
    """, unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.metric("Documents", rag_system["pdf_count"])
        st.metric("Text chunks", rag_system["chunk_count"])
    with s2:
        st.metric("PDF pages", rag_system["page_count"])
        st.metric("Retriever", "FAISS")
    st.markdown('<div class="side-label">AI ASSISTANT</div>', unsafe_allow_html=True)
    st.caption("Language model")
    st.write("Groq · GPT-OSS 120B")
    st.caption("Embedding model")
    st.write("MiniLM-L6-v2")
    st.caption("Search engine")
    st.write("FAISS similarity search")
    if LANGCHAIN_API_KEY:
        st.success("LangSmith tracing enabled")
    else:
        st.caption("LangSmith tracing disabled")
    st.divider()
    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()
    st.markdown("<div class='side-note'>Answers are generated from available Zyro Dynamics HR policy documents. Refer to cited policies for official guidance.</div>", unsafe_allow_html=True)

# Landing header and compact knowledge-base metrics.
st.markdown('<div class="eyebrow">EMPLOYEE SELF-SERVICE PORTAL</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">How can we help?</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Your workplace questions, answered intelligently. Explore HR policies, understand your benefits, and find the information you need — all in one place.</div>', unsafe_allow_html=True)
st.write("")
metric_cols = st.columns(3)
for col, (label, value, symbol) in zip(metric_cols, [
    ("HR policy documents", rag_system["pdf_count"], "📚"),
    ("PDF pages indexed", rag_system["page_count"], "📄"),
    ("Knowledge chunks", rag_system["chunk_count"], "✦"),
]):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{symbol}&nbsp;&nbsp;{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.write("")
st.markdown('<div class="section-title">Explore HR services</div><div class="section-sub">Choose a topic to get started, or ask your own question.</div>', unsafe_allow_html=True)

services = [
    ("🌿", "Leave & holidays", "Leave entitlements, earned leave, holidays and carry-forward rules.", "Ask about leave", "How much Earned Leave can I carry forward to next year?"),
    ("💳", "Salary & benefits", "Payroll dates, compensation, insurance and employee benefits.", "Ask about salary", "What date does my salary get credited every month?"),
    ("💻", "Workplace policies", "Work from home, code of conduct and workplace guidelines.", "Ask about workplace", "What is the work from home policy?"),
    ("🤝", "People & culture", "Onboarding, performance reviews and employee guidelines.", "Ask about people", "What is the employee onboarding process?"),
]
cols = st.columns(4)
for i, (col, service) in enumerate(zip(cols, services)):
    icon, title, desc, button, question = service
    with col:
        st.markdown(f'<div class="service-card"><div class="service-icon">{icon}</div><div class="service-title">{title}</div><div class="service-desc">{desc}</div></div>', unsafe_allow_html=True)
        if st.button(button, key=f"service_{i}", use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()

# Chat section: keep the existing RAG function and answerability guardrail.
st.write("")
st.divider()
st.markdown('<div class="section-title">Your HR AI assistant</div><div class="section-sub">Ask anything about Zyro Dynamics. Answers are grounded in company HR documents.</div>', unsafe_allow_html=True)

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("Hello! Welcome to the Zyro Dynamics HR Help Desk. I can help you understand company policies, leave, salary, benefits and workplace guidelines. What would you like to know?")

for message in st.session_state.messages:
    avatar = "🧑‍💼" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📄 View policy sources ({len(message['sources'])})"):
                for source in message["sources"]:
                    st.markdown(f"- **{source['source']}** · Page {source['page']}")

typed_question = st.chat_input("Ask your HR question here...")
user_question = st.session_state.pending_question or typed_question

if user_question:
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": user_question, "sources": []})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_question)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Checking the HR knowledge base..."):
            try:
                result = ask_bot(user_question, rag_system)
                answer = result["answer"]
                sources = result["sources"]
                st.markdown(answer)
                if result["status"] == "ANSWERABLE":
                    if sources:
                        with st.expander(f"📄 View policy sources ({len(sources)})"):
                            for source in sources:
                                st.markdown(f"- **{source['source']}** · Page {source['page']}")
                else:
                    st.info("This question is outside the available HR policy information.")
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as error:
                message = str(error)
                if "429" in message or "rate_limit" in message.lower():
                    message = "The AI service is temporarily rate-limited. Please wait a moment and try again."
                else:
                    message = f"Unable to process your question: {message}"
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message, "sources": []})

st.write("")
st.divider()
st.markdown('<div class="footer-note"><strong style="color:#173B55">ZYRO DYNAMICS</strong> &nbsp; · &nbsp; PEOPLE OPERATIONS<br>AI-powered, document-grounded employee assistance.</div>', unsafe_allow_html=True)
