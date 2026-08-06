"""
AI Study Buddy (Advanced)
---------------------------
A multi-subject RAG-based study assistant.

Features:
  - Upload PDFs per subject (notes, slides, past papers)
  - Ask questions -> answered ONLY from your uploaded material (RAG)
  - Auto-generate quizzes (MCQs) from any topic/chapter
  - Weak-area tracker: logs which quiz questions you get wrong,
    and tells you which topics to revise

Run with:
    streamlit run study_buddy.py

Requires an ANTHROPIC_API_KEY environment variable set, e.g.:
    export ANTHROPIC_API_KEY="your-key-here"
"""

import json
import os
import re
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_DIR = Path("study_buddy_data")
DATA_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150
TOP_K = 4                # how many chunks to retrieve per question
GROQ_MODEL = "llama-3.3-70b-versatile"  # free-tier Groq model

def get_api_key():
    """Get the Groq API key from Streamlit secrets (when deployed on
    Streamlit Cloud) or from an environment variable (when running locally)."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("GROQ_API_KEY")


api_key = get_api_key()
if not api_key:
    st.error(
        "No Groq API key found. Set GROQ_API_KEY as an environment "
        "variable locally, or add it under Settings -> Secrets on Streamlit Cloud. "
        "Get a free key at https://console.groq.com/keys"
    )
    st.stop()

client = Groq(api_key=api_key)


# ---------------------------------------------------------------------
# EMBEDDING MODEL (cached)
# ---------------------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------
# PDF -> TEXT -> CHUNKS
# ---------------------------------------------------------------------
def extract_text_from_pdf(file) -> str:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text = re.sub(r"\s+", " ", text).strip()
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 30]


# ---------------------------------------------------------------------
# SUBJECT STORAGE (each subject = its own FAISS index + chunk list)
# ---------------------------------------------------------------------
def subject_dir(subject: str) -> Path:
    d = DATA_DIR / subject
    d.mkdir(exist_ok=True)
    return d


def save_subject_index(subject: str, index, chunks: list):
    d = subject_dir(subject)
    faiss.write_index(index, str(d / "index.faiss"))
    with open(d / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_subject_index(subject: str):
    d = subject_dir(subject)
    index_path = d / "index.faiss"
    chunks_path = d / "chunks.pkl"
    if not index_path.exists() or not chunks_path.exists():
        return None, []
    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def list_subjects():
    return [d.name for d in DATA_DIR.iterdir() if d.is_dir()]


def add_document_to_subject(subject: str, text: str, embedder):
    new_chunks = chunk_text(text)
    if not new_chunks:
        return 0

    embeddings = embedder.encode(new_chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    index, existing_chunks = load_subject_index(subject)
    if index is None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    all_chunks = existing_chunks + new_chunks
    save_subject_index(subject, index, all_chunks)
    return len(new_chunks)


def retrieve_relevant_chunks(subject: str, query: str, embedder, k=TOP_K):
    index, chunks = load_subject_index(subject)
    if index is None or index.ntotal == 0:
        return []
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, min(k, index.ntotal))
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# ---------------------------------------------------------------------
# CLAUDE CALLS
# ---------------------------------------------------------------------
def ask_claude(question: str, context_chunks: list) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful study tutor. Answer the student's question using "
        "ONLY the provided course material context. If the context doesn't "
        "contain enough information to answer, say so clearly instead of "
        "guessing. Keep answers clear and exam-focused."
    )
    user_message = f"Course material context:\n{context}\n\nQuestion: {question}"

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=800,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return ""


def generate_quiz(topic: str, context_chunks: list, num_questions=5) -> list:
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a quiz generator for a university student. Based ONLY on "
        "the provided course material, generate multiple-choice questions. "
        "Respond ONLY with valid JSON, no preamble, no markdown fences. "
        "Format: a JSON list of objects, each with keys: "
        "'question', 'options' (list of 4 strings), 'correct_index' (0-3), "
        "'topic' (a short tag for what sub-topic this tests)."
    )
    user_message = (
        f"Course material context:\n{context}\n\n"
        f"Generate {num_questions} multiple-choice questions about: {topic}"
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return []

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        st.error("Couldn't parse quiz output. Try generating again.")
        return []


# ---------------------------------------------------------------------
# WEAK-AREA TRACKER
# ---------------------------------------------------------------------
def log_quiz_result(subject: str, topic: str, correct: bool):
    log_path = DATA_DIR / subject / "quiz_log.json"
    log = []
    if log_path.exists():
        with open(log_path, "r") as f:
            log = json.load(f)
    log.append({"topic": topic, "correct": correct})
    with open(log_path, "w") as f:
        json.dump(log, f)


def get_weak_areas(subject: str, top_n=5):
    log_path = DATA_DIR / subject / "quiz_log.json"
    if not log_path.exists():
        return []
    with open(log_path, "r") as f:
        log = json.load(f)

    topic_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for entry in log:
        topic_stats[entry["topic"]]["total"] += 1
        if entry["correct"]:
            topic_stats[entry["topic"]]["correct"] += 1

    ranked = []
    for topic, stats in topic_stats.items():
        accuracy = stats["correct"] / stats["total"]
        ranked.append((topic, accuracy, stats["total"]))

    ranked.sort(key=lambda x: x[1])  # lowest accuracy first
    return ranked[:top_n]


# ---------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Study Buddy", page_icon="📚", layout="wide")
    st.title("📚 AI Study Buddy")

    embedder = load_embedder()

    # ---- Sidebar: subject management ----
    st.sidebar.header("Subjects")
    existing_subjects = list_subjects()
    new_subject = st.sidebar.text_input("Create new subject (e.g. Computer Networks)")
    if st.sidebar.button("Add Subject") and new_subject.strip():
        subject_dir(new_subject.strip())
        st.sidebar.success(f"Subject '{new_subject}' created.")
        st.rerun()

    if not existing_subjects:
        st.info("Create a subject from the sidebar to get started.")
        return

    selected_subject = st.sidebar.selectbox("Select Subject", existing_subjects)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📤 Upload Material", "💬 Ask Questions", "📝 Quiz Me", "📊 Weak Areas"]
    )

    # ---- Tab 1: Upload ----
    with tab1:
        st.subheader(f"Upload material for: {selected_subject}")
        uploaded_files = st.file_uploader(
            "Upload PDFs (notes, slides, past papers)", type=["pdf"], accept_multiple_files=True
        )
        if uploaded_files and st.button("Process & Add to Knowledge Base"):
            total_chunks = 0
            for f in uploaded_files:
                with st.spinner(f"Processing {f.name}..."):
                    text = extract_text_from_pdf(f)
                    count = add_document_to_subject(selected_subject, text, embedder)
                    total_chunks += count
            st.success(f"Added {total_chunks} chunks to '{selected_subject}' knowledge base.")

    # ---- Tab 2: Ask Questions (RAG) ----
    with tab2:
        st.subheader(f"Ask about: {selected_subject}")
        question = st.text_input("Your question")
        if st.button("Ask") and question.strip():
            with st.spinner("Thinking..."):
                chunks = retrieve_relevant_chunks(selected_subject, question, embedder)
                if not chunks:
                    st.warning("No material found for this subject yet. Upload some PDFs first.")
                else:
                    answer = ask_claude(question, chunks)
                    st.markdown("**Answer:**")
                    st.write(answer)
                    with st.expander("Source chunks used"):
                        for i, c in enumerate(chunks, 1):
                            st.caption(f"Chunk {i}")
                            st.text(c[:400] + "...")

    # ---- Tab 3: Quiz ----
    with tab3:
        st.subheader(f"Quiz yourself: {selected_subject}")
        quiz_topic = st.text_input("Topic to be quizzed on (e.g. 'TCP handshake')")
        num_q = st.slider("Number of questions", 3, 10, 5)

        if st.button("Generate Quiz") and quiz_topic.strip():
            with st.spinner("Generating quiz..."):
                chunks = retrieve_relevant_chunks(selected_subject, quiz_topic, embedder, k=6)
                if not chunks:
                    st.warning("No material found for this subject yet. Upload some PDFs first.")
                else:
                    quiz = generate_quiz(quiz_topic, chunks, num_q)
                    st.session_state["current_quiz"] = quiz
                    st.session_state["quiz_subject"] = selected_subject
                    st.session_state["quiz_answers"] = {}

        if "current_quiz" in st.session_state and st.session_state["current_quiz"]:
            quiz = st.session_state["current_quiz"]
            for idx, q in enumerate(quiz):
                st.markdown(f"**Q{idx+1}. {q['question']}**")
                choice = st.radio(
                    "Select an answer:",
                    q["options"],
                    key=f"quiz_{idx}",
                    index=None,
                )
                st.session_state["quiz_answers"][idx] = choice

            if st.button("Submit Quiz"):
                score = 0
                for idx, q in enumerate(quiz):
                    selected = st.session_state["quiz_answers"].get(idx)
                    correct_answer = q["options"][q["correct_index"]]
                    is_correct = selected == correct_answer
                    if is_correct:
                        score += 1
                    log_quiz_result(selected_subject, q.get("topic", quiz_topic), is_correct)

                    if is_correct:
                        st.success(f"Q{idx+1}: Correct!")
                    else:
                        st.error(f"Q{idx+1}: Wrong. Correct answer: {correct_answer}")

                st.info(f"Final Score: {score}/{len(quiz)}")
                del st.session_state["current_quiz"]

    # ---- Tab 4: Weak Areas ----
    with tab4:
        st.subheader(f"Weak areas: {selected_subject}")
        weak_areas = get_weak_areas(selected_subject)
        if not weak_areas:
            st.write("No quiz data yet. Take some quizzes first!")
        else:
            for topic, accuracy, total in weak_areas:
                st.write(f"**{topic}** — {accuracy*100:.0f}% accuracy ({total} attempts)")
                st.progress(accuracy)
            st.info("Focus your revision on the topics with lowest accuracy above.")


if __name__ == "__main__":
    main()
