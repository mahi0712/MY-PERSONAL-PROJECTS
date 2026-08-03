Readme · MD
# 📚 AI Study Buddy

A multi-subject RAG-based (Retrieval-Augmented Generation) study assistant. Upload your own course material (PDFs), ask questions answered strictly from that material, generate quizzes, and track your weak areas.

## Features

- **Multi-subject support** — separate knowledge base per subject (e.g. Computer Networks, Linear Algebra, RISC-V/COAL)
- **PDF ingestion** — upload notes, slides, or past papers; text is extracted, chunked, and embedded
- **Ask Questions (RAG)** — ask anything about your uploaded material; answers are generated only from your content, not generic internet knowledge
- **Quiz Generator** — auto-generates multiple-choice quizzes on any topic from your uploaded material
- **Weak-Area Tracker** — logs your quiz performance per topic and shows you which topics need the most revision
- **Persistent storage** — your uploaded material and quiz history are saved locally, so you don't need to re-upload every session

## Tech Stack

| Component | Library |
|---|---|
| PDF text extraction | PyMuPDF (`fitz`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector search | FAISS |
| LLM (answers + quiz generation) | Anthropic Claude API |
| UI | Streamlit |
| Storage | JSON + pickle (local files) |

## Project Structure

```
AI STUDY BUDDY/
├── study_buddy.py                 # main application
├── study_buddy_requirements.txt   # dependencies
├── README.md                      # this file
└── study_buddy_data/              # auto-created — stores per-subject indexes & quiz logs
    └── <subject_name>/
        ├── index.faiss
        ├── chunks.pkl
        └── quiz_log.json
```

## Setup

### 1. Install Python
Make sure Python 3.9+ is installed and added to PATH. Verify with:
```bash
python --version
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

Activate it:
- **Windows (PowerShell):** `venv\Scripts\activate`
  - If you get an execution policy error, run this first:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```
- **Windows (Command Prompt):** `venv\Scripts\activate.bat`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Install dependencies
```bash
pip install -r study_buddy_requirements.txt
```

### 4. Set your Anthropic API key
Get a key from [console.anthropic.com](https://console.anthropic.com).

- **Windows (PowerShell):**
  ```powershell
  $env:ANTHROPIC_API_KEY="your-key-here"
  ```
- **Windows (Command Prompt):**
  ```cmd
  set ANTHROPIC_API_KEY=your-key-here
  ```
- **Mac/Linux:**
  ```bash
  export ANTHROPIC_API_KEY="your-key-here"
  ```

### 5. Run the app
```bash
streamlit run study_buddy.py
```

The app will open automatically in your browser at `http://localhost:8501`. If it doesn't, copy the URL shown in the terminal into your browser manually.

## How to Use

1. **Create a subject** from the sidebar (e.g. "Computer Networks")
2. Go to the **Upload Material** tab and upload your PDFs for that subject
3. Go to the **Ask Questions** tab and ask anything about the uploaded content
4. Go to the **Quiz Me** tab, type a topic, and generate a quiz to test yourself
5. Check the **Weak Areas** tab periodically to see which topics need more revision

## Notes

- The Anthropic API is billed per request (a small cost) — it is not free like a chat interface.
- Each subject's data persists locally, so you only need to upload material once per subject.
- To reset a subject's knowledge base, delete its folder inside `study_buddy_data/`.