# 🎓 AI Teaching Assistant (Professor Nova)
### Next-Gen RAG Educational Platform powered by Google Gemini 2.0 & Multimodal Ingestion

An AI-powered Teaching Assistant web application built with a modern glassmorphism UI, **Google Gemini 2.0 / 1.5**, and **Retrieval-Augmented Generation (RAG)**. It ingests course materials (PDFs, Word docs, PowerPoint presentations, YouTube lecture videos, Wikipedia academic topics, and raw notes) to provide deep conceptual clarity, interactive practice quizzes, 3D study flashcards, and student learning analytics.

---

## ✨ Key Features

1. **💬 Tutor Chat & Doubt Resolution (Professor Nova)**
   - Conversational AI tutor designed for pedagogical excellence: intuitive explanations, step-by-step proofs, formulas, and runnable code.
   - **Inline Citations**: Every answer directly references source documents, slides, pages, or video timestamps (e.g. `[Slide 4]`, `[03:25]`).
   - **Voice Input & Text-to-Speech**: Speak doubts directly into your microphone; listen to answers read aloud.
   - **Follow-up Suggestions**: Generates 3 intelligent follow-up questions to promote active learning.

2. **📚 Multimodal Course Materials Hub**
   - **Multi-File Uploads**: Drag-and-drop support for PDF (`.pdf`), Word (`.docx`), PowerPoint (`.pptx`), Markdown, and code files.
   - **YouTube Lecture Ingestion**: Paste any YouTube lecture URL to automatically extract transcripts with minute-by-minute timestamps.
   - **Wikipedia Academic Ingestion**: Pull authoritative foundational notes on any academic topic.
   - **Document Library**: Inspect indexed materials, chunk counts, or delete specific sources.

3. **🎯 Interactive Practice Quiz & Exam Prep**
   - Instant AI-generated multiple-choice questions (MCQs) tailored to your uploaded materials.
   - Configurable question count (3, 5, 10) and difficulty (*Easy*, *Medium*, *Challenging*).
   - Real-time feedback, detailed answer explanations, progress bar, and score tracking with celebratory confetti.

4. **💡 Executive Study Guide & 3D Flashcards**
   - High-yield study sheets summarizing core concepts, definitions, and equations.
   - Interactive 3D flip flashcards for rapid spaced-repetition revision.

5. **📈 Student Progress & Learning Analytics**
   - Tracks total doubts asked, quizzes taken, overall accuracy rate (%), and study streaks.
   - Searchable doubt history log allowing students to review past explanations and re-ask queries.

6. **⚙️ Flexible Gemini Configuration**
   - Select between `gemini-2.0-flash`, `gemini-1.5-flash`, and `gemini-1.5-pro`.
   - Update your API key directly through the UI or `.env`.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ installed
- (Optional) Free Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 2. Setup & Installation
```bash
# Clone the repository
git clone https://github.com/Abhinandan-49/Aiteacherassistence.git
cd Aiteacherassistence

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # On Windows
# source .venv/bin/activate # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Web Application
Double-click `START_ASSISTANT.bat` on Windows, or run:
```bash
python app.py
```
Open your browser to: **[http://localhost:5000](http://localhost:5000)**

---

## 🏗️ Project Architecture

```
AI_teacher_Assistence/
├── app.py                  # Flask web server & REST API endpoints
├── config.py               # Environment configuration & model settings
├── document_loader.py      # Multimodal loaders (PDF, DOCX, PPTX, YouTube, Wiki)
├── text_splitter.py        # Recursive chunking preserving context and metadata
├── vector_store.py         # Semantic search with Gemini text-embedding-004
├── teaching_assistant.py   # Core pedagogy engine & RAG prompt pipeline
├── analytics_tracker.py    # Student progress, doubt history, & quiz stats
├── templates/
│   └── index.html          # Modern glassmorphism single-page application UI
├── START_ASSISTANT.bat     # Windows one-click browser launcher
├── START_WEB_UI.bat        # Windows one-click server starter
└── requirements.txt        # Production dependencies
```

---

## 🛡️ License
MIT License. Created for high-impact educational AI research and student learning.
