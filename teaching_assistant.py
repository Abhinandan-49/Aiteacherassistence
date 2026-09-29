"""
Core AI Teaching Assistant Engine
Powered by Google Gemini 2.0 / 1.5, RAG retrieval, and intelligent pedagogy
"""
import os
import re
import json
from typing import List, Dict, Any, Optional

from config import Config
from document_loader import DocumentLoader, Document
from text_splitter import TextChunker
from vector_store import VectorStoreManager
from analytics_tracker import AnalyticsTracker

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class AITeachingAssistant:
    """Intelligent AI Teaching Assistant with Gemini RAG pipeline"""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = (api_key or Config.GEMINI_API_KEY).strip()
        self.model_name = model_name or Config.DEFAULT_MODEL
        self.vector_store_manager = VectorStoreManager(api_key=self.api_key)
        self.chunker = TextChunker()
        self.analytics = AnalyticsTracker()

    def set_api_key(self, api_key: str):
        """Update API key across all sub-components"""
        self.api_key = api_key.strip()
        Config.update_api_key(self.api_key)
        self.vector_store_manager.set_api_key(self.api_key)

    def set_model(self, model_name: str):
        """Update active Gemini model"""
        self.model_name = model_name.strip()

    def _get_client(self):
        """Create GenAI client"""
        if not self.api_key or genai is None:
            return None
        try:
            return genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"Failed to initialize Gemini Client: {e}")
            return None

    def _generate_with_fallback(self, client, contents, config=None):
        """Generate content with model cascade fallbacks (handles 429 quota exhaustion and model availability)"""
        models_to_try = [
            self.model_name,
            'gemini-flash-lite-latest',
            'gemini-flash-latest',
            'gemini-2.5-flash'
        ]
        seen = set()
        unique_models = []
        for m in models_to_try:
            if m and m not in seen:
                seen.add(m)
                unique_models.append(m)

        last_error = None
        for model in unique_models:
            try:
                if config:
                    return client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config
                    )
                else:
                    return client.models.generate_content(
                        model=model,
                        contents=contents
                    )
            except Exception as e:
                err_str = str(e)
                print(f"[Model Cascade] Model {model} failed ({type(e).__name__}: {err_str[:60]}), trying fallback...")
                last_error = e
                continue
        raise last_error or RuntimeError("All Gemini model cascades failed.")

    def ingest_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Ingest multiple files (PDF, DOCX, PPTX, TXT, etc.) into the knowledge base"""
        all_docs: List[Document] = []
        errors = []

        for path in file_paths:
            try:
                docs = DocumentLoader.load_any_file(path)
                all_docs.extend(docs)
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {str(e)}")

        if not all_docs:
            return {
                "success": False,
                "message": "No text content could be extracted from uploaded files.",
                "errors": errors
            }

        chunks = self.chunker.split_documents(all_docs)
        self.vector_store_manager.add_documents(chunks)

        return {
            "success": True,
            "message": f"Successfully indexed {len(file_paths)} file(s).",
            "documents_loaded": len(all_docs),
            "chunks_created": len(chunks),
            "errors": errors
        }

    def ingest_youtube(self, url: str) -> Dict[str, Any]:
        """Transcribe and index YouTube video lecture"""
        try:
            docs = DocumentLoader.load_from_youtube(url)
            if not docs:
                return {"success": False, "message": "No transcript available for this video."}

            chunks = self.chunker.split_documents(docs)
            self.vector_store_manager.add_documents(chunks)

            title = docs[0].metadata.get("source", "YouTube Lecture")
            return {
                "success": True,
                "message": f"Successfully transcribed and indexed: {title}",
                "title": title,
                "chunks": len(chunks)
            }
        except Exception as e:
            return {"success": False, "message": str(e)}

    def ingest_wikipedia(self, query: str) -> Dict[str, Any]:
        """Fetch and index Wikipedia topic"""
        try:
            docs = DocumentLoader.load_from_wikipedia(query)
            chunks = self.chunker.split_documents(docs)
            self.vector_store_manager.add_documents(chunks)
            return {
                "success": True,
                "message": f"Indexed Wikipedia article for '{query}'",
                "chunks": len(chunks)
            }
        except Exception as e:
            return {"success": False, "message": str(e)}

    def ingest_text_note(self, title: str, content: str) -> Dict[str, Any]:
        """Index direct text notes or lecture transcription"""
        try:
            if not content.strip():
                return {"success": False, "message": "Content cannot be empty."}

            doc = Document(
                page_content=content.strip(),
                metadata={"source": title.strip() or "Custom Lecture Note", "type": "note"}
            )
            chunks = self.chunker.split_documents([doc])
            self.vector_store_manager.add_documents(chunks)
            return {
                "success": True,
                "message": f"Indexed note '{title}' ({len(chunks)} chunks)",
                "chunks": len(chunks)
            }
        except Exception as e:
            return {"success": False, "message": str(e)}

    def ask(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None, student_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the AI Teaching Assistant with context retrieval, citation tracking,
        and student personalization
        """
        question = question.strip()
        if not question:
            return {"error": "Question cannot be empty"}

        # Step 1: Retrieve relevant context
        search_results = self.vector_store_manager.similarity_search_with_score(question, k=Config.TOP_K_RESULTS)

        context_blocks = []
        sources_list = []
        seen_sources = set()

        for doc, score in search_results:
            src = doc.metadata.get("source", "Course Materials")
            page = doc.metadata.get("page")
            slide = doc.metadata.get("slide")
            timestamp = doc.metadata.get("timestamp")
            url = doc.metadata.get("url")

            ref_tag = src
            if page:
                ref_tag += f" (Page {page})"
            elif slide:
                ref_tag += f" (Slide {slide})"
            elif timestamp:
                ref_tag += f" [Timestamp {timestamp}]"

            context_blocks.append(f"--- Document Source: {ref_tag} ---\n{doc.page_content}")

            if ref_tag not in seen_sources:
                seen_sources.add(ref_tag)
                sources_list.append({
                    "title": ref_tag,
                    "url": url,
                    "type": doc.metadata.get("type", "document"),
                    "relevance_score": round(score, 2)
                })

        context_str = "\n\n".join(context_blocks) if context_blocks else "No specific course materials found. Answer using general academic knowledge."

        # Step 2: System prompt for pedagogical excellence & strict course grounding
        system_instruction = (
            "You are Professor Nova, an exceptional, highly encouraging, and pedagogy-focused AI Teaching Assistant.\n\n"
            "CRITICAL INSTRUCTIONS FOR COURSE GROUNDING:\n"
            "1. You MUST ALWAYS ground and formulate your answer according to the provided RELEVANT COURSE MATERIALS CONTEXT below.\n"
            "2. If the student asks about a concept, formula, problem, or topic covered in their uploaded materials (such as uploaded PDFs, lecture slides, or notes), answer directly based on those materials.\n"
            "3. Cite the exact sources, pages, or timestamps throughout your explanation using tags like [Source: filename (Page X)] or [Slide X] or [03:25].\n"
            "4. Explain with deep clarity: state the core principle first, followed by a step-by-step breakdown with intuitive explanations, formulas, or code examples where helpful.\n"
            "5. If the provided course materials do not contain the answer, state what the materials say first, then provide supplementary academic guidance.\n"
            "6. At the very end of your response, strictly output 3 concise follow-up questions formatted exactly as:\n"
            "[FOLLOW_UP_QUESTIONS]\n"
            "- Question 1\n"
            "- Question 2\n"
            "- Question 3"
        )

        # Personalization Profile Context
        profile_context = ""
        if student_profile and isinstance(student_profile, dict):
            s_name = student_profile.get("name", "Student")
            s_major = student_profile.get("major", "General Studies")
            s_level = student_profile.get("academic_level", "Undergraduate")
            s_style = student_profile.get("learning_style", "Intuitive & Practical")
            s_goal = student_profile.get("goal", "Exam Preparation")
            profile_context = (
                f"STUDENT PERSONALIZATION PROFILE:\n"
                f"- Name: {s_name}\n"
                f"- Major / Field of Study: {s_major}\n"
                f"- Academic Level: {s_level}\n"
                f"- Preferred Learning Style: {s_style}\n"
                f"- Primary Target Goal: {s_goal}\n\n"
                f"PEDAGOGICAL INSTRUCTION:\n"
                f"Address {s_name} by name in a warm faculty mentor tone. Connect concepts and analogies to their field ({s_major}) where appropriate, and tailor your explanation style to match {s_style}.\n\n"
            )

        history_context = ""
        if chat_history and isinstance(chat_history, list):
            turns = []
            for msg in chat_history[-6:]:
                role = "Student" if msg.get("role") in ["user", "student"] else "Professor Nova"
                content = (msg.get("content") or msg.get("text") or "").strip()
                if content:
                    turns.append(f"{role}: {content[:400]}")
            if turns:
                history_context = "PREVIOUS CONVERSATION CONTEXT:\n" + "\n".join(turns) + "\n\n"

        user_prompt = (
            f"{profile_context}"
            f"RELEVANT COURSE MATERIALS CONTEXT:\n{context_str}\n\n"
            f"{history_context}"
            f"STUDENT QUERY:\n{question}\n\n"
            "Please provide a complete, clear, and comprehensive teaching explanation."
        )

        client = self._get_client()

        if not client:
            # Fallback response when API key is not yet set
            sample_answer = (
                f"### Welcome to AI Teaching Assistant!\n\n"
                f"I received your question: **\"{question}\"**\n\n"
                f"> **Gemini API Key Required:** To receive real-time neural explanations and deep insights, "
                f"please enter your **Google Gemini API Key** in the **Settings** tab (or set `GEMINI_API_KEY` in your `.env` file).\n\n"
                f"#### Matched Materials Preview:\n"
                f"Here are relevant excerpts found in your course knowledge base:\n\n"
            )
            for idx, doc in enumerate(search_results[:2]):
                src = doc[0].metadata.get("source", "Document")
                sample_answer += f"**Excerpt {idx+1} ({src}):**\n> {doc[0].page_content[:250]}...\n\n"

            sample_answer += "\nOnce your API key is configured, I will synthesize, solve equations, write code, and answer any doubt in real time!"

            follow_ups = [
                "How do I obtain a free Google Gemini API key?",
                "How do I upload custom lecture PDFs and notes?",
                "Can you generate an interactive quiz from these notes?"
            ]

            source_titles = [s["title"] for s in sources_list]
            self.analytics.log_query(question, sample_answer, source_titles)

            return {
                "answer": sample_answer,
                "sources": sources_list,
                "follow_ups": follow_ups,
                "model": "Setup Mode"
            }

        try:
            # Call Gemini with proper system_instruction and model cascade fallback
            response = self._generate_with_fallback(
                client=client,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=Config.TEMPERATURE,
                    max_output_tokens=2500
                )
            )

            raw_text = response.text or ""

            # Extract follow-up questions
            follow_ups = []
            answer_text = raw_text
            if "[FOLLOW_UP_QUESTIONS]" in raw_text:
                parts = raw_text.split("[FOLLOW_UP_QUESTIONS]")
                answer_text = parts[0].strip()
                follow_up_lines = parts[1].strip().split("\n")
                for line in follow_up_lines:
                    cleaned = re.sub(r'^[-*0-9.)\s]+', '', line).strip()
                    if cleaned:
                        follow_ups.append(cleaned)

            if not follow_ups:
                follow_ups = [
                    "Can you explain this with a real-world example?",
                    "What are the most common exam questions on this topic?",
                    "How does this connect to the previous chapter?"
                ]

            # Log to student analytics
            source_titles = [s["title"] for s in sources_list]
            self.analytics.log_query(question, answer_text, source_titles)

            return {
                "answer": answer_text,
                "sources": sources_list,
                "follow_ups": follow_ups[:3],
                "model": self.model_name
            }

        except Exception as e:
            return {
                "error": f"Gemini API Error: {str(e)}",
                "answer": f"⚠️ An error occurred while communicating with Google Gemini: {str(e)}\n\nPlease verify your API key in Settings.",
                "sources": sources_list,
                "follow_ups": []
            }

    def generate_quiz(self, topic: Optional[str] = None, num_questions: int = 5, difficulty: str = "Medium") -> Dict[str, Any]:
        """Generate structured interactive quiz questions from course materials"""
        client = self._get_client()
        if not client:
            return {
                "success": False,
                "error": "Google Gemini API Key is required to generate custom quizzes. Please enter your key in Settings."
            }

        # Gather context
        query = topic or "key course concepts, principles, formulas, definitions, algorithms"
        docs = self.vector_store_manager.similarity_search(query, k=6)
        context_text = "\n\n".join([d.page_content for d in docs]) if docs else "General foundational topics."

        prompt = (
            f"You are an expert university examiner. Generate an interactive practice quiz with {num_questions} "
            f"multiple-choice questions at '{difficulty}' difficulty based on the following course materials.\n\n"
            f"COURSE MATERIALS CONTEXT:\n{context_text}\n\n"
            f"Strict Output Format:\n"
            f"Return ONLY valid JSON (no markdown formatting, no code fences, no extra text) with this exact schema:\n"
            f"[\n"
            f"  {{\n"
            f"    \"id\": 1,\n"
            f"    \"question\": \"Question text here?\",\n"
            f"    \"options\": [\"Option A\", \"Option B\", \"Option C\", \"Option D\"],\n"
            f"    \"correct_index\": 0,\n"
            f"    \"explanation\": \"Detailed explanation of why this answer is correct.\",\n"
            f"    \"difficulty\": \"{difficulty}\",\n"
            f"    \"topic\": \"Specific subtopic\"\n"
            f"  }}\n"
            f"]"
        )

        try:
            response = self._generate_with_fallback(
                client=client,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json")
            )

            text = response.text.strip()
            # Clean up potential markdown formatting if any
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
            if match:
                text = match.group(0)

            questions = json.loads(text)
            cleaned_questions = []
            for idx, q in enumerate(questions):
                opts = q.get("options", [])
                ci = q.get("correct_index", 0)
                if isinstance(ci, str):
                    ci_str = ci.strip().upper()
                    if ci_str in ['A', 'B', 'C', 'D']:
                        ci = ord(ci_str) - ord('A')
                    elif ci_str.isdigit():
                        ci = int(ci_str)
                    else:
                        ci = 0
                elif not isinstance(ci, int):
                    ci = 0
                
                # Check for 1-based indexing
                if ci == len(opts) and len(opts) > 0:
                    ci = len(opts) - 1
                if ci < 0 or (len(opts) > 0 and ci >= len(opts)):
                    ci = 0

                q["id"] = idx + 1
                q["correct_index"] = ci
                cleaned_questions.append(q)

            return {
                "success": True,
                "topic": topic or "Course Review",
                "difficulty": difficulty,
                "questions": cleaned_questions
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate quiz: {str(e)}"
            }

    def generate_summary(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """Generate high-yield executive summary and interactive revision flashcards"""
        client = self._get_client()
        if not client:
            return {
                "success": False,
                "error": "Google Gemini API Key is required. Please configure your key in Settings."
            }

        query = topic or "core summary, key principles, formulas, definitions, takeaways"
        docs = self.vector_store_manager.similarity_search(query, k=6)
        context_text = "\n\n".join([d.page_content for d in docs]) if docs else "Course knowledge base."

        prompt = (
            f"You are a master academic tutor. Based on the following course materials, generate an Executive Study Guide "
            f"and 6 interactive revision flashcards.\n\n"
            f"COURSE CONTEXT:\n{context_text}\n\n"
            f"Strict Output Format:\n"
            f"Return ONLY valid JSON matching this schema:\n"
            f"{{\n"
            f"  \"topic\": \"{topic or 'Course Mastery'}\",\n"
            f"  \"executive_summary\": \"Markdown formatted 3-4 paragraph overview with key takeaways and formulas\",\n"
            f"  \"key_concepts\": [\"Concept 1 description\", \"Concept 2 description\", \"Concept 3 description\"],\n"
            f"  \"flashcards\": [\n"
            f"    {{\"front\": \"Term or Question\", \"back\": \"Definition, formula, or concise explanation\"}}\n"
            f"  ]\n"
            f"}}"
        )

        try:
            response = self._generate_with_fallback(
                client=client,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3, response_mime_type="application/json")
            )

            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                text = match.group(0)

            data = json.loads(text)
            return {
                "success": True,
                **data
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate study summary: {str(e)}"
            }
