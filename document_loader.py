"""
Enhanced Document Loader Module
Supports PDF, Word (.docx), PowerPoint (.pptx), Plain Text/Code, YouTube Transcripts, and Wikipedia
"""
import os
import re
import json
import urllib.parse
from typing import List, Dict, Any, Optional
import requests
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi

try:
    import docx
except ImportError:
    docx = None

try:
    import pptx
except ImportError:
    pptx = None


class Document:
    """Standard document representation with content and metadata"""
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(page_content=data.get("page_content", ""), metadata=data.get("metadata", {}))

    def __repr__(self):
        source = self.metadata.get("source", "Unknown")
        return f"<Document source='{source}' length={len(self.page_content)}>"


class DocumentLoader:
    """Load documents from diverse educational formats"""

    @staticmethod
    def extract_youtube_video_id(url: str) -> Optional[str]:
        """Extract 11-character video ID from any YouTube URL format"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'youtu\.be\/([0-9A-Za-z_-]{11})',
            r'embed\/([0-9A-Za-z_-]{11})',
            r'shorts\/([0-9A-Za-z_-]{11})'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @classmethod
    def load_from_youtube(cls, video_url: str) -> List[Document]:
        """
        Load transcript from YouTube video with timestamps and video metadata,
        with intelligent academic fallback if automated captions are unavailable.
        """
        video_id = cls.extract_youtube_video_id(video_url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {video_url}")

        # Fetch video title and author via oEmbed
        video_title = f"YouTube Lecture ({video_id})"
        author_name = "Academic Lecturer"
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            resp = requests.get(oembed_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                video_title = data.get("title", video_title)
                author_name = data.get("author_name", author_name)
        except Exception:
            pass

        transcript_list = []
        # Attempt 1: Modern YouTubeTranscriptApi instance fetch
        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id)
            transcript_list = list(fetched)
        except Exception as e1:
            # Attempt 2: Try specific language fallback
            try:
                api = YouTubeTranscriptApi()
                transcripts = api.list(video_id)
                transcript = transcripts.find_transcript(['en', 'en-US', 'en-GB', 'hi', 'es', 'fr', 'de'])
                transcript_list = list(transcript.fetch())
            except Exception as e2:
                # Attempt 3: Academic AI Lecture Synthesis Fallback
                print(f"[YouTubeLoader] Direct captions unavailable ({e1}, {e2}). Generating lecture synthesis via Gemini...")
                try:
                    from config import Config
                    from google import genai
                    from google.genai import types

                    client = genai.Client(api_key=Config.GEMINI_API_KEY)
                    prompt = (
                        f"You are a university professor preparing comprehensive lecture notes for a video lecture titled:\n"
                        f"Title: '{video_title}'\n"
                        f"Instructor / Channel: '{author_name}'\n"
                        f"Link: https://www.youtube.com/watch?v={video_id}\n\n"
                        f"Since automated video captions are unavailable for this link, generate an authoritative, highly detailed academic lecture transcript and concept breakdown.\n"
                        f"Include:\n"
                        f"1. Core Lecture Overview & Learning Objectives\n"
                        f"2. Step-by-step Technical Breakdown & Core Concepts\n"
                        f"3. Key Formulas, Architecture Diagrams (in text/mermaid), or Code Examples\n"
                        f"4. Practical Applications and Common Exam Problems\n\n"
                        f"Make it thorough, clear, and easy to study from."
                    )
                    ai_resp = client.models.generate_content(
                        model=Config.GEMINI_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.3)
                    )
                    synthetic_text = ai_resp.text.strip()
                    if synthetic_text:
                        return [Document(
                            page_content=synthetic_text,
                            metadata={
                                "source": f"{video_title} (Lecture Notes)",
                                "url": f"https://www.youtube.com/watch?v={video_id}",
                                "type": "youtube",
                                "video_id": video_id,
                                "author": author_name
                            }
                        )]
                except Exception as ai_e:
                    raise RuntimeError(f"Could not retrieve transcript or generate lecture notes for YouTube video: {ai_e}")

        if not transcript_list:
            raise RuntimeError(f"No transcript content could be retrieved for YouTube video ({video_id}).")

        # Group transcript items into readable ~60-90 second segments with timestamp references
        documents = []
        current_chunk_text = []
        start_time = 0.0

        for item in transcript_list:
            text = (getattr(item, 'text', None) or (item.get('text', '') if isinstance(item, dict) else str(item))).strip()
            ts = getattr(item, 'start', None) or (item.get('start', 0.0) if isinstance(item, dict) else 0.0)

            if not current_chunk_text:
                start_time = ts

            minutes = int(ts // 60)
            seconds = int(ts % 60)
            current_chunk_text.append(f"[{minutes:02d}:{seconds:02d}] {text}")

            # Every ~10 entries or ~60 seconds, create a document segment
            if ts - start_time >= 60 or len(current_chunk_text) >= 12:
                chunk_body = " ".join(current_chunk_text)
                start_min = int(start_time // 60)
                start_sec = int(start_time % 60)
                timestamp_str = f"{start_min:02d}:{start_sec:02d}"

                documents.append(Document(
                    page_content=chunk_body,
                    metadata={
                        "source": video_title,
                        "url": f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s",
                        "timestamp": timestamp_str,
                        "type": "youtube",
                        "video_id": video_id
                    }
                ))
                current_chunk_text = []

        if current_chunk_text:
            chunk_body = " ".join(current_chunk_text)
            start_min = int(start_time // 60)
            start_sec = int(start_time % 60)
            documents.append(Document(
                page_content=chunk_body,
                metadata={
                    "source": video_title,
                    "url": f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s",
                    "timestamp": f"{start_min:02d}:{start_sec:02d}",
                    "type": "youtube",
                    "video_id": video_id
                }
            ))

        return documents

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """Fix concatenated words, normalize spacing, and preserve readable layout"""
        if not text:
            return ""
        # Separate joined camelCase words like LargeLanguageModels -> Large Language Models
        cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        cleaned = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', cleaned)
        cleaned = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', cleaned)
        # Collapse multiple horizontal whitespace but keep newlines
        lines = []
        for line in cleaned.splitlines():
            line_str = re.sub(r'[ \t]+', ' ', line).strip()
            if line_str:
                lines.append(line_str)
        return '\n'.join(lines)

    @classmethod
    def _ocr_with_gemini(cls, pdf_path: str, filename: str, api_key: Optional[str] = None) -> List[Document]:
        """Use Gemini Multimodal Vision to accurately transcribe handwritten, scanned, or complex diagrammatic PDFs"""
        try:
            from config import Config
            from google import genai
            from google.genai import types

            key = (api_key or Config.GEMINI_API_KEY).strip()
            if not key:
                return []

            client = genai.Client(api_key=key)
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()

            prompt = (
                "You are an expert academic document transcriber and OCR engine.\n"
                "Please extract and transcribe the entire educational content of this PDF file with absolute precision.\n"
                "Structure your output strictly using page headers in this exact format:\n"
                "--- PAGE [page_number] ---\n"
                "For each page, transcribe all titles, paragraphs, handwritten notes, mathematical formulas, "
                "diagram descriptions, workflow steps, and definitions so a student can study and ask questions about them.\n"
                "Do not summarize or skip content. Transcribe comprehensively."
            )

            resp = client.models.generate_content(
                model=Config.DEFAULT_MODEL,
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
                    prompt
                ],
                config=types.GenerateContentConfig(temperature=0.1)
            )

            raw_transcription = resp.text or ""
            if not raw_transcription:
                return []

            documents = []
            page_blocks = re.split(r'---\s*PAGE\s*(\d+)\s*---', raw_transcription, flags=re.IGNORECASE)
            if len(page_blocks) > 1:
                # page_blocks: [preamble, "1", text_page_1, "2", text_page_2, ...]
                for i in range(1, len(page_blocks), 2):
                    page_num = int(page_blocks[i])
                    page_text = page_blocks[i + 1].strip()
                    if page_text:
                        documents.append(Document(
                            page_content=page_text,
                            metadata={
                                "source": filename,
                                "page": page_num,
                                "type": "pdf",
                                "extraction_method": "gemini_multimodal_ocr"
                            }
                        ))
            else:
                documents.append(Document(
                    page_content=raw_transcription.strip(),
                    metadata={
                        "source": filename,
                        "page": 1,
                        "type": "pdf",
                        "extraction_method": "gemini_multimodal_ocr"
                    }
                ))
            return documents
        except Exception as e:
            print(f"Gemini multimodal OCR fallback error: {e}")
            return []

    @classmethod
    def load_from_pdf(cls, pdf_path: str, api_key: Optional[str] = None) -> List[Document]:
        """Load text from PDF with layout preservation and smart multimodal fallback"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        filename = os.path.basename(pdf_path)
        documents = []

        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)

            for page_idx, page in enumerate(reader.pages):
                # Try layout mode first to preserve spaces between words and columns
                text = ""
                try:
                    text = page.extract_text(extraction_mode="layout") or ""
                except Exception:
                    pass
                if not text:
                    text = page.extract_text() or ""

                cleaned = cls._clean_extracted_text(text)
                if cleaned:
                    documents.append(Document(
                        page_content=cleaned,
                        metadata={
                            "source": filename,
                            "page": page_idx + 1,
                            "total_pages": total_pages,
                            "type": "pdf"
                        }
                    ))
        except Exception as err:
            print(f"pypdf reader error: {err}")

        # Check if extracted text is sparse, garbled, or handwriting/stylus artifacts
        total_words = sum(len(d.page_content.split()) for d in documents)
        avg_words_per_page = (total_words / max(1, len(documents))) if documents else 0

        # Detect handwriting/stylus artifact: high ratio of isolated single letters (e.g. 'T e a c h i n g')
        all_words = [w for d in documents for w in d.page_content.split()]
        single_chars = [w for w in all_words if len(w) == 1 and w.lower() not in ['a', 'i']]
        single_char_ratio = (len(single_chars) / max(1, len(all_words))) if all_words else 1.0

        # If sparse, missing, or high ratio of single-letter fragments, trigger multimodal OCR
        if (not documents or avg_words_per_page < 100 or single_char_ratio > 0.10) and os.path.exists(pdf_path):
            print(f"PDF needs multimodal OCR (avg {avg_words_per_page:.1f} words/page, single-char ratio {single_char_ratio:.2f}). Transcribing '{filename}' via Gemini...")
            ocr_docs = cls._ocr_with_gemini(pdf_path, filename, api_key=api_key)
            if ocr_docs:
                return ocr_docs

        return documents

    @staticmethod
    def load_from_docx(docx_path: str) -> List[Document]:
        """Load text from Microsoft Word documents"""
        if docx is None:
            raise ImportError("python-docx is not installed.")
        if not os.path.exists(docx_path):
            raise FileNotFoundError(f"DOCX not found: {docx_path}")

        doc = docx.Document(docx_path)
        filename = os.path.basename(docx_path)
        full_text = []

        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())

        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    full_text.append(" | ".join(row_text))

        combined = "\n\n".join(full_text)
        if not combined:
            return []

        return [Document(
            page_content=combined,
            metadata={
                "source": filename,
                "type": "docx"
            }
        )]

    @staticmethod
    def load_from_pptx(pptx_path: str) -> List[Document]:
        """Load slide contents from PowerPoint presentations"""
        if pptx is None:
            raise ImportError("python-pptx is not installed.")
        if not os.path.exists(pptx_path):
            raise FileNotFoundError(f"PPTX not found: {pptx_path}")

        prs = pptx.Presentation(pptx_path)
        filename = os.path.basename(pptx_path)
        documents = []

        for slide_idx, slide in enumerate(prs.slides):
            slide_texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text.strip())

            if slide_texts:
                content = "\n".join(slide_texts)
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": filename,
                        "slide": slide_idx + 1,
                        "total_slides": len(prs.slides),
                        "type": "pptx"
                    }
                ))
        return documents

    @staticmethod
    def load_from_text(file_path: str) -> List[Document]:
        """Load text/code file with resilient UTF-8 decoding"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        content = ""
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if not content.strip():
            return []

        return [Document(
            page_content=content,
            metadata={
                "source": filename,
                "type": "text"
            }
        )]

    @staticmethod
    def load_from_wikipedia(query: str) -> List[Document]:
        """Fetch article extract directly from Wikipedia REST API"""
        clean_query = query.strip()
        encoded = urllib.parse.quote(clean_query.replace(' ', '_'))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

        headers = {
            "User-Agent": "AITeachingAssistant/2.0 (contact@example.com) EducationalTool"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"Wikipedia topic '{query}' not found.")

        data = resp.json()
        title = data.get("title", query)
        extract = data.get("extract", "")
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{encoded}")

        if not extract:
            raise RuntimeError(f"No extract available for '{query}' on Wikipedia.")

        return [Document(
            page_content=extract,
            metadata={
                "source": f"Wikipedia: {title}",
                "url": page_url,
                "type": "wikipedia"
            }
        )]

    @classmethod
    def load_any_file(cls, file_path: str, api_key: Optional[str] = None) -> List[Document]:
        """Automatically route file based on extension"""
        ext = os.path.splitext(file_path)[1].lower().replace('.', '')
        if ext == 'pdf':
            return cls.load_from_pdf(file_path, api_key=api_key)
        elif ext == 'docx':
            return cls.load_from_docx(file_path)
        elif ext == 'pptx':
            return cls.load_from_pptx(file_path)
        elif ext in {'txt', 'md', 'csv', 'json', 'py', 'java', 'c', 'cpp', 'html', 'js', 'css'}:
            return cls.load_from_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: .{ext}")
