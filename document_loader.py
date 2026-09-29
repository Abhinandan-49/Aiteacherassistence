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
        Load transcript from YouTube video with timestamps and video metadata
        """
        video_id = cls.extract_youtube_video_id(video_url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {video_url}")

        # Fetch video title via oEmbed
        video_title = f"YouTube Lecture ({video_id})"
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            resp = requests.get(oembed_url, timeout=5)
            if resp.status_code == 200:
                video_title = resp.json().get("title", video_title)
        except Exception:
            pass

        try:
            # Try fetching transcript
            # Use YouTubeTranscriptApi.get_transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            # Try fallback language or list_transcripts
            try:
                transcript_info = YouTubeTranscriptApi.list_transcripts(video_id)
                # Find any available transcript (generated or manual)
                transcript = transcript_info.find_transcript(['en', 'en-US', 'en-GB', 'hi', 'es', 'fr', 'de'])
                transcript_list = transcript.fetch()
            except Exception as inner_e:
                raise RuntimeError(f"Could not retrieve transcript for YouTube video: {inner_e}")

        # Group transcript items into readable ~60-90 second segments with timestamp references
        documents = []
        current_chunk_text = []
        start_time = 0.0

        for item in transcript_list:
            text = item.get("text", "").strip()
            ts = item.get("start", 0.0)
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
    def load_from_pdf(pdf_path: str) -> List[Document]:
        """Load text from PDF with page numbers preserved"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PdfReader(pdf_path)
        documents = []
        filename = os.path.basename(pdf_path)

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_idx + 1,
                        "total_pages": len(reader.pages),
                        "type": "pdf"
                    }
                ))
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
    def load_any_file(cls, file_path: str) -> List[Document]:
        """Automatically route file based on extension"""
        ext = os.path.splitext(file_path)[1].lower().replace('.', '')
        if ext == 'pdf':
            return cls.load_from_pdf(file_path)
        elif ext == 'docx':
            return cls.load_from_docx(file_path)
        elif ext == 'pptx':
            return cls.load_from_pptx(file_path)
        elif ext in {'txt', 'md', 'csv', 'json', 'py', 'java', 'c', 'cpp', 'html', 'js', 'css'}:
            return cls.load_from_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: .{ext}")
