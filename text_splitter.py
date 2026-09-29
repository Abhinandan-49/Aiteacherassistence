"""
Text Splitter Module
Splits documents into context-preserving overlapping chunks with rich metadata
"""
from typing import List, Optional
from config import Config
from document_loader import Document


class TextChunker:
    """Intelligent recursive text chunker preserving context and metadata"""

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split plain text string into overlapping chunks"""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Find best split point near end
            split_pos = -1
            search_window = text[max(start, end - 150):end]
            for sep in self.separators:
                pos = search_window.rfind(sep)
                if pos != -1:
                    split_pos = max(start, end - 150) + pos + len(sep)
                    break

            if split_pos == -1 or split_pos <= start:
                split_pos = end

            chunk = text[start:split_pos].strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward with overlap
            start = max(start + 1, split_pos - self.chunk_overlap)

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents and inject chunk index metadata"""
        all_chunks: List[Document] = []

        for doc_idx, doc in enumerate(documents):
            text_chunks = self.split_text(doc.page_content)
            total = len(text_chunks)
            for chunk_idx, text_chunk in enumerate(text_chunks):
                chunk_meta = dict(doc.metadata)
                chunk_meta["chunk_index"] = chunk_idx + 1
                chunk_meta["total_chunks"] = total
                chunk_meta["doc_index"] = doc_idx + 1

                all_chunks.append(Document(
                    page_content=text_chunk,
                    metadata=chunk_meta
                ))

        return all_chunks
