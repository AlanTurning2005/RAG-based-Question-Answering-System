import hashlib
import re
from typing import Any, Dict, List


class SentenceWindowChunker:

    def __init__(self, window_size: int = 1):
        """Sentence-Window Chunker cho bài đọc RACE.

        :param window_size: Số câu lấy thêm ở phía trước và phía sau câu mục
        tiêu (k)
        """
        self.window_size = window_size

    @staticmethod
    def hash_text(text: str) -> str:
        """Hash nội dung bài đọc để khử trùng lặp (Deduplication)."""
        cleaned = " ".join(text.strip().lower().split())
        return hashlib.md5(cleaned.encode("utf-8")).hexdigest()

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Tách câu chính xác theo ranh giới ngữ pháp tiếng Anh."""
        # Regex tách câu theo dấu kết thúc ., !, ? kèm khoảng trắng
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s.strip()) > 0
        ]
        return sentences

    def chunk_article(
        self, example_id: str, article_text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Chia bài đọc thành các đơn vị câu và ngữ cảnh mở rộng

        (Large).
        """
        sentences = self.split_sentences(article_text)
        n = len(sentences)
        chunks = []
        base_meta = metadata or {}

        for i, sentence in enumerate(sentences):
            start_idx = max(0, i - self.window_size)
            end_idx = min(n, i + self.window_size + 1)
            window_context = " ".join(sentences[start_idx:end_idx])

            chunk_record = {
                "id": f"{example_id}_sent_{i:03d}",
                "text_to_embed": sentence,  
                "window_context": window_context,  
                "metadata": {
                    **base_meta,
                    "example_id": example_id,
                    "sentence_idx": i,
                    "total_sentences": n,
                },
            }
            chunks.append(chunk_record)

        return chunks

    