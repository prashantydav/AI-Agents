import re
from typing import List


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks

    result = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        overlap_text = previous[-overlap:] if len(previous) > overlap else previous
        if overlap_text and not current.startswith(overlap_text):
            result.append(f"{overlap_text} {current}")
        else:
            result.append(current)

    return result


def split_fixed(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap

    return chunks


def split_recursive(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [text]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph
        else:
            if current:
                chunks.extend(split_recursive(current, chunk_size, overlap))
            if len(paragraph) <= chunk_size:
                current = paragraph
            else:
                sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', paragraph) if sentence.strip()]
                sentence_chunk = ""
                for sentence in sentences:
                    if len(sentence_chunk) + len(sentence) + 1 <= chunk_size:
                        sentence_chunk = f"{sentence_chunk} {sentence}".strip()
                    else:
                        if sentence_chunk:
                            chunks.append(sentence_chunk)
                        sentence_chunk = sentence
                if sentence_chunk:
                    chunks.append(sentence_chunk)
                current = ""

    if current:
        chunks.extend(split_recursive(current, chunk_size, overlap))

    return _add_overlap(chunks, overlap)


def split_semantic(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []

    sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', text) if sentence.strip()]
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return _add_overlap(chunks, overlap)
