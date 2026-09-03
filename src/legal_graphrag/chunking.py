from __future__ import annotations

import re

from .models import DocumentChunk, LegalDocument

ARTICLE_RE = re.compile(r"(?i)(?<!\w)(Điều\s+\d+[a-zA-ZĐđ]?(?:\s*[.:])?)\s*")
BREAK_RE = re.compile(r"[.;!?](?:\s+|$)")


def _article_matches(text: str) -> list[re.Match[str]]:
    matches: list[re.Match[str]] = []
    for match in ARTICLE_RE.finditer(text):
        prefix = text[max(0, match.start() - 40) : match.start()].rstrip()
        if not prefix or prefix[-1] in ".;:!?":
            matches.append(match)
            continue
        if re.search(r"(?i)Chương\s+[IVXLCDM\d]+$", prefix):
            matches.append(match)
    return matches


def _section_segments(text: str) -> list[tuple[str, str]]:
    # A bare regex would also split references such as "theo Điều 10 Nghị định...".
    # Legal headings in the flattened corpus normally begin the text or follow punctuation.
    matches = _article_matches(text)
    if not matches:
        return [("Toàn văn", text)]

    segments: list[tuple[str, str]] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        segments.append(("Phần mở đầu", prefix))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[match.start() : end].strip()
        if section_text:
            section = re.sub(r"\s+", " ", match.group(1)).strip(" .:")
            segments.append((section, section_text))
    return segments


def _window_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    windows: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            candidate = text[start:hard_end]
            minimum = int(max_chars * 0.6)
            boundaries = [match.end() for match in BREAK_RE.finditer(candidate)]
            usable = [position for position in boundaries if position >= minimum]
            if usable:
                end = start + usable[-1]

        piece = text[start:end].strip()
        if piece:
            windows.append(piece)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        while next_start < end and not text[next_start].isspace():
            next_start += 1
        start = next_start if next_start < end else max(start + 1, end - overlap)
    return windows


def chunk_document(
    document: LegalDocument,
    *,
    max_chars: int,
    overlap: int,
) -> list[DocumentChunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars")

    chunks: list[DocumentChunk] = []
    for section, section_text in _section_segments(document.full_text):
        header = f"{document.title}\nSố hiệu: {document.number}\nMục: {section}\n"
        body_limit = max(200, max_chars - len(header))
        body_overlap = min(overlap, max(0, body_limit - 1))
        for body in _window_text(section_text, body_limit, body_overlap):
            index = len(chunks)
            chunks.append(
                DocumentChunk(
                    id=f"{document.id}:{index}",
                    document_id=document.id,
                    index=index,
                    section=section,
                    text=f"{header}{body}".strip(),
                )
            )

    if not chunks:
        chunks.append(
            DocumentChunk(
                id=f"{document.id}:0",
                document_id=document.id,
                index=0,
                section="Toàn văn",
                text=f"{document.title}\nSố hiệu: {document.number}",
            )
        )
    return chunks
