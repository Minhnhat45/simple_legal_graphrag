"""Conservative provision identities without changing existing chunk identifiers."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

from .chunking import _article_matches
from .models import DocumentChunk, LegalDocument


@dataclass(frozen=True, slots=True)
class Provision:
    id: str
    document_id: str
    parent_id: str
    level: str
    label: str
    text: str
    start: int
    end: int


def article_id(document_id: str, section: str) -> str:
    match = re.fullmatch(r"Điều\s+(\d+[a-zđ]?)", section, re.IGNORECASE)
    if match:
        return f"{document_id}:article:{match[1].lower()}"
    return ""


def chunk_article_id(chunk: DocumentChunk) -> str:
    return article_id(chunk.document_id, chunk.section)


def parse_structure(document: LegalDocument) -> list[Provision]:
    """Only parse visible numbering. Never manufacture missing clause/point parents.

    Repeated article headings (e.g. attached regulations) use occurrence suffixes;
    their ambiguous section-only chunk mapping is intentionally left unresolved.
    """
    text = document.full_text
    articles = _article_matches(text)
    chapters = []
    for match in re.finditer(r"(?i)(?<!\w)Chương\s+([IVXLCDM]+|\d+)\b", text):
        prefix = text[: match.start()].rstrip()
        if not prefix or prefix[-1] in ".;:!?" or "\n" in text[len(prefix) : match.start()]:
            chapters.append(match)
    output: list[Provision] = []
    chapter_ids = {}
    seen: Counter = Counter()
    for i, match in enumerate(chapters):
        key = match[1].upper()
        seen[key] += 1
        cid = f"{document.id}:chapter:{key}:{seen[key]}"
        chapter_ids[match.start()] = cid
        end = chapters[i + 1].start() if i + 1 < len(chapters) else len(text)
        output.append(
            Provision(
                cid,
                document.id,
                document.id,
                "Chapter",
                match[0],
                text[match.start() : end],
                match.start(),
                end,
            )
        )
    labels = [re.sub(r"\s+", " ", m[1]).strip(" .:") for m in articles]
    counts = Counter(labels)
    seen.clear()
    for i, (match, label) in enumerate(zip(articles, labels, strict=True)):
        end = articles[i + 1].start() if i + 1 < len(articles) else len(text)
        # A chapter heading between articles belongs to the following chapter.
        following = [c.start() for c in chapters if match.start() < c.start() < end]
        if following:
            end = min(following)
        preceding = [c.start() for c in chapters if c.start() < match.start()]
        parent = chapter_ids[max(preceding)] if preceding else document.id
        seen[label] += 1
        aid = article_id(document.id, label)
        if counts[label] > 1:
            aid += f":occurrence:{seen[label]}"
        body = text[match.start() : end]
        output.append(
            Provision(aid, document.id, parent, "Article", label, body, match.start(), end)
        )
        # Flattened CSV bodies retain punctuation but not line boundaries.
        clauses = list(re.finditer(r"(?:^|(?<=[.;:\n]))\s*(\d+)\.\s+", body))
        clause_numbers = [c[1] for c in clauses]
        if len(set(clause_numbers)) != len(clauses):
            continue  # Nested/quoted numbering is ambiguous; retain article evidence.
        for j, clause in enumerate(clauses):
            stop = clauses[j + 1].start() if j + 1 < len(clauses) else len(body)
            cid = f"{aid}:clause:{clause[1]}"
            raw_value = body[clause.start() : stop]
            value = raw_value.strip()
            value_start = match.start() + clause.start() + len(raw_value) - len(raw_value.lstrip())
            output.append(
                Provision(
                    cid,
                    document.id,
                    aid,
                    "Clause",
                    clause[1],
                    value,
                    match.start() + clause.start(),
                    match.start() + stop,
                )
            )
            points = list(re.finditer(r"(?:^|(?<=[.;:\n]))\s*([a-zđ])\)\s+", value))
            if len({p[1] for p in points}) != len(points):
                continue
            for k, point in enumerate(points):
                pend = points[k + 1].start() if k + 1 < len(points) else len(value)
                output.append(
                    Provision(
                        f"{cid}:point:{point[1]}",
                        document.id,
                        cid,
                        "Point",
                        point[1],
                        value[point.start() : pend].strip(),
                        value_start + point.start(),
                        value_start + pend,
                    )
                )
    return output


def map_chunks(document: LegalDocument, chunks: list[DocumentChunk]) -> dict[str, str]:
    articles = [p for p in parse_structure(document) if p.level == "Article"]
    by_label: dict[str, list[str]] = {}
    for provision in articles:
        by_label.setdefault(provision.label, []).append(provision.id)
    return {c.id: by_label[c.section][0] for c in chunks if len(by_label.get(c.section, [])) == 1}


def structure_hash(document: LegalDocument) -> str:
    return hashlib.sha256(("structure/v2:" + document.fingerprint).encode()).hexdigest()
