from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from .models import LegalDocument, Mention

DOCUMENT_NUMBER_RE = re.compile(
    r"(?<!\w)(\d{1,4}\s*/\s*(?:\d{4}\s*/\s*)?"
    r"[A-ZĐ][A-ZĐ0-9]*(?:\s*-\s*[A-ZĐ0-9]+)*)(?!\w)",
    re.IGNORECASE,
)

KIND_PATTERNS = (
    ("repeals", re.compile(r"\bbãi\s+bỏ\b", re.IGNORECASE)),
    ("replaces", re.compile(r"\bthay\s+thế\b", re.IGNORECASE)),
    ("amends", re.compile(r"\b(?:sửa\s+đổi|bổ\s+sung)\b", re.IGNORECASE)),
)


def normalize_document_number(number: str) -> str:
    normalized = unicodedata.normalize("NFC", number).upper().strip(" .,;:()[]")
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    return re.sub(r"\s+", "", normalized)


def normalize_entity_key(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def _mention_kind(text: str, start: int, end: int) -> str:
    window = text[max(0, start - 100) : min(len(text), end + 50)]
    for kind, pattern in KIND_PATTERNS:
        if pattern.search(window):
            return kind
    return "cites"


def extract_mentions(document: LegalDocument, *, max_unique: int = 500) -> list[Mention]:
    own_number = normalize_document_number(document.number)
    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {"number": "", "kinds": set(), "count": 0}
    )

    for match in DOCUMENT_NUMBER_RE.finditer(document.full_text):
        number = normalize_document_number(match.group(1))
        if not number or number == own_number:
            continue
        item = grouped[number]
        item["number"] = number
        kinds = item["kinds"]
        assert isinstance(kinds, set)
        kinds.add(_mention_kind(document.full_text, match.start(), match.end()))
        item["count"] = int(item["count"]) + 1
        if len(grouped) >= max_unique:
            break

    return [
        Mention(
            document_id=document.id,
            number=str(item["number"]),
            number_norm=number_norm,
            kinds=tuple(sorted(item["kinds"])),
            count=int(item["count"]),
        )
        for number_norm, item in grouped.items()
    ]
