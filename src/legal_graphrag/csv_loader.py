from __future__ import annotations

import csv
import re
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from .models import LegalDocument

REQUIRED_COLUMNS = (
    "ID",
    "Title",
    "Toàn văn",
    "Số hiệu",
    "Loại văn bản",
    "Ngành",
    "Ngày ban hành",
    "Lĩnh vực",
    "Ngày có hiệu lực",
    "Tình trạng hiệu lực",
    "Ngày hết hiệu lực",
    "Cơ quan ban hành",
    "Chức danh",
    "Người ký",
)

MISSING_VALUES = {"", "--", "null", "none", "n/a"}
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str | None) -> str:
    return WHITESPACE_RE.sub(" ", value or "").strip()


def normalize_optional(value: str | None) -> str:
    cleaned = normalize_text(value)
    return "" if cleaned.casefold() in MISSING_VALUES else cleaned


def normalize_date(value: str | None) -> str:
    cleaned = normalize_optional(value)
    if not cleaned:
        return ""
    for date_format in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, date_format).date().isoformat()
        except ValueError:
            continue
    return cleaned


def _validate_header(fieldnames: list[str] | None) -> None:
    available = set(fieldnames or [])
    missing = [column for column in REQUIRED_COLUMNS if column not in available]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")


def stream_documents(
    csv_path: str | Path,
    *,
    limit: int | None = None,
) -> Iterator[LegalDocument]:
    """Yield documents without retaining the complete CSV in memory."""

    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")

    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_header(reader.fieldnames)

        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            document_id = normalize_text(row.get("ID"))
            if not document_id:
                raise ValueError(f"Row {index + 2} has an empty ID")
            yield LegalDocument(
                id=document_id,
                title=normalize_text(row.get("Title")),
                full_text=normalize_text(row.get("Toàn văn")),
                number=normalize_text(row.get("Số hiệu")),
                document_type=normalize_text(row.get("Loại văn bản")),
                sector=normalize_text(row.get("Ngành")),
                issued_date=normalize_date(row.get("Ngày ban hành")),
                field=normalize_text(row.get("Lĩnh vực")),
                effective_date=normalize_date(row.get("Ngày có hiệu lực")),
                status=normalize_text(row.get("Tình trạng hiệu lực")),
                expiry_date=normalize_date(row.get("Ngày hết hiệu lực")),
                agency=normalize_optional(row.get("Cơ quan ban hành")),
                signer_position=normalize_optional(row.get("Chức danh")),
                signer=normalize_optional(row.get("Người ký")),
            )
