#!/usr/bin/env python3
"""Find the most frequent word n-grams in the cleaned legal corpus.

The corpus is too large for an unbounded ``Counter``.  This script therefore
uses the Misra-Gries heavy-hitter algorithm to find candidates in one streaming
pass, then makes a second pass to report exact counts for those candidates.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator
from pathlib import Path


DEFAULT_FIELDS = ("Title", "Toàn văn")
DEFAULT_OUTPUT = Path("output/ngram_frequencies.txt")
WORD_PATTERN = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)
Ngram = tuple[str, ...]


def raise_csv_field_limit() -> None:
    """Allow multi-megabyte full-text CSV fields on every platform."""

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def find_csv_files(directory: Path) -> list[Path]:
    """Return all CSV files below ``directory`` in deterministic order."""

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv"
    )


def iter_ngrams(text: str, n: int) -> Iterator[Ngram]:
    """Yield normalized word n-grams without crossing a text-field boundary."""

    window: deque[str] = deque(maxlen=n)
    for match in WORD_PATTERN.finditer(text.casefold()):
        window.append(match.group())
        if len(window) == n:
            yield tuple(window)


def iter_csv_texts(path: Path, fields: tuple[str, ...]) -> Iterator[str]:
    """Stream selected text fields from one cleaned CSV file."""

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        missing = [field for field in fields if field not in columns]
        if missing:
            raise ValueError(
                f"{path} is missing requested column(s): {', '.join(missing)}"
            )

        for row in reader:
            for field in fields:
                text = row.get(field) or ""
                if text:
                    yield text


def iter_corpus_texts(
    paths: Iterable[Path], fields: tuple[str, ...]
) -> Iterator[str]:
    for path in paths:
        yield from iter_csv_texts(path, fields)


class MisraGries:
    """Bounded-memory frequent-items candidate finder.

    Counts are stored relative to a global decrement level.  Buckets let a
    full-table decrement remove zero-count entries without scanning the table.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._level = 0
        self._counts: dict[Ngram, int] = {}
        self._buckets: defaultdict[int, set[Ngram]] = defaultdict(set)

    def add(self, item: Ngram) -> None:
        old_count = self._counts.get(item)
        if old_count is not None:
            bucket = self._buckets[old_count]
            bucket.remove(item)
            if not bucket:
                del self._buckets[old_count]
            new_count = old_count + 1
            self._counts[item] = new_count
            self._buckets[new_count].add(item)
            return

        if len(self._counts) < self.capacity:
            new_count = self._level + 1
            self._counts[item] = new_count
            self._buckets[new_count].add(item)
            return

        # Standard Misra-Gries step for an unseen item when the table is full:
        # decrement every candidate and discard candidates that reach zero.
        self._level += 1
        expired = self._buckets.pop(self._level, ())
        for candidate in expired:
            del self._counts[candidate]

    @property
    def candidates(self) -> set[Ngram]:
        return set(self._counts)


def scan_candidates(
    paths: list[Path],
    fields: tuple[str, ...],
    n: int,
    candidate_limit: int,
    progress_every: int,
) -> tuple[set[Ngram], int, int]:
    """Run the bounded-memory candidate pass."""

    finder = MisraGries(candidate_limit)
    text_fields = 0
    total_ngrams = 0
    started = time.monotonic()

    for text in iter_corpus_texts(paths, fields):
        text_fields += 1
        for ngram in iter_ngrams(text, n):
            finder.add(ngram)
            total_ngrams += 1

        if progress_every and text_fields % progress_every == 0:
            elapsed = time.monotonic() - started
            print(
                f"candidate pass: fields={text_fields:,}, "
                f"n-grams={total_ngrams:,}, elapsed={elapsed:.1f}s",
                file=sys.stderr,
            )

    return finder.candidates, text_fields, total_ngrams


def count_candidates(
    paths: list[Path],
    fields: tuple[str, ...],
    n: int,
    candidates: set[Ngram],
    progress_every: int,
) -> Counter[Ngram]:
    """Re-read the corpus and obtain exact counts for candidate n-grams."""

    counts: Counter[Ngram] = Counter()
    text_fields = 0
    started = time.monotonic()

    for text in iter_corpus_texts(paths, fields):
        text_fields += 1
        for ngram in iter_ngrams(text, n):
            if ngram in candidates:
                counts[ngram] += 1

        if progress_every and text_fields % progress_every == 0:
            elapsed = time.monotonic() - started
            print(
                f"exact-count pass: fields={text_fields:,}, "
                f"elapsed={elapsed:.1f}s",
                file=sys.stderr,
            )

    return counts


def write_report(
    output_path: Path,
    ranked: list[tuple[Ngram, int]],
    *,
    n: int,
    fields: tuple[str, ...],
    paths: list[Path],
    text_fields: int,
    total_ngrams: int,
    candidate_limit: int,
) -> None:
    """Write a UTF-8 plain-text frequency report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(f"N-gram size: {n}\n")
        output.write(f"Fields: {', '.join(fields)}\n")
        output.write(f"CSV files: {len(paths)}\n")
        output.write(f"Text fields read: {text_fields}\n")
        output.write(f"N-grams scanned: {total_ngrams}\n")
        output.write(f"Candidate limit: {candidate_limit}\n")
        output.write("\nRank\tCount\tPhrase\n")
        for rank, (ngram, count) in enumerate(ranked, start=1):
            output.write(f"{rank}\t{count}\t{' '.join(ngram)}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find frequently repeated word n-grams in every cleaned CSV and "
            "write the ranked phrases to a text file."
        )
    )
    parser.add_argument(
        "input_directory",
        nargs="?",
        type=Path,
        default=Path("data/clean"),
        help="directory scanned recursively for CSV files (default: data/clean)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"text report path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=3,
        help="number of words in each phrase (default: 3)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="number of ranked phrases to export (default: 100)",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=100_000,
        metavar="COUNT",
        help=(
            "maximum heavy-hitter candidates kept in memory; larger values "
            "improve recall (default: 100000)"
        ),
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(DEFAULT_FIELDS),
        metavar="COLUMN",
        help='CSV text columns to analyze (default: "Title" "Toàn văn")',
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        metavar="FIELDS",
        help="show progress after this many text fields; 0 disables it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input_directory.is_dir():
        parser.error(f"directory does not exist: {args.input_directory}")
    if args.n < 1:
        parser.error("-n must be at least 1")
    if args.top < 1:
        parser.error("--top must be at least 1")
    if args.candidate_limit < args.top:
        parser.error("--candidate-limit must be greater than or equal to --top")
    if args.progress_every < 0:
        parser.error("--progress-every must be zero or greater")
    if not args.fields or any(not field.strip() for field in args.fields):
        parser.error("--fields must contain at least one non-empty column name")

    paths = find_csv_files(args.input_directory)
    if not paths:
        parser.error(f"no CSV files found under {args.input_directory}")

    raise_csv_field_limit()
    fields = tuple(args.fields)
    try:
        candidates, text_fields, total_ngrams = scan_candidates(
            paths,
            fields,
            args.n,
            args.candidate_limit,
            args.progress_every,
        )
        exact_counts = count_candidates(
            paths,
            fields,
            args.n,
            candidates,
            args.progress_every,
        )
        ranked = exact_counts.most_common(args.top)
        write_report(
            args.output,
            ranked,
            n=args.n,
            fields=fields,
            paths=paths,
            text_fields=text_fields,
            total_ngrams=total_ngrams,
            candidate_limit=args.candidate_limit,
        )
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {len(ranked)} ranked {args.n}-gram phrases to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
