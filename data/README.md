# Project data

Paths in experiment commands are relative to the repository root. The data files
and `reports/runs/` are Git-ignored and should accompany a checkout when moving
the experiments to another machine.

## VLSP inputs

The original VLSP files used by the experiments are stored byte for byte in
`data/vlsp/`. Their SHA-256 digests match the frozen run records.

| Path relative to this directory | Use | SHA-256 |
| --- | --- | --- |
| [`vlsp/corpus.csv`](vlsp/corpus.csv) | Original article text, law numbers, and article IDs | `79bf936ef8de4208beaf13e932075b12e6b9d67043311d602bf386b66ffe2edc` |
| [`vlsp/eval.csv`](vlsp/eval.csv) | Original 100-question evaluation and gold article IDs | `5768fcf9760582426862b1f35df3160c682f55b7a96ed8da269dc07de4c8cab7` |
| [`vlsp/train.csv`](vlsp/train.csv) | Source questions for relational benchmark selection | `33826d891cde9e992be2e67d4c7d10fba309464ce138dfae54791a1235020554` |

## Frozen evaluation inputs

| Repository path | Contents |
| --- | --- |
| [`reports/runs/linux_20261118/coverage_v2/coverage.json`](../reports/runs/linux_20261118/coverage_v2/coverage.json) | The Linux run's frozen diagnostics and article mappings. |
| [`reports/runs/openrouter_20261119/snapshot/coverage.json`](../reports/runs/openrouter_20261119/snapshot/coverage.json) | The 100 diagnostics and mappings read by the GPT-6-Luna evaluator. |
| [`reports/runs/linux_20261118/review/relational_candidates.json`](../reports/runs/linux_20261118/review/relational_candidates.json) | The frozen split of 30 development candidates and 30 reserved test candidates. |

Keep the frozen inputs with their run directories so the saved protocols can
verify them. See the [Linux handoff](../reports/linux_experiment_handoff_20261113.md)
and [OpenRouter report](../reports/openrouter_experiment_20261119.md) for reproduction
commands.

## Filtered legal documents

The ingestion dataset contains source records whose document number exactly
matches a `law_id` in `data/vlsp/corpus.csv`.

### Matching rule

- Extract unique, nonempty `law_id` values using a CSV parser, including support
  for quoted multiline article content.
- For department CSV files, compare the complete `Số hiệu` field with the corpus
  ID. Preserve every matching row and every original column, grouped by the
  source CSV filename. The numeric `ID` column is not the law number.
- For ViLegalExperts, read the `Số hiệu:` field in context metadata across all
  JSON datasets outside `contexts/`. Resolve the law document using the referenced
  context filename or an exactly equal document URL.
- Also inspect local context documents for their own number: the first line
  beginning with `Số:`, `Luật số:`, or `Pháp lệnh số:` within the first 2,000
  characters of `passage`. Compare the complete extracted value after stripping
  surrounding whitespace. The label may vary in case; the number must match
  case-sensitively. This conservative header rule may miss differently formatted
  headers when no metadata is available.
- Do not normalize ID case, accents, punctuation, leading zeros, or internal
  whitespace. Do not match partial IDs, URL slugs, titles, or body citations.
- Keep all matching source records. A shared law number is not proof of the
  same issuing authority, date, or version; no such disambiguation is inferred.

### Files

| Path | Contents |
| --- | --- |
| `law_ids.csv` | Unique corpus law IDs and source record counts. |
| `departments/*.csv` | Matching rows with all original fields and field values. |
| `ViLegalExperts/contexts/*.json` | Matching local context documents, copied byte for byte. |
| `ViLegalExperts/metadata_matches.jsonl` | Matching metadata grouped by referenced context filename and law ID, including an original example, source-file occurrence counts, and local-document availability. These entries can exist even when their document is missing locally. |
| `matches.jsonl` | One entry per saved department row or ViLegalExperts document/law-ID pair, with source location, matching evidence, and checksum. Department record indices are one-based data-row indices, excluding the CSV header. Department checksums cover canonical JSON field values; context checksums cover raw file bytes. |
| `coverage.csv` | Per-law document counts, metadata counts, and availability status. |
| `missing_law_ids.csv` | Corpus IDs with no matching local document in either source. Metadata-only matches are identified separately. |
| `issues.jsonl` | Missing referenced documents and disagreements between header numbers and metadata. |
| `summary.json` | Input paths, totals, matching rule, and completed validation counts. |

Source files remain untouched. Department CSV formatting may change during
serialization, but parsed field values are preserved. Repeated matching records
remain separate; each context file is copied only once. Metadata samples and
occurrence counts provide provenance; they are not a filtered question-answer
training split.

### Rebuild the source filter

The experiments ingest the existing filtered dataset directly. To rebuild the
filter itself, supply the original unfiltered department and ViLegalExperts
directories and choose a new output directory:

```sh
python3 data/build.py --corpus data/vlsp/corpus.csv \
  --departments /path/to/raw/departments --experts /path/to/raw/ViLegalExperts \
  --output data_filter_rebuilt
```

The script uses only the Python standard library. Existing generated outputs
are never overwritten. Files are built in a temporary directory and
validated before being moved into the output directory; `summary.json` is moved
last. Validation rereads every saved document, checks that all matched IDs are
in the corpus, and compares saved department field values and copied context
bytes against checksums computed from the source records.
