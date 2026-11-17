# Execution report: failure-fixing plan, 2026-11-13

**Status: partially executed.** A new searchable source snapshot and citation graph were built, the scoring/control implementation was added, and regression checks passed. The first controlled hybrid/GraphRAG attempt stopped at Qdrant connection refusal. Docker and Ollama were subsequently recovered; the full re-embedding run remains deferred while source/article identity gaps are reviewed. No new Recall@10, answer-quality improvement, or graph advantage is claimed.

This report executes the available portions of [the plan](../plan_fixing_failure_11_11.md), using both [the original experiment report](semantic_graphrag_20261107.md) and [the failure analysis](failure_analysis_20261107.md), including their JSON companions. The original reports and plan were preserved.

## Verification of the previous experiment

The reports describe a Linux workspace under `/home/minhnhat/Projects/`, whereas this checkout is under `/Users/minhnhat/PycharmProjects/master-phenikaa/`. The failure-analysis JSON records the original evaluation at `/home/minhnhat/Projects/data_for_legal_rag/VLSP_legal_data/eval.csv`. The original snapshot belonged to `reports/runs/20261107/`, with collection `legal_filtered_semantic_20261107_v1`.

The experiment report explicitly states that the raw snapshots, verifier caches, graph audit and frozen predictions are **ignored by Git**. The repository's `.gitignore` confirms this. The compact reports are available, but they do not contain the complete original identity mapping, embeddings, graph, or hybrid seed lists. A Git checkout alone therefore cannot reproduce that frozen experiment. The recorded paths suggest a different original execution environment; they do not identify a remote hostname.

The corpus was recovered at `../data/corpus.csv`. Its SHA-256 is `79bf936ef8de4208beaf13e932075b12e6b9d67043311d602bf386b66ffe2edc`, exactly matching the failure analysis. The 100 diagnostic questions and unchanged gold IDs were recovered from the M0 records in the saved failure-analysis JSON. This is a derived diagnostic input, not a byte-identical restoration of `eval.csv` or an untouched test set.

The local settings and Compose file specify Qdrant on port 6333, Ollama on 11434 and Neo4j on 7687. Initially, all three localhost checks returned `[Errno 61] Connection refused`. Starting the installed Docker and Ollama applications restored Qdrant and Ollama. Qdrant contains only `legal_chunks_bge_m3_v1`, with 1,091 points; the old `legal_filtered_semantic_20261107_v1` collection is absent. BGE-M3 is installed and a real embedding probe succeeded. Neo4j remains unavailable. A broader search of this user’s PycharmProjects, Downloads and Documents found no original `eval.csv`, frozen evaluation, or chunk-audit files. No alternate server hostname is recorded in either report. The new comparison runner requires Qdrant and Ollama; it reads the citation graph directly from the snapshot and does not require Neo4j.

## Measured snapshot results

The final snapshot at `runs/20261113/coverage_final/` indexes every parsed article from the available 2,051 source documents. Unlike the old preparation script, it does not require an article to already have a reusable embedding. Articles are chunked independently, with an explicit article owner even when headings repeat. This produces a local SQLite FTS staging index; it does **not** establish that these articles are indexed in Qdrant.

| Measurement | Observed result |
| --- | ---: |
| Source documents read | 2,051 |
| Parsed, locally searchable articles | 55,898 |
| Chunks with explicit article ownership | 91,405 |
| Documents without parsed article headings | 46 |
| Original corpus articles parsed | 59,636 |
| Unambiguously mapped original article IDs | 31,307 |
| Resolved rule-based citation edges | 19,674 |
| Citation construction time | 31.43 seconds |
| Total snapshot build time | 81.30 seconds |
| Diagnostic gold occurrences mapped | 53 / 113 |
| Questions with all gold IDs mapped | 47 / 100 |

Every staged article has at least one chunk, and every citation endpoint exists in the snapshot. A real FTS search for `hôn nhân` returned 247 articles from the final persisted index. These are integrity/search smoke checks, not benchmark performance measurements. The SQLite artifact is approximately 699 MiB and has SHA-256 `01bfa48936fd223120fbf9894d0983b64f367c81ef55417d585ff2b2dad6bb49`.

**Article identity remains unresolved for much of the corpus.** The new audit normalizes NFC Unicode, whitespace and straight/curly quote typography without changing stored source evidence. It requires the complete normalized corpus article body, at least 30 characters long, to occur in exactly one parsed source article under the same law number. It rejects many-to-one mappings and never infers original IDs from article ordinal. There are 25,749 original IDs without an exact article match, 1,106 with ambiguous matches, and 1,474 rejected because multiple original IDs resolve to one source article. These categories do not by themselves prove that the legal source is absent; text differences, document variants and parsing granularity can also prevent mapping.

The old pilot reported 68 fully covered questions using validated stored **chunk** matches. The new strict **full-body** mapping covers 47. These are different mapping procedures and snapshots: the result cannot be called a coverage improvement, nor a measured retrieval regression. Restore the old manifest/chunk audit and review these discrepancies before using the new snapshot for a defensible before/after claim. The 46 documents without parsed articles remain explicit ingestion gaps.

| Diagnostic qid | Gold IDs mapped in new snapshot | Still unresolved |
| --- | --- | --- |
| 6920 | 5469 | — |
| 5124 | 59320 | — |
| 16281 | 46664 | — |
| 1164 | 8863 | — |
| 2217 | — | 16978, 17419 |
| 4190 | — | 28492 |
| 9882 | 56006 | 56079 |

These mappings confirm available evidence identities only; they do not show that retrieval now answers these questions.

The parser fix recognizes article headings after uppercase chapter/section titles in flattened documents. Previously, for example, Article 129 of the land-law source was absorbed into Article 128. The change has regression tests that also reject ordinary inline references. Explicit snapshot article ownership is persisted in Qdrant payloads, avoiding ambiguous section-label inference during later retrieval.

| Snapshot stage | Parsed articles | Mapped original IDs | Fully mapped questions |
| --- | ---: | ---: | ---: |
| Initial audit | 47,155 | 24,927 | 45 |
| Heading-boundary fix | 55,898 | 30,569 | 47 |
| Heading fix plus quote/NFC normalization | 55,898 | 31,307 | 47 |

All three snapshots are preserved. The measured improvement here is in parsing/mapping coverage under this audit, not retrieval accuracy. [Diagnostic mapping suggestions](mapping_review_final_20261113.json) rank possible source matches for review; similarity does not turn any suggestion into an accepted identity.

## Implemented comparison and diagnostics

The legacy scoring behavior remains the default. The opt-in shared scorer applies `0.7 × max(0, cosine) + 0.15 × query-token overlap` to every confirmed candidate. This is a transparent shared feature scorer, not a trained cross-encoder. A separate `shared_path` option adds the graph bonus. Per-article traces record cosine, overlap, hybrid rank, shared score, bonus, final score and candidate rank.

The new runner prepares these configurations against one new Qdrant collection:

| Configuration | Candidate cap | Scoring |
| --- | ---: | --- |
| Hybrid | 60 | Existing hybrid order |
| Larger hybrid pool | 120 | Existing hybrid order |
| Hybrid with shared scoring | 60 | Shared query/article features |
| Citation expansion with shared scoring | 60 | Same shared features |
| Citation expansion with path features | 60 | Shared features plus graph bonus |
| Citation expansion with legacy scoring | 60 | Existing mixed formula |
| Shuffled citation control | 60 | Shared features |

All configurations use Top-K 10 and an 18,000-character context cap. These are matched upper bounds; realized candidate counts and context lengths can differ. The larger-pool run is explicitly separate. Actual hybrid results are frozen across configurations sharing a pool size. Gold labels are used only for reporting, never seed or path selection.

Citation traversal is bounded, supports reverse traversal and retains the actual legal direction. The shuffled control permutes target endpoints with seed 20261113, preserving directed endpoint counts; randomized connections are never presented as legal evidence. Reachability diagnostics distinguish graph-node reachability from articles with staged chunks at one and two hops, before the operational expansion limits. Complete-evidence rate means coverage of all gold article IDs among representative chunks; it does not prove that those chunks contain every clause needed for an answer. The runner records confirmed candidate recall, new gold candidates, Recall@10, complete-evidence rate, recovered/lost gold evidence, latency, and paired percentile bootstrap intervals on full and fully mapped subsets. Embedding time is recorded separately.

**These comparison metrics were not produced in this execution.** The initial attempt failed on the Qdrant collection check before creating a collection. Its traceback is retained in `runs/20261113/comparison_attempt.log`. After recovering the services, a 64-chunk BGE-M3 probe took 4.37 seconds. A simple extrapolation for the final 91,405 chunks is about 104 minutes for embeddings alone, excluding indexing overhead; this is an estimate, not a measured full-build cost. Following the plan’s coverage-first sequence, the full run was deferred pending identity review. The recovered service/model details are saved in `runs/20261113/recovered_services.json`. Answer generation, correctness and citation-support judgments remain unexecuted. No generation-quality conclusion can be drawn from the unit tests.

## Plan completion and remaining work

| Plan stage | Result |
| --- | --- |
| Repair corpus coverage | New local article/chunk index built; identity audit completed. Vector ingestion, 46 unparsed documents and unresolved mappings remain. |
| Measure reachability from actual hybrid seeds | Implemented; real retrieval run deferred pending identity review. Saved compact reports cannot reconstruct actual old seeds. |
| Expand precise relations | Full local rule-based citation extraction completed. Additional implementing/amending and semantic relations were not fabricated or independently reviewed. |
| Compare scoring and controls | Implemented and regression-tested; initial attempt failed; services recovered, full comparison deferred. |
| Independently reviewed relational benchmark | A 42-item review queue was created from existing multi-gold and failed natural questions. It is not a validated benchmark or an independently authored test set. |

The [review queue](relational_review_queue_20261113.json) deliberately leaves relation category, linkage requirement, reviewer and reviewed answer unset. Add independently authored cases, obtain independent source review, and freeze document-family-separated development/test data before tuning. Existing diagnostic questions must not become an allegedly untouched test set.

The 19,674 citation edges are automatically resolved by the existing conservative rule validator; their legal precision has not been independently measured. More edges are not evidence of better retrieval. The old report's 245 citations came from a much smaller searchable pilot, so raw edge counts are not an effectiveness comparison.

## Validation and reproduction

Validation ran **61 tests: 59 passed and two live Neo4j tests were skipped**. New tests cover shared scoring, context limits, reverse two-hop traversal, shuffled links without fabricated evidence, paired intervals, corpus parsing and persisted explicit article ownership. Ruff passed for all changed Python files. The actual snapshot also passed article/chunk coverage and citation-endpoint integrity checks.

Build another snapshot into a fresh directory:

```bash
.venv/bin/python scripts/repair_coverage.py \
  --corpus ../data/corpus.csv \
  --output reports/runs/20261113/coverage_NEW
```

With the recovered Qdrant and Ollama services, and after reviewing the identity gaps, run the prepared comparison using the existing snapshot and a new output path:

```bash
.venv/bin/python scripts/run_repaired_experiment.py \
  --snapshot reports/runs/20261113/coverage_final \
  --collection legal_repair_20261113_NEW \
  --output reports/runs/20261113/comparison_NEW
```

This will embed 91,405 chunks; the probe-based estimate is about 104 minutes for embeddings alone. Full-run cost has not been measured. It refuses to overwrite an existing collection or output directory. The new experiment records the model digest, snapshot hash, configuration, partial raw predictions and final results. Resolve the mapping discrepancies before treating its benchmark scores as a coverage-repair result.

Compact results are in [fixing_failure_20261113.json](fixing_failure_20261113.json). Detailed coverage, every mapping gap, per-question diagnostics, SQLite evidence, service checks, test/lint logs and installed package versions are under `reports/runs/20261113/`. As with the original experiment, this raw artifact directory is ignored by Git and must be copied separately when moving machines.
