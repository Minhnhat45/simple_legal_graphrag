# Linux experiment results, 2026-11-18

**Status: diagnostic retrieval completed; answer and relationship evaluation attempts failed model-output validation.** No experiment jobs remain running. Independent legal review and a validated relational test benchmark remain unfinished.

The controlled diagnostic comparison does not establish a reliable graph advantage over both shared hybrid scoring and shuffled links.

This run continues [the plan](../plan_fixing_failure_11_11.md) and [the Linux handoff](linux_experiment_handoff_20261113.md). The original 2026-11-07 artifacts and collection were preserved. No tuning used these results.

## Corpus and identity coverage

The original corpus, evaluation CSV, manifest and frozen evaluation match all four hashes in the handoff. All 2,051 source-document fingerprints match the original Linux manifest. The Mac SQLite snapshot was absent, so it was rebuilt in a new directory. The first rebuild reproduced every reported count.

A tested normalization fix inserts the missing space before an uppercase sentence start after punctuation. It preserves letters, numbers, punctuation, source evidence and gold labels. It accepts only unique full-body matches within the same law number. It adds 4,919 mappings and rejects 119 previously unique mappings that become ambiguous (118 many-to-one IDs and one duplicate source match), for a net gain of 4,800.

| Measure | Rebuilt handoff snapshot | Frozen Linux snapshot |
| --- | ---: | ---: |
| Documents | 2,051 | 2,051 |
| Articles | 55,898 | 55,898 |
| Chunks | 91,405 | 91,405 |
| Validated original IDs | 31,307 | 36,107 |
| Fully mapped questions | 47 / 100 | 59 / 100 |
| Mapped gold occurrences | 53 / 113 | 66 / 113 |
| Automatically resolved citations | 19,674 | 19,674 |

The old chunk-containment rule and the new full-body rule cover 53 questions in common. Fifteen satisfy only the old rule and six only the new rule. Thus the original 68-question coverage count and the new 59-question count use different identity standards. This is not a measured retrieval regression. Forty-six documents still have no parsed articles. Ambiguous duplicates and substantive text/parser differences remain unresolved.

Under these accepted identity mappings, the all-question mean recall ceiling is 60.50%, and the complete-evidence ceiling is 59%. Unmapped identities may still have source text present; they cannot receive gold-ID retrieval credit without validated mapping.

## Controlled retrieval results

All seven configurations share the new BGE-M3 hybrid index, unchanged questions/gold labels, Top-K 10 and an 18,000-character source-text cap. Candidate caps are 60 except the explicitly larger 120-candidate baseline. These are matched upper bounds; realized counts differ. Shared scoring is the existing cosine-plus-token-overlap formula, not a trained cross-encoder. Query embeddings and actual hybrid pools are reused within each question. The successful output contains 700 records: 100 questions × seven configurations.

| Configuration | Candidate recall | Recall@10 | Complete evidence | Recall@10, fully mapped | New gold candidates | Recovered / lost gold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid` | 57.50% | 52.00% | 49.00% | 86.44% | 0 | 0 / 0 |
| `hybrid_larger` | 57.50% | 52.00% | 49.00% | 86.44% | 0 | 0 / 0 |
| `hybrid_shared` | 57.50% | 50.50% | 48.00% | 83.90% | 0 | 2 / 3 |
| `graph_shared` | 57.00% | 50.50% | 48.00% | 83.90% | 1 | 2 / 3 |
| `graph_shared_path` | 57.00% | 50.50% | 48.00% | 83.90% | 1 | 2 / 3 |
| `graph_legacy` | 57.00% | 52.50% | 50.00% | 87.29% | 1 | 1 / 0 |
| `shuffled_shared` | 54.00% | 49.50% | 47.00% | 82.20% | 0 | 2 / 4 |

New candidates and recovered/lost occurrences are relative to `hybrid`. Complete evidence means all gold article IDs occur among the selected chunks; it does not prove all required clauses were selected.

Paired percentile bootstrap intervals use 2,000 resamples and the frozen seed. The following Recall@10 differences are percentage points, treatment minus control. They are exploratory and not adjusted for multiple comparisons.

| Comparison | All 100 questions, difference [95% CI] | Fully mapped 59 |
| --- | ---: | ---: |
| `hybrid_larger_minus_hybrid` | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] |
| `hybrid_shared_minus_hybrid` | -1.50 pp [-6.00, +2.50] | -2.54 pp [-9.32, +4.24] |
| `graph_shared_minus_hybrid_shared` | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] |
| `graph_shared_minus_shuffled_shared` | +1.00 pp [+0.00, +3.00] | +1.69 pp [+0.00, +5.08] |
| `graph_shared_path_minus_graph_shared` | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] |
| `graph_shared_minus_graph_legacy` | -2.00 pp [-6.00, +2.00] | -3.39 pp [-10.17, +3.39] |

Two-hop graph traversal from actual hybrid seeds reaches 1 gold occurrences outside the baseline candidate pool. Operational bounded expansion adds 1 of them as confirmed candidates. Reachability includes unrestricted one/two-hop diagnostics; deployed expansion also applies degree, confidence, confirmation and candidate limits.

The sole new gold candidate is 56079 for qid 9882. It ranks 53rd with shared scoring, 44th with the path bonus and 49th with legacy graph scoring, so none selects it in the top 10. Graph expansion also displaces gold 53375 from the candidate pool for qid 567. The small legacy gain comes from qid 15959, whose recovered gold was already in the hybrid candidate pool. These cases support a narrow scoring/selection diagnosis, not a general benefit from newly reachable graph evidence.

| Diagnostic qid | Mapped gold | Hybrid top-10 gold | Graph/shared top-10 gold | Reachable beyond hybrid candidates |
| --- | --- | --- | --- | --- |
| 4190 | — | — | — | — |
| 9882 | 56006, 56079 | 56006 | 56006 | 56079 |
| 6920 | 5469 | — | — | — |
| 1164 | 8863 | 8863 | 8863 | — |
| 5124 | 59320 | — | — | — |
| 16281 | 46664 | — | 46664 | — |
| 2217 | — | — | — | — |

## Sampled answer evaluation

Before retrieval results were available, 20 diagnostic questions were selected by SHA-256 order with seed 20261118. Gemma generated answers for every configuration; Qwen was asked to judge correctness against full gold corpus provisions and support against the retrieved excerpts, blinded to configuration. Both use fixed temperature/seed and saved prompts. Identical ordered contexts reuse the exact answer and judgment. **All 140 answer judgments failed the required schema. Correctness and citation-support measurements are unavailable.** The 69 unique generation requests and 69 unique judge requests are preserved; shared contexts account for the 140 configuration records. The generator also produced truncated outputs in several cases. These failures must be repaired and rerun before drawing answer-quality conclusions.

| Configuration | Judged / 20 | Reference sufficient | Automated correctness | Automated full citation support | Truncated answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hybrid` | 0 | 0 | unavailable | unavailable | 8 |
| `hybrid_larger` | 0 | 0 | unavailable | unavailable | 7 |
| `hybrid_shared` | 0 | 0 | unavailable | unavailable | 7 |
| `graph_shared` | 0 | 0 | unavailable | unavailable | 7 |
| `graph_shared_path` | 0 | 0 | unavailable | unavailable | 7 |
| `graph_legacy` | 0 | 0 | unavailable | unavailable | 8 |
| `shuffled_shared` | 0 | 0 | unavailable | unavailable | 7 |

Correctness denominators exclude judgments marked insufficient reference; support denominators include all valid judgments. Failed or oversized-reference judgments are retained separately, not silently counted as correct. Raw answers, cited-source checks, model outputs and validation errors are saved. There are no valid paired answer-quality intervals.

## Relationship review and relational benchmark

The 50-relation precision sample contains 25 within-document and 25 cross-document edges selected before retrieval outcomes. The first Qwen response was prose rather than the required JSON, and the audit stopped. No precision measurement was produced. The exact request, response and traceback are preserved.

| Stratum | Population | Sample | Model-supported |
| --- | ---: | ---: | ---: |
| within_document | 17160 | 25 | unavailable |
| cross_document | 2514 | 25 | unavailable |

The population-weighted automated support rate is unavailable. The 50 sampled relations and 30 rejected/ambiguous cases are ready for independent review. No new implementing, amending or semantic edges were accepted from model judgments.

A separate 60-question natural review set was selected from 237 eligible multi-gold questions in the supplied training CSV, without selecting successful graph paths or requiring complete mapping. Thirty are development candidates and thirty are reserved test candidates. Law families join equal law numbers, co-gold laws and cross-document citations. The splits share no such family, and diagnostic families are excluded. Historical exposure of this training data is unknown; the reserved candidates were not run in this experiment. Reviewers must establish actual linkage requirements, rule/exception, condition/procedure and cross-document categories, gold identity and answers before this becomes a validated benchmark. No synthetic questions were added.

## Cost, reproducibility and limits

- Snapshot build: 71.50 seconds, including 26.50 seconds for citation extraction.
- Full embedding and Qdrant ingestion: 1183.43 seconds (19.72 minutes), batch size 64.
- Answer generation: 69 unique requests, 715.74 measured service seconds.
- Answer judgments: 69 unique requests, 335.89 measured service seconds.
- Hardware: NVIDIA RTX 5060 Ti, 16 GiB VRAM; Python 3.13.11. Package versions and model metadata are archived. No cloud API charges were incurred; electricity cost was not measured.

| Configuration | Mean retrieval ms | Median ms | P95 ms | Mean confirmed articles | Mean source characters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hybrid` | 24.36 | 24.13 | 29.61 | 47.72 | 12526 |
| `hybrid_larger` | 30.81 | 30.85 | 34.83 | 93.52 | 12558 |
| `hybrid_shared` | 23.90 | 23.88 | 28.52 | 47.72 | 12930 |
| `graph_shared` | 27.50 | 27.15 | 34.88 | 50.93 | 12929 |
| `graph_shared_path` | 27.13 | 27.02 | 33.41 | 50.93 | 12873 |
| `graph_legacy` | 27.56 | 26.95 | 34.79 | 50.93 | 12536 |
| `shuffled_shared` | 32.31 | 29.99 | 45.97 | 54.21 | 12919 |

Retrieval timings include the measured shared hybrid lookup and exclude query embedding. They are single-run local measurements, not a production load benchmark.

- Snapshot SHA-256: `3aee6d3d36c9f3e075b7e5a1ef3a1d0c394bf3720aac5ca1daa19aff1ec57113`.
- BGE-M3 digest: `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab` (matches the Mac digest).
- Collection: `legal_repair_linux_20261118_v2`.
- Validation: 66 tests, 64 passed and two opt-in live Neo4j tests skipped; lint passed. A live three-chunk smoke experiment also passed interruption/resumption and the existing-collection guard. All snapshot citation quotes occur in their saved endpoint chunks. The completed retrieval output was checked for all expected records, unique top-K IDs and candidate/context caps.

The initial run was interrupted by a server restart after 1,856 indexed chunks. Its partial collection and checkpoint are preserved; the final run uses a distinct snapshot and collection. The runner now validates snapshot/model/code/settings and existing point payloads on explicit `--resume`, and saves selected evidence for answer evaluation. Completed results cannot be overwritten through resume.

Raw artifacts are under `reports/runs/linux_20261118` (Git-ignored). They include both rebuilt snapshots, predictions, score/path traces, model requests and responses, identity differences, review packets, frozen protocol, logs, package versions, code hashes and a source archive. Copy this directory separately when moving machines. The compact machine-readable companion is [fixing_failure_linux_20261118.json](fixing_failure_linux_20261118.json).

To rerun, use the frozen snapshot with a new collection/output and `EMBED_BATCH_SIZE=64 .venv-linux/bin/python scripts/run_repaired_experiment.py`. Then use `summarize_repaired_experiment.py`, `run_answer_experiment.py` with the saved answer protocol and `--corpus data/vlsp/corpus.csv`, and `review_snapshot_relations.py` with the saved relation sample. The archived `finish_stages.sh` records the original command arguments. See the [data inventory](../data/README.md) for the current input paths.

**Remaining work:** repair the generator/judge output failures and rerun the frozen answer evaluation; complete the citation audit. Independent source/legal review is needed to validate relationship precision, answer correctness and the relational benchmark. The reserved test set remains unevaluated pending that review. The diagnostic and automated results do not establish universal GraphRAG superiority.
