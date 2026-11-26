# OpenRouter GraphRAG experiment, 2026-11-19

**Status: computational stages completed.** Graph rebuilding, controlled retrieval, answer generation/judging, and automated relationship review used the new experiment stores. Independent human review and the reserved relational test evaluation remain pending.

The paired diagnostic results do not establish a reliable retrieval advantage over both shared hybrid scoring and shuffled links.
The new semantic edges did not activate on the diagnostic questions, so this run does not provide an effective test of their retrieval value.
The completed remote-model judgments resolve the earlier output-format failure. Their validity as measurements of legal correctness still requires independent review.

This run continues [the Linux experiment](fixing_failure_linux_20261118.md) and [the failure-fixing plan](../plan_fixing_failure_11_11.md). No earlier collection, graph snapshot, prediction file or raw model response was overwritten.

## Model and isolated databases

- Chat model: `openai/gpt-6-luna` through OpenRouter; discovered canonical version `openai/gpt-6-luna-20260922`.
- Every model call requests **medium reasoning**, strict JSON schemas where applicable, and no fallback model. Requests are cached by endpoint and complete payload. Truncated, refused, malformed or schema-invalid outputs fail closed.
- New Qdrant collection: `legal_openrouter_20261119_v1`.
- New graph database: `reports/runs/openrouter_20261119/snapshot/snapshot.sqlite`.
- Source corpus: 2,051 documents, 55,898 articles and 91,405 chunks, unchanged from the frozen Linux snapshot.
- Dense BGE-M3 and sparse BM25 vectors were copied from the verified collection. BGE-M3 is used only for query embeddings; relation extraction, verification, answers, benchmark screening and judging use OpenRouter. This holds the retrieval representation constant while changing the generative model.
- Fully mapped diagnostic questions: 59 / 100; mapped gold occurrences: 66 / 113. The mapping-based all-question recall ceiling remains 60.5%.

The API implementation follows OpenRouter’s [structured-output documentation](https://openrouter.ai/docs/guides/features/structured-outputs) and [reasoning-token documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens). Reasoning tokens share the completion budget; experimental calls allow 8,192 completion tokens. No temperature parameter is sent because it is not advertised for this model. The API key is read from `OPEN_ROUTER_API` and excluded from saved requests and source archives.

## Rebuilt relations

The source-only protocol samples 1,000 articles by stable article-ID hash and classifies 1,000 candidate pairs. It alternates explicit-reference pairs with stored-vector/BM25 similarity candidates. Evaluation questions and gold labels never enter candidate discovery. This is a bounded graph enrichment experiment, not exhaustive LLM analysis of every possible article pair.

Rule extraction rebuilt 19,674 citation edges. Luna produced 285 accepted LLM relation occurrences after exact source/target quote validation, clause-scope checks and a separate verification request. 539 proposals were rejected. Duplicate edge keys are consolidated in the final graph.

| Relation kind | Accepted LLM occurrences | Final graph edges |
| --- | ---: | ---: |
| CITES | 206 | 19678 |
| PENALTY_FOR | 1 | 1 |
| REPEALS | 2 | 2 |
| SETS_CONDITION_FOR | 7 | 7 |
| SPECIFIES | 69 | 69 |

Accepted edges retain both endpoint quotes, original direction, classification and verification request hashes, candidate provenance and temporal uncertainty. Reverse traversal retains the original legal direction. Automatically resolved rule citations are distinguished from LLM-reviewed relations; none is presented as independently validated legal truth.

## Diagnostic retrieval: 100 questions

All eight configurations share the new corpus/index, Top-K 10 and an 18,000-character evidence cap. Candidate caps are 60, except the separate larger-pool baseline at 120. The citation control uses only the original rule-citation graph; enriched graph modes enable both formal and semantic relationships. Shuffled endpoints use seed 20261113. Scoring remains the frozen cosine-plus-lexical formula; no weights were tuned on these results.

| Configuration | Candidate recall | Recall@10 | Complete evidence | Recall@10, fully mapped | New gold candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hybrid` | 57.50% | 52.00% | 49.00% | 86.44% | 0 |
| `hybrid_larger` | 57.50% | 52.00% | 49.00% | 86.44% | 0 |
| `hybrid_shared` | 57.50% | 50.50% | 48.00% | 83.90% | 0 |
| `citation_shared` | 57.00% | 50.50% | 48.00% | 83.90% | 1 |
| `graph_shared` | 57.00% | 50.50% | 48.00% | 83.90% | 1 |
| `graph_shared_path` | 57.00% | 50.50% | 48.00% | 83.90% | 1 |
| `graph_legacy` | 57.00% | 52.50% | 50.00% | 87.29% | 1 |
| `shuffled_shared` | 56.00% | 50.50% | 48.00% | 83.90% | 0 |

Non-citation relation kinds were confirmed during expansion for 0 diagnostic queries. The copied-index hybrid baseline changed the ordered top-10 list on 45 queries relative to the previous run, and changed gold recall on 0. The main comparisons are paired within this new run.

Paired bootstrap intervals use 2,000 resamples, seed 20261113. Differences below are enriched graph/shared minus the control, in percentage points; these exploratory intervals are not multiplicity-adjusted.

| Control | Recall difference [95% CI] | Complete-evidence difference [95% CI] |
| --- | ---: | ---: |
| `hybrid` | -1.50 pp [-6.00, +2.50] | -1.00 pp [-5.00, +3.00] |
| `hybrid_shared` | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] |
| `citation_shared` | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] |
| `shuffled_shared` | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] |

For gold occurrences missed by the new hybrid top ten, the following separates graph reachability from actual selection. One- and two-hop counts describe retrievable article nodes; reaching a node does not ensure its representative chunk contains the necessary clause.

| Diagnostic quantity | Gold occurrences |
| --- | ---: |
| Missed by hybrid top ten | 58 |
| Those reachable within 1 graph hop(s) | 2 |
| Those reachable within 2 graph hop(s) | 2 |
| Recovered in enriched graph/shared top ten | 2 |

The cases named in the failure-fixing plan remain diagnostics, with no weights tuned to them:

| Question ID | Fully mapped | Hybrid Recall@10 | Citation/shared Recall@10 | Enriched graph/shared Recall@10 |
| --- | --- | ---: | ---: | ---: |
| 6920 | True | 0.00% | 0.00% | 0.00% |
| 5124 | True | 0.00% | 0.00% | 0.00% |
| 16281 | True | 0.00% | 100.00% | 100.00% |
| 2217 | False | 0.00% | 0.00% | 0.00% |
| 1164 | True | 100.00% | 100.00% | 100.00% |
| 4190 | False | 0.00% | 0.00% | 0.00% |
| 9882 | True | 50.00% | 50.00% | 50.00% |

## Answer quality: all 100 diagnostic questions

All 800 question/configuration records have schema-valid model judgments. Identical ordered evidence contexts reuse the same cached answer and judgment (353 unique generations and 353 unique judgments). The judge sees the full corpus gold provisions, retrieved evidence and answer, but not the retrieval configuration.

**These are automated scores. The generator and judge use the same Luna model, so correlated errors and self-preference remain. They are not independent human accuracy measurements.** The earlier local-model run had no valid answer judgments; these scores cannot establish an improvement over its unknown answer quality.

| Configuration | Reference-sufficient judgments | Automated correctness | Fully supported citations | Invalid citation indices |
| --- | ---: | ---: | ---: | ---: |
| `hybrid` | 94 / 100 | 68.09% | 73.00% | 0 |
| `hybrid_larger` | 93 / 100 | 67.74% | 76.00% | 0 |
| `hybrid_shared` | 94 / 100 | 69.15% | 76.00% | 0 |
| `citation_shared` | 94 / 100 | 69.15% | 76.00% | 0 |
| `graph_shared` | 94 / 100 | 69.15% | 76.00% | 0 |
| `graph_shared_path` | 94 / 100 | 70.21% | 78.00% | 1 |
| `graph_legacy` | 92 / 100 | 69.57% | 73.00% | 0 |
| `shuffled_shared` | 94 / 100 | 68.09% | 74.00% | 0 |

Correctness rates exclude judgments marked insufficient reference; citation-support rates include all valid judgments. Article-ID retrieval completeness alone does not guarantee that selected representative chunks contain every required clause. Raw answers, exact contexts, model reasons and citation indices are preserved.

| Answer correctness comparison | Paired questions | Difference [95% CI] |
| --- | ---: | ---: |
| `graph_shared_minus_hybrid_shared` | 94 | +0.00 pp [+0.00, +0.00] |
| `graph_shared_minus_citation_shared` | 94 | +0.00 pp [+0.00, +0.00] |
| `graph_shared_minus_shuffled_shared` | 94 | +1.06 pp [+0.00, +3.19] |

## Relational development set and reserved test set

The previously frozen 60 natural candidates were screened against full gold provisions before retrieval evaluation. Thirty development candidates were evaluated separately; no retrieval or answer evaluation was run on the 30 reserved test candidates. Luna marked 5 development questions as requiring linked provisions with sufficient references. This is a model-screened draft, not an independently validated relational benchmark. No synthetic questions were added.

The original family-separated split is unchanged. Families join equal law numbers, co-gold laws and cross-document citations; diagnostic families are excluded. Historical model exposure to the source training CSV is unknown. The category labels and draft reference answers remain review material.

| Configuration | Recall@10, all 30 development candidates | Recall@10, model-screened relational subset | Automated correctness, all 30 |
| --- | ---: | ---: | ---: |
| `hybrid` | 33.33% | 40.00% | 53.33% |
| `hybrid_larger` | 33.33% | 40.00% | 56.67% |
| `hybrid_shared` | 35.00% | 40.00% | 66.67% |
| `citation_shared` | 35.00% | 40.00% | 70.00% |
| `graph_shared` | 35.00% | 40.00% | 70.00% |
| `graph_shared_path` | 40.56% | 50.00% | 63.33% |
| `graph_legacy` | 34.44% | 40.00% | 53.33% |
| `shuffled_shared` | 35.00% | 40.00% | 70.00% |

## Automated relationship review

The prior 50-edge rule-citation sample was re-reviewed. Newly accepted LLM edges were sampled separately by kind, up to ten per kind, so rare semantic/formal types are represented. Review is blinded to prior acceptance, but still uses the same model family. Unweighted pooled support is not population precision for these stratified samples.

| Stratum | Reviewed | Model-supported | Model support rate |
| --- | ---: | ---: | ---: |
| llm_CITES | 10 | 10 | 100.00% |
| llm_PENALTY_FOR | 1 | 1 | 100.00% |
| llm_REPEALS | 2 | 2 | 100.00% |
| llm_SETS_CONDITION_FOR | 7 | 5 | 71.43% |
| llm_SPECIFIES | 10 | 8 | 80.00% |
| rule_cross_document | 25 | 24 | 96.00% |
| rule_within_document | 25 | 25 | 100.00% |

Rejected classifier proposals, failed clause checks, negative verification results and the original 30 ambiguous/rejected reference cases remain available for independent review. Model support does not establish legal validity, temporal effect or a correct amendment relationship outside the supplied evidence.

## Cost, validation and reproduction

- Copying 91,405 frozen vectors: 68.16 seconds. No document embeddings were recomputed.
- Final graph-building invocation after the copy: 2293.89 seconds; resumed invocations reuse saved results, so this is not total experiment wall time.
- Unique API requests, including preflight and all stages: 2,553.
- Reported API usage: 6,245,395 prompt tokens and 1,398,452 completion tokens.
- Provider-reported cost in saved responses: **$1.471671 USD**, reported for 2553 requests. Transport failures without a parseable response have unknown usage; this sum may differ from billing. Request usage and failed-attempt logs are archived for auditing.
- Actual response model identifiers: `openai/gpt-6-luna`.
- Graph snapshot SHA-256: `b707b16f29c972615c98af5008dd95e9d21128f78aaa7ba8857787cc3d46f240`.
- Original evidence snapshot SHA-256: `3aee6d3d36c9f3e075b7e5a1ef3a1d0c394bf3720aac5ca1daa19aff1ec57113`.
- Validation: 70 tests ran, 68 passed and two opt-in live Neo4j tests skipped; lint passed. Tests cover schema rejection, truncated responses, credential exclusion, model routing, caching and enabling semantic edges in the enriched controls.
- Final artifact audit: passed; 2,553 cached requests checked for model/effort and integrity, 2,090 structured responses revalidated, all 1,040 retrieval and 1,040 answer records present, source hashes unchanged and credential scan passed. See `final_audit.json` in the run directory.

| Configuration | Mean retrieval ms | Recovered / lost gold occurrences vs hybrid |
| --- | ---: | ---: |
| `hybrid` | 16.73 | 0 / 0 |
| `hybrid_larger` | 24.63 | 0 / 0 |
| `hybrid_shared` | 16.53 | 2 / 3 |
| `citation_shared` | 19.77 | 2 / 3 |
| `graph_shared` | 19.62 | 2 / 3 |
| `graph_shared_path` | 19.39 | 2 / 3 |
| `graph_legacy` | 19.58 | 1 / 0 |
| `shuffled_shared` | 21.52 | 2 / 3 |

Retrieval timings include hybrid lookup and graph confirmation/ranking, but exclude query embedding and answer generation. Actual candidate/context usage is preserved in each trace; matched caps do not imply identical realized costs.

Raw artifacts are in `reports/runs/openrouter_20261119` (Git-ignored): private SQLite database, copied-vector manifest, frozen protocols, every candidate/classification/verification, API request/response caches without credentials, retrieval traces, answers, judgments, benchmark screening, review packets, costs, environment and source archive. Copy this directory separately when transferring machines. The companion [JSON report](openrouter_experiment_20261119.json) contains detailed paired results and hashes.

Reproduction commands:

```bash
.venv-linux/bin/python scripts/build_openrouter_graph.py \
  --source-snapshot reports/runs/linux_20261118/coverage_v2 \
  --source-collection legal_repair_linux_20261118_v2 \
  --collection NEW_COLLECTION --output NEW_RUN --sources 1000 --pairs 1000 --workers 4
.venv-linux/bin/python scripts/evaluate_openrouter_experiment.py \
  --run NEW_RUN --corpus data/vlsp/corpus.csv \
  --review-root reports/runs/linux_20261118/review --workers 4
```

Keep the supplied key in the ignored `.env`. Existing collection names are rejected unless accompanied by this experiment’s matching copy checkpoint. Caches and per-query records permit restarting the same frozen run; mismatched protocols/code are rejected.

**Remaining limitation:** independent review of source identity, relation precision, benchmark linkage and answer correctness is still required. The 30 reserved test candidates remain untouched by retrieval/answer evaluation. The current results support only the diagnostic and explicitly model-screened development claims reported here.
