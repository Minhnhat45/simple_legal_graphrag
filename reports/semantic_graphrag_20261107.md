# Vietnamese Legal GraphRAG: implementation and measured pilot

2026-11-07. The existing implementation now supports an evidence-backed provision graph and comparable M0–M4 retrieval. **This bounded pilot did not improve Article recall over hybrid retrieval.** All modes achieved Recall@10 **0.6383** across the original 100 queries and **0.9265** across the 68 fully covered queries. Keep graph expansion opt-in until broader relation coverage and held-out evaluation demonstrate a benefit.

**Existing work reused and extended.** The original chunker, stable Document/Chunk identifiers, ingestion, Ollama client, Neo4j/Qdrant stores, TF-IDF/n-gram candidate discovery, filters, answer grounding and citation validation remain in place. The standalone hybrid baseline's FastEmbed BM25 settings are shared with the new retrieval path. The legacy API remains the default. New `structure.py`, `ontology.py`, `semantic.py` and `evaluation.py` add provision structure, validation and evaluation without replacing document retrieval. `plan.md` retains its historical content with a current-status note.

The existing VLSP evaluation file and metric conventions were located in the sibling `data_for_legal_rag/VLSP_legal_data` project and reused read-only. Its historical hybrid Hit@10 of 0.95 used a different indexed corpus and chunk-level metrics; it is not directly comparable with the current filtered, Article-level pilot. Original reports were preserved.

**Authoritative data and experiment scope.** The current manifest lists 2,157 unique law IDs: 1,869 with local sources and 288 missing. The loader reads the existing 1,969 department records and 82 context files, retaining 2,051 source documents without repeating filtering. Source document IDs remain distinct where authorities reuse a law number; ambiguous references are not guessed.

The new `legal_filtered_semantic_20261107_v1` collection contains 6,707 chunks mapped to 1,863 original Article IDs across 709 documents. Existing embeddings were reused only after checking model digest and exact normalized text against the current Article. The 168 vectors whose original embedding context extended beyond that Article were rebuilt from validated current text. No stored training questions were read. A final audit confirmed all 6,707 chunk bodies belong to their mapped current Articles. This is a partial indexing pilot, not full-corpus ingestion. Fresh preparation with the improved parser yields a different 6,788-point snapshot. The comparison retains the original collection and Article mappings; its chunk audit records all source point IDs. It should be reproduced against that saved snapshot, not treated as identical to fresh ingestion.

**Graph and extraction.** `CONTAINS` represents Document → Chapter → Article → Clause → Point, skipping absent levels. IDs retain parent context; repeated headings receive occurrence suffixes, and ambiguous chunk mappings remain unresolved. The scoped graph contains 128,685 provision nodes, including 25,394 Articles; only 1,863 have searchable evidence.

Directed `PROVISION_RELATION` edges store source/target Article and chunk IDs, exact quotes from both provisions, confidence, method, model/request provenance, source hashes and temporal bounds. The ontology supports `CITES`, `AMENDS`, `ADDS_TO`, `REPEALS`, `REPLACES`, `DETAILS`, `SPECIFIES`, `SETS_CONDITION_FOR`, `EXCEPTION_TO` and `PENALTY_FOR`, retaining legacy guidance kinds. `DEFINES` and `REGULATES` connect Articles to grounded concepts, while facts capture entities, obligations, conditions, exceptions and penalties. Shared concepts discover candidates; they do not automatically establish normative relations.

Candidate discovery combines explicit references, reused sparse n-grams, stored dense vectors and previously grounded concepts. The pilot sampled 100 source Articles independently of evaluation gold, discovered 3,886 directed candidates and classified 40 pairs. Gemma `gemma4:e4b` receives short local ID aliases and evidence passages. Exact-span, endpoint, direction, temporal and referenced-clause checks precede an independent actor/action/scope verification pass. Formal legal changes require explicit operative references; similarity never establishes amendment or repeal.

The resulting graph has **245 CITES edges, one SPECIFIES edge and 54 grounded facts**, including 35 concept relations. Six citation edges cross documents; the semantic edge is within one document. The accepted semantic example connects Article 15's duty to build and approve cemetery management rules to Article 17's general requirement for such rules. Other ontology kinds are implemented but did not yield accepted edges in this sample.

Five classification responses were malformed or incomplete and failed closed. Of 15 model verification decisions, one supported an edge and 14 returned unsupported; three other proposals failed the clause-scope check. These are model-validation outcomes, not measured human relation precision. The final build took 394.7 seconds, using 26 cached classifications, 14 new classification calls and 15 verification calls; earlier diagnostic attempts are preserved separately and excluded from that timing. The local model sometimes emitted commentary despite the requested schema. A narrow final-JSON-block parser, bounded output and verifier-specific thinking settings address observed failures without repairing truncated output or accepting unsupported edges.

Temporal scope is known for 130 edges and unknown for 116. Known intervals reflect document metadata overlap, not inferred legal effect. Concept relations explicitly retain unknown temporal applicability. `as_of` gates dated legal expansion; it is not a complete historical-law resolver.

**Retrieval and comparable experiments.** The opt-in path is Qdrant dense/BM25 RRF → Article seeds → Neo4j expansion → Qdrant evidence confirmation → deduplication/reranking → Top-K Articles. It supports relation weights, confidence, depth and candidate limits. Reverse traversal retains the legal edge's original direction. All evidence is confirmed under the same metadata filters. Similarity and generic metadata do not create normative edges.

All modes use the same 100 questions, unchanged original `relevant_aids`, Top-K 10, candidate budget 60, five seeds, depth two, confidence 0.8 and graph weight 0.15. The evaluator freezes one actual hybrid result per query across all modes: an initial diagnostic run exposed changing Qdrant seed order between repeated searches. That run is retained, but the authoritative comparison below uses frozen seeds. All 100 seed-order invariants were checked.

| Mode | Hit@10 | Precision@10 | Recall@10 | MRR | nDCG@10 | F1@10 | Latency p50 / p95, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 Hybrid | 0.6600 | 0.0700 | 0.6383 | 0.5628 | 0.5688 | 0.1253 | 13.3 / 14.8 |
| M1 Structural | 0.6600 | 0.0700 | 0.6383 | 0.5626 | 0.5685 | 0.1253 | 20.9 / 27.3 |
| M2 Explicit | 0.6600 | 0.0700 | 0.6383 | 0.5628 | 0.5691 | 0.1253 | 17.6 / 21.1 |
| M3 Semantic | 0.6600 | 0.0700 | 0.6383 | 0.5628 | 0.5688 | 0.1253 | 14.3 / 16.3 |
| M4 Combined | 0.6600 | 0.0700 | 0.6383 | 0.5626 | 0.5689 | 0.1253 | 23.5 / 31.4 |

Latency includes the shared measured hybrid search plus each mode's expansion, confirmation and reranking; query embedding is measured separately (median 77.1 ms). Sparse initialization was warmed. This is one sequential local run, not a statistically established speed comparison.

**What the graph contributed.** Structural, explicit, semantic and combined modes expanded 96, 56, one and 97 queries respectively. None recovered or lost a gold Article at Top-10, and none introduced a gold Article beyond M0's candidate set. Candidate recall stayed 0.6533. Citation expansion moved already-retrieved gold Article `54193` from rank seven to five for query `15959`; this explains the small M2 nDCG change, not new evidence recovery. The 68 fully covered queries have Hit@10 0.9412 and Recall@10 0.9265 in every mode. Only ten questions have multiple gold Articles, and only one fully covered question spans documents, so this benchmark cannot establish a multi-hop or cross-document advantage.

**Validation and preservation.** The suite ran 50 tests successfully with one destructive legacy integration test skipped; additive live Neo4j checks covered two-hop traversal, direction, temporal/confidence gates, idempotence and invalidation. Ruff and whitespace checks pass. Replaying all 246 saved edges left the count at 246. Ingestion now checks collection-specific chunk counts and repairs missing vectors without clearing unchanged graph relations. Original Qdrant counts remain 358, 12,298 and 218,285; historical Thesis/E2 Neo4j document and Article counts remain unchanged. No original database was reset and no baseline result was overwritten.

**Next improvements.** First validate relation precision with a human-labeled, stratified sample, including rejected proposals and ambiguous jurisdictions. Then index more of the current corpus and broaden verified cross-document candidates; one semantic edge is insufficient to judge semantic GraphRAG. Resolve local structured-output reliability before scaling LLM calls. Create a separate multi-hop development set and calibrate ranking weights there, without tuning on these 100 evaluation questions. Temporal version resolution and semantic relation coverage remain research limitations. Current results support retaining M0 as the baseline and keeping graph modes available for controlled experiments.

The compact [machine-readable results](semantic_graphrag_20261107.json) include all metrics, coverage subsets, relation outcomes and input hashes. Full local artifacts are under `reports/runs/20261107/`: `manifest.json`, `current_chunk_audit.json`, `embedding_repair.json`, `graph_audit.json`, `final/relations.json`, verifier caches and **`frozen_seeds/evaluation.json`**. Full artifacts are ignored by Git to preserve local research data. Re-evaluate the frozen graph into a new output directory:

```bash
PYTHONPATH=src python scripts/run_experiment.py \
  --manifest reports/runs/20261107/manifest.json \
  --eval ../data_for_legal_rag/VLSP_legal_data/eval.csv \
  --output-dir reports/runs/20261107/repeat_NEW --skip-build
```

Implementation references: [Qdrant hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/), [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs), and [Neo4j MERGE](https://neo4j.com/docs/cypher-manual/current/clauses/merge/).
