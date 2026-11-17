# Plan for fixing failures and evaluating legal GraphRAG

Based on the failure analysis dated 2026-11-11 of the frozen-seed experiment dated 2026-11-07.

## Objective

Test whether verified legal relationships improve evidence completeness for questions requiring linked provisions, while measuring their effect on ordinary questions.

The recommended sequence is to repair coverage, measure whether the graph can reach missing evidence, and then test a focused graph method against a strong hybrid baseline. The current results are inconclusive about the potential of graph knowledge; they do not demonstrate a graph advantage.

No improvement is guaranteed. All reported results must come from actual experiments. Synthetic questions are acceptable when clearly labeled and independently reviewed; fabricated performance measurements are not.

## Evidence from the existing reports

Sources:

- [Failure analysis](reports/failure_analysis_20261107.md)
- [Experiment report](reports/semantic_graphrag_20261107.md)

Three findings determine the next steps:

1. **Coverage dominates.** Of 43 missed gold-article occurrences, 36 were unindexed. Neither hybrid nor graph retrieval can return evidence absent from the searchable corpus.
2. **Semantic graph coverage is too small.** The graph contains only one accepted semantic edge, and semantic expansion activated for one query. This is insufficient to assess semantic GraphRAG.
3. **Ranking alone has little room.** Hybrid candidate recall was 0.6533, versus final Recall@10 of 0.6383. Perfect reranking of that existing candidate pool could recover at most approximately 1.5 percentage points. Larger gains require better candidates.

The graph currently has 245 citation edges. Targeted citation expansion is therefore a more substantial starting point than semantic expansion.

## Experiment sequence

| Experiment | Action | What it establishes |
| --- | --- | --- |
| 1. Complete the searchable corpus | Index available articles consistently and recover missing sources where possible. Give both systems the identical corpus. | Separates ingestion failures from retrieval failures. Coverage improvements alone are not evidence of graph benefit. |
| 2. Measure graph reachability | On development data, check whether missing gold articles are reachable from actual hybrid seeds at one or two hops. Distinguish graph-node reachability from retrievable evidence. | Whether useful paths exist before tuning ranking. |
| 3. Expand precise relationships | Prioritize resolved article citations, reverse citation traversal, and explicit implementing/amending references. Add semantic relations where supported by evidence. | Whether relationships introduce relevant articles beyond hybrid candidates. |
| 4. Fix candidate scoring | Compare the current formula with a shared query–article reranker for all candidates. Test graph-path features separately. | Whether graph candidates are useful but suppressed by scoring. |
| 5. Evaluate questions requiring multiple provisions | Build an independently reviewed set covering rule plus exception, conditions plus procedure, and cross-document implementation. | Whether graph knowledge improves evidence completeness and answers on its intended task. |

### 1. Repair coverage first

Complete ingestion and article mapping for available source documents. Recover missing source documents where feasible, and record unresolved coverage gaps explicitly.

Preserve the original experiment artifacts. Create a new corpus/index snapshot and compare all retrieval modes within that snapshot. Both hybrid and graph methods must receive the same searchable evidence.

Use qids 6920, 5124, 16281 and the indexed portion of 2217 to inspect candidate-generation failures. Use 1164 and 4190 to inspect final-ranking failures, accounting for 4190's unindexed gold article. Query 9882 also exposes a final article-ranking failure. These are diagnostic cases, not an untouched test set.

### 2. Diagnose reachability before tuning weights

For each development query, record the following stages:

```text
gold indexed
  → reachable from hybrid seeds
  → added as a candidate
  → selected in top K
  → used correctly in the answer
```

If a gold article is unreachable, changing graph weights cannot recover it through the current graph. If reachable evidence is discarded, investigate candidate selection and ranking.

Gold labels may be used for this diagnostic analysis, but must not be used by the deployed retrieval method to choose seeds or expansion paths.

### 3. Implement a focused graph retrieval method

Start with the following pipeline:

```text
hybrid seeds
  → bounded citation/reference expansion
  → fetch article evidence
  → shared reranker
  → fixed answer-context budget
```

Resolve references to the correct article and document. Retain evidence, relation direction and provenance. Reverse traversal should preserve the original legal relationship direction in explanations.

Measure whether expansion adds gold articles beyond the hybrid candidate pool. Adding more graph nodes or edges is not itself evidence of retrieval improvement.

### 4. Test the scoring bottleneck

In [retrieval.py](src/legal_graphrag/retrieval.py), existing hybrid articles receive rank-based scores, while graph additions depend on cosine similarity, lexical overlap and a graph bonus. This is a plausible scoring bottleneck, not a demonstrated cause of failure.

Log component scores and gold-candidate ranks. Compare the current scoring method with a shared query–article reranker applied to every candidate, regardless of candidate source. Evaluate graph-path features as a separate ablation.

### 5. Add a separately reported relational benchmark

Keep the original benchmark. Only ten current questions have multiple gold articles, and only one fully covered question spans documents. The `multiple_chunks` label does not establish multi-hop reasoning.

Create an additional benchmark containing independently reviewed questions that require linked provisions, including:

- A general rule and an applicable exception.
- Eligibility conditions and the associated procedure.
- A provision and its implementing regulation in another document.

Include independently authored questions and failed cases, rather than only questions constructed from successful graph paths. If synthetic questions are used, label and report them separately from natural questions, and review their answers against source provisions.

## Comparison and controls

Evaluate these configurations:

1. Hybrid retrieval.
2. Hybrid retrieval with a larger retrieval pool.
3. Hybrid retrieval plus the shared reranker.
4. Hybrid retrieval plus graph expansion plus the same reranker.

Use matched candidate and answer-context budgets for the main comparison. Report any larger-pool experiment with its actual budget and cost. Keep corpus, chunking, embedding model, filters and generation settings consistent across comparable runs.

Add a shuffled-edge control. If random connections perform equally well, any improvement may come from extra candidates rather than meaningful graph relationships.

Use the existing 100 questions for diagnosis. Tune on separate development data, then freeze an untouched test set, preferably separated by document families. Do not repeatedly tune on the final test results.

## Measurements and reporting

Report:

- Candidate recall and newly recovered gold articles beyond hybrid candidates.
- Recall@10 and complete-evidence retrieval rate.
- Answer correctness and whether cited evidence supports the answer.
- Recovered and lost gold evidence relative to the baseline.
- Paired confidence intervals for performance differences.
- Retrieval latency and graph-building cost.

Report results on the full benchmark and relevant coverage subsets. Report the additional relational benchmark separately. Preserve raw predictions, configurations, input hashes and experiment artifacts so that every reported number is reproducible.

## Decision criteria

- If coverage repair helps both systems equally, attribute the gain to ingestion.
- If graph expansion cannot reach missing evidence, improve relation coverage or seed selection before tuning ranking.
- If graph expansion adds relevant candidates but they do not reach top K, investigate selection and scoring.
- If meaningful edges outperform shuffled edges under matched budgets, that supports a contribution from graph knowledge.
- If gains occur only on relational questions, make that narrower claim explicitly.
- If no reliable gain appears, report that result and the diagnosed limitations rather than claiming superiority.

The intended outcome is a reproducible test of when verified legal relationships help retrieval and answer completeness. A defensible improvement on a clearly defined task is more useful than an unsupported claim of universal superiority over hybrid RAG.
