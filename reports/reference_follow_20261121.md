# Retrieving cross-document legal citations with GraphRAG

On 107 held-out Vietnamese legal citation questions, graph retrieval returned both the source article and the cited article in **105 cases (98.13%)**. Default dense+BM25 hybrid returned both in **42 cases (39.25%)**. Hybrid with the citation text and a filter for the named law returned both in **83 cases (77.57%)**. Exact article lookup returned both in all 107 cases.

## Benchmark

The question names a source law and article and quotes the part of a clause immediately before its citation. The system must retrieve the source article and identify the article it cites in another document. The two gold article IDs come from the source text and the indexed document and article identifiers. The benchmark builder reads the article text and index mappings directly. It selects examples before opening the relationship table or running retrieval.

The builder found 745 eligible spans. A fixed hash order selected 132 examples with no law number reused in either the source or target role. The first 25 were used for development and 107 were held out. The gold labels were generated from explicit citation text and verified against the stored articles. They have no independent legal annotation. The task measures retrieval of cited articles from the frozen corpus. It measures neither current legal effect nor answer correctness.

## Retrieval controls

Each configuration used the same 91,405 indexed chunks, BGE-M3 dense embeddings, BM25, a candidate cap of 60 articles, and an 18,000-character cap for the top 10 distinct articles. The mean candidate counts for the filtered hybrid and graph arms were 50.77 and 50.70.

| Configuration | Follow-up after reading the source article |
| --- | --- |
| Default hybrid | Ranks the original question with dense+BM25 reciprocal rank fusion. |
| Hybrid with citation query | Gives dense+BM25 the article and law wording from the citation. |
| Hybrid with citation query and law filter | Also filters the follow-up search to the cited law, with unfiltered hybrid backfill. |
| Graph with hybrid backfill | Uses a confirmed outgoing citation edge to retrieve the target, with the same filtered hybrid backfill. |
| Shuffled graph with hybrid backfill | Uses the same source-side edge evidence and a fixed permutation of its target endpoints. |
| Exact article lookup | Resolves the law number and article number directly against indexed article identifiers. |

All five follow-up configurations use the same source lookup and candidate allocation. The graph and shuffled arms share the same fallback. The graph arm uses the existing rule-derived citations in the frozen November 19 snapshot.

## Held-out results

| Configuration | Both articles at 10 | Article Recall@10 | Cited article at 3 |
| --- | ---: | ---: | ---: |
| Default hybrid | 42/107 (39.25%) | 68.69% | 14.02% |
| Hybrid with citation query | 40/107 (37.38%) | 68.69% | 14.02% |
| Hybrid with citation query and law filter | 83/107 (77.57%) | 88.79% | 20.56% |
| **Graph with hybrid backfill** | **105/107 (98.13%)** | **99.07%** | **88.79%** |
| Shuffled graph with hybrid backfill | 79/107 (73.83%) | 86.92% | 5.61% |
| Exact article lookup | 107/107 (100.00%) | 100.00% | 100.00% |

The graph improved complete retrieval by **20.56 percentage points** over the filtered hybrid control (paired bootstrap 95% interval **13.08 to 28.04 pp**, exact paired *p* = **4.77 × 10⁻⁷**). It recovered 22 of that control's misses without losing any of its successes. In each of the 22 gains, the logged traversed edge returned the gold target. Against shuffled links the gain was **24.30 pp** (95% interval **16.82 to 32.71 pp**).

The graph failed on two questions where traversal returned no matching edge and hybrid backfill missed the target. Exact article lookup resolved both. This makes exact lookup a strong option when a reference includes a unique law number and article number. The result supports using correctly linked citations for cross-document evidence retrieval, with broader performance measured separately on natural questions.

## Reproduction

Run from the repository root with the November 19 snapshot and Qdrant collection available. Use a new output directory for a rerun.

```bash
run=reports/runs/reference_follow_REPLAY
PYTHONPATH=src .venv-linux/bin/python scripts/run_reference_experiment.py --output "$run" prepare
PYTHONPATH=src .venv-linux/bin/python scripts/run_reference_experiment.py --output "$run" development
PYTHONPATH=src .venv-linux/bin/python scripts/run_reference_experiment.py --output "$run" held_out
```

The run directory contains the [frozen benchmark](runs/reference_follow_20261121/benchmark.json), [protocol](runs/reference_follow_20261121/protocol.json), [development summary](runs/reference_follow_20261121/development/summary.json), and [held-out summary and paired statistics](runs/reference_follow_20261121/held_out/summary.json). Individual query records include ranked article and chunk IDs, evidence budgets, and graph targets.
