# Graph expansion on held-out Vietnamese legal questions

GraphRAG reached **92.46% article Recall@10** on 400 natural VLSP questions, compared with **89.96% for default dense+BM25 hybrid**. The paired gain was **2.50 percentage points** (95% bootstrap interval **1.44 to 3.75 pp**). The graph retrieved every gold article for **358 of 400 questions**, compared with **341** for default hybrid and **348** for the tuned hybrid controls.

The separate citation experiment tested the graph's ability to follow an explicit cross-document reference. It recovered both articles for 98.13% of tasks, compared with 77.57% for hybrid with citation text and a target-law filter. [Cross-document citation experiment](reference_follow_20261121.md)

## Benchmark and scoring

The 130 development questions comprise the original 100 diagnostics and the 30 previous relational development questions. The new 400-question test set comes from existing VLSP training questions whose gold articles all have validated mappings into the index. The deterministic selection excludes shared gold law numbers with development and the earlier 30 reserved questions. It removes duplicate questions and permits each gold law to occur only once in the new test set. The selection prioritizes questions with multiple gold articles, which account for 57 of the 400. The original questions and gold IDs define relevance. The test is held out from this experiment's ranking calibration. Its sample comes from VLSP training data, with historical model exposure unknown.

Both methods use the November 19 Qdrant collection with 91,405 chunks and the same BGE-M3 and BM25 configuration. Each receives the same initial hybrid result per question. All controls have a 60-article candidate cap, return at most 10 distinct articles, and use an 18,000-character cap for returned evidence. The graph and document-scoped arms may add up to 24 challenger articles within that candidate cap.

The graph expands from the first 20 distinct hybrid articles. Its links include nearby articles in the same document, the frozen graph's citation and typed relationships at confidence 0.8 or above, and cross-law citation spans resolved from source text. The latter pass found 1,145 spans, some connecting nodes already linked by the original graph. The graph ranks challengers by query relevance and uses path support in final ranking. The shuffled control uses the graph's calibrated settings with permuted endpoints. Hybrid reranking and document-scoped hybrid tune their corresponding ranking parameters on the same development questions. The parameter files were written before either held-out evaluation.

## Results

| Configuration | Recall@10 | Full gold set at 10 | Candidate recall |
| --- | ---: | ---: | ---: |
| Default hybrid | 89.96% | 85.25% | 96.44% |
| Hybrid with reranking | 91.27% | 87.00% | 96.44% |
| Document-scoped hybrid | 91.27% | 87.00% | 96.81% |
| Graph with shuffled endpoints | 91.27% | 87.00% | 95.81% |
| **Graph** | **92.46%** | **89.50%** | **97.25%** |

Compared with default hybrid, the graph improved Recall@10 on 19 questions and tied on 381. It recovered 19 gold article occurrences, including two that were outside the initial hybrid candidate pool. The other 17 came from reranking candidates already in the pool. It lost zero of default hybrid's retrieved gold occurrences on this sample.

Compared with tuned hybrid, document-scoped hybrid, and shuffled links, the graph's lead on the primary Recall@10 metric was **1.19 pp** (95% interval **−0.19 to 2.56 pp**). On the secondary full-evidence metric, the graph led each control by **2.50 pp** (exploratory 95% interval **0.75 to 4.50 pp**). It added complete evidence on 13 questions and lost it on three versus each control.

On the **57 multi-evidence questions**, graph Recall@10 was **78.65%**, versus **64.62%** for default hybrid and **68.57%** for tuned hybrid. The exploratory paired gain over tuned hybrid was **10.09 pp** (95% interval **5.26 to 15.35 pp**). On the 343 single-evidence questions the graph reached **94.75%**, compared with **95.04%** for tuned hybrid. The multi-evidence subgroup is defined by the number of original gold articles. It can include questions that do not require linking those articles.

On the previously reserved 30 questions, graph Recall@10 was **53.33%**, compared with **47.22%** for default hybrid and **45.56%** for tuned hybrid. The intervals were **−0.56 to 13.89 pp** and **1.67 to 14.44 pp**, respectively. Only 12 of these questions have mappings for every gold article. Their benchmark status remains pending independent review.

## Reproduction

Run from the repository root with the November 19 snapshot and Qdrant collection available. Use a new output directory for a rerun.

```bash
run=reports/runs/natural_graph_REPLAY
PYTHONPATH=src .venv-linux/bin/python scripts/run_natural_graph_experiment.py prepare --output "$run"
PYTHONPATH=src .venv-linux/bin/python scripts/run_natural_graph_experiment.py known_development --output "$run"
PYTHONPATH=src .venv-linux/bin/python scripts/run_natural_graph_experiment.py held_out --output "$run"
PYTHONPATH=src .venv-linux/bin/python scripts/run_natural_graph_experiment.py legacy_reserved --output "$run"
```

The run directory contains the [benchmark](runs/natural_graph_20261121/benchmark.json), [protocol](runs/natural_graph_20261121/protocol.json), [frozen ranking parameters](runs/natural_graph_20261121/ranking.json), [400-question results](runs/natural_graph_20261121/held_out/summary.json), and [30-question results](runs/natural_graph_20261121/legacy_reserved/summary.json). Per-query files record every scored candidate. The audit re-scored all 560 question records and checked the benchmark and script hashes, top-10 limits, candidate caps, and context caps.
