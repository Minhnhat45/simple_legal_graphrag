# Linux handoff: resume the failure-fixing experiment

**OpenRouter follow-up, 2026-11-19:** the rerun using `openai/gpt-6-luna` with medium reasoning completed in separate graph and vector stores. See [the OpenRouter report](openrouter_experiment_20261119.md) and [machine-readable results](openrouter_experiment_20261119.json). All 1,000 candidate relation pairs, 1,040 retrieval records, 1,040 answer judgments and 80 automated relation reviews are complete. Final artifact validation passed. Shared graph retrieval did not outperform shared hybrid retrieval; new semantic relations did not activate on the diagnostic questions. Independent review and retrieval/answer evaluation of the reserved 30-question test set remain pending. No experiment jobs remain running. Earlier handoff notes below are retained for provenance.

**Linux follow-up, 2026-11-18:** the 91,405-chunk index and all 700 diagnostic retrieval records are complete. See [the result report](fixing_failure_linux_20261118.md) and [machine-readable results](fixing_failure_linux_20261118.json). Answer/relationship evaluation attempts failed model-output validation; their raw outputs are preserved. Independent review and the relational test benchmark remain pending. No experiment jobs remain running. The original handoff below is retained for provenance.

Prepared 2026-11-13. Continue [plan_fixing_failure_11_11.md](../plan_fixing_failure_11_11.md) on Linux. The Mac work is wrapped up; the full embedding/retrieval experiment has **not** been run. See [the execution report](fixing_failure_20261113.md) for measured results and limitations.

## 1. Current state

Completed:

- Fixed article boundaries after uppercase chapter/section titles and persisted explicit chunk-to-article ownership.
- Built a complete staging index of the articles parsed from available source documents.
- Added shared query/article scoring, a separate graph-feature ablation, fixed context budgets and component/path traces.
- Added a seven-configuration experiment runner with citation expansion, shuffled-edge control, reachability diagnostics, coverage subsets and paired bootstrap intervals.
- Created identity-mapping and relational-benchmark review queues.
- Ran 61 tests: 59 passed, two live Neo4j checks skipped. Lint passed.

Final staging snapshot: `reports/runs/20261113/coverage_final/`.

| Item | Result |
| --- | ---: |
| Documents | 2,051 |
| Parsed articles | 55,898 |
| Chunks to embed | 91,405 |
| Unambiguously mapped original article IDs | 31,307 |
| Automatically resolved citations | 19,674 |
| Fully mapped diagnostic questions | 47 / 100 |
| Mapped gold occurrences | 53 / 113 |
| Documents without parsed articles | 46 |

This is a SQLite FTS staging index, **not an existing Qdrant hybrid index**. No new retrieval accuracy, answer correctness or graph advantage has been measured. The earlier 80,865-chunk estimate referred to the initial snapshot; use **91,405** for the final snapshot.

## 2. Transfer the code and artifacts

The work is currently uncommitted. Moving only the previously committed repository will omit the changes. Transfer or commit the intended changes yourself, including these files:

```text
plan_fixing_failure_11_11.md
src/legal_graphrag/chunking.py
src/legal_graphrag/ontology.py
src/legal_graphrag/qdrant_store.py
src/legal_graphrag/retrieval.py
src/legal_graphrag/experiment.py
scripts/repair_coverage.py
scripts/audit_mapping_gaps.py
scripts/run_repaired_experiment.py
tests/test_failure_plan.py
tests/test_qdrant_store.py
reports/fixing_failure_20261113.md
reports/fixing_failure_20261113.json
reports/mapping_review_final_20261113.json
reports/relational_review_queue_20261113.json
reports/linux_experiment_handoff_20261113.md
```

Also retain the existing source, tests, dependency files and both previous reports with their JSON companions. The new scripts depend on that existing code.

**The datasets under `data/` and the artifacts under `reports/runs/` are Git-ignored. Copy the required research data separately.** See the [data inventory](../data/README.md) for the local VLSP files and frozen evaluation inputs. To reuse the exact final snapshot, transfer its entire directory, including `snapshot.sqlite`, `coverage.json` and `mapping_gaps.json`. The database is approximately 699 MiB. Keep the other `reports/runs/20261113/` logs and intermediate snapshots for provenance if space permits.

An optional archive command, run from the Mac repository root:

```bash
tar -czf /tmp/legal_graphrag_snapshot_20261113.tar.gz \
  reports/runs/20261113/coverage_final \
  reports/runs/20261113/environment.txt \
  reports/runs/20261113/tests.log \
  reports/runs/20261113/lint.log \
  reports/runs/20261113/recovered_services.json
```

This archive contains research artifacts only; it does not transfer the code. Extract it from the Linux repository root after copying it. Do not transfer the Mac `.venv`; create a Linux environment.

If rebuilding instead, use the filtered `data/` directory, `data/vlsp/corpus.csv`, and `reports/failure_analysis_20261107.json`.

## 3. Recover and verify the original Linux experiment

The previous reports recorded this repository at:

```text
/home/minhnhat/Projects/simple_legal_graphrag/
```

Look for these original artifacts on Linux before changing mappings:

```text
reports/runs/20261107/manifest.json
reports/runs/20261107/current_chunk_audit.json
reports/runs/20261107/embedding_repair.json
reports/runs/20261107/graph_audit.json
reports/runs/20261107/frozen_seeds/evaluation.json
reports/runs/20261107/final/relations.json
data/vlsp/eval.csv
data/vlsp/corpus.csv
```

The original Qdrant collection was `legal_filtered_semantic_20261107_v1`. Preserve it and the original Neo4j graph. The Mac had a different collection; it did not contain this frozen experiment. Do not reset databases or recreate an existing collection.

Verify available files with `sha256sum` and these recorded hashes:

| File | Expected SHA-256 |
| --- | --- |
| Original corpus CSV | `79bf936ef8de4208beaf13e932075b12e6b9d67043311d602bf386b66ffe2edc` |
| Original eval CSV | `5768fcf9760582426862b1f35df3160c682f55b7a96ed8da269dc07de4c8cab7` |
| Original manifest | `cc848f27d83cecbda2fe4f0ad09ea6f0bf9d671f6d037e1041a9954b36ea9b8e` |
| Frozen evaluation JSON | `56f7222ea6f54a26bbee7c8954063a263626d3c1d3b3f615705fed7ad0a53f33` |
| Copied final SQLite snapshot | `01bfa48936fd223120fbf9894d0983b64f367c81ef55417d585ff2b2dad6bb49` |

A freshly rebuilt SQLite database need not have the same byte hash; record its new hash and compare the source/document hashes, counts and mappings. An exact copied snapshot must match. Absolute Mac paths in its metadata are provenance, not Linux paths to open: the new runner reads files relative to `--snapshot`.

The new runner gets its 100 diagnostic questions and gold IDs from `coverage.json`, derived from the saved failure analysis. Use `data/vlsp/eval.csv` to verify the derived input or reproduce the original experiment.

## 4. Prepare Linux and check services

From the Linux repository root, using Python 3.11 or later:

```bash
python3 -m venv .venv-linux
.venv-linux/bin/python -m pip install -r requirements.txt
.venv-linux/bin/python -m unittest discover -s tests -v
```

`reports/runs/20261113/environment.txt` records the Mac package versions for comparison; it is not a portable Linux lock file. Preserve the Linux package versions in the new run artifacts.

Configure the local `.env` or exported settings for the actual Linux services. Check them before starting new containers:

```bash
curl --fail http://localhost:6333/collections
curl --fail http://localhost:11434/api/tags
```

Use the actual host URLs if they differ. The new runner uses `QDRANT_URL`, `QDRANT_API_KEY` when required, and `OLLAMA_BASE_URL` through `Settings.from_env()`. It explicitly selects hybrid indexing and `bge-m3:latest`. The model must be available in Ollama; install it with `ollama pull bge-m3` if absent.

The new runner uses the citation graph saved in SQLite, so **Neo4j is not required for this new comparison**. Neo4j is needed for the original `run_experiment.py` workflow and its live integration tests. Gemma is not needed for the new retrieval-only comparison; answer generation and semantic verification are separate unfinished stages.

The Mac embedding digest was:

```text
7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab
```

The runner records the Linux model digest. If it differs, report the change and do not describe cross-environment results as using identical embeddings. Within a new comparison all configurations share the same indexed vectors.

## 5. Resolve identity coverage before interpreting performance

The original pilot had 68 fully covered questions using validated stored chunk matches. The new audit has 47 under a stricter full-body mapping procedure. That difference is unresolved, not a measured retrieval regression.

Review:

- `reports/runs/20261113/coverage_final/coverage.json`: per-question coverage and accepted mappings.
- `reports/runs/20261113/coverage_final/mapping_gaps.json`: every rejected or unresolved corpus mapping.
- `reports/mapping_review_final_20261113.json`: diagnostic similarity suggestions, **not accepted mappings**.
- The original manifest and chunk audit, if recovered on Linux.

Inspect qids 6920, 5124, 16281, 2217, 1164, 4190 and 9882. In the final Mac snapshot, 16978 and 17419 for qid 2217, 28492 for qid 4190, and 56079 for qid 9882 remain unresolved. Account for duplicated source records, authority/version differences, article boundaries and text differences. Do not force identities from article ordinals or similarity, change gold labels to improve scores, or silently merge different legal documents.

If an identity/parser fix is justified, implement and test it, then rebuild into a **new** directory. Keep the existing snapshot and its hash for reproduction. Pass the in-project corpus explicitly to the coverage builder because the Luna run pins that script's hash.

```bash
.venv-linux/bin/python scripts/repair_coverage.py \
  --dataset data \
  --corpus data/vlsp/corpus.csv \
  --analysis reports/failure_analysis_20261107.json \
  --output reports/runs/linux_coverage_review_v1

.venv-linux/bin/python scripts/audit_mapping_gaps.py \
  --snapshot reports/runs/linux_coverage_review_v1 \
  --corpus data/vlsp/corpus.csv \
  --output reports/mapping_review_linux_v1.json
```

Both outputs must be new. The coverage builder checks the corpus against the failure-analysis hash. Record unresolved gaps explicitly even if proceeding with a diagnostic run.

To regenerate the Linux review inputs, including the relational split, use the
frozen Linux snapshot and both local VLSP source files:

```bash
.venv-linux/bin/python scripts/prepare_linux_review.py \
  --snapshot reports/runs/linux_20261118/coverage_v2 \
  --corpus data/vlsp/corpus.csv --train data/vlsp/train.csv \
  --output reports/runs/linux_review_rebuilt
```

## 6. Run the full diagnostic comparison

Choose either the copied final snapshot or a newly reviewed snapshot. The command below uses the copied one. Run inside a persistent terminal session such as `tmux` because indexing can take a long time.

```bash
set -o pipefail
RUN_TAG="linux_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="reports/runs/${RUN_TAG}"
mkdir -p "$RUN_ROOT"
.venv-linux/bin/python -m pip freeze > "$RUN_ROOT/environment.txt"
git rev-parse HEAD > "$RUN_ROOT/base_commit.txt"
git diff > "$RUN_ROOT/tracked_changes.patch"

.venv-linux/bin/python -u scripts/run_repaired_experiment.py \
  --snapshot reports/runs/20261113/coverage_final \
  --collection "legal_repair_${RUN_TAG}" \
  --output "$RUN_ROOT/comparison" \
  2>&1 | tee "$RUN_ROOT/run.log"
```

The commit and patch record does not capture untracked files; preserve the transferred new scripts and modules too, or commit the intended changes before running.

The runner embeds every chunk, creates one new hybrid Qdrant collection, checks the indexed count, and evaluates:

| Configuration | Candidate cap |
| --- | ---: |
| `hybrid` | 60 |
| `hybrid_larger` | 120 |
| `hybrid_shared` | 60 |
| `graph_shared` | 60 |
| `graph_shared_path` | 60 |
| `graph_legacy` | 60 |
| `shuffled_shared` | 60 |

Top-K is 10 and the context cap is 18,000 characters. Actual hybrid results are frozen across configurations sharing a pool size. Shared scoring is a cosine-plus-lexical feature formula, not a trained cross-encoder. Randomized links are a candidate-generation control and are not legal evidence.

The Mac probe took 4.37 seconds for 64 chunks, giving an estimate of roughly 104 minutes for 91,405 embeddings alone. Linux runtime may differ substantially; measure it rather than carrying over this estimate as a result.

**Restart limitation:** the current runner refuses existing collections and output directories; it does not implement index resumption. A failed or interrupted run may leave a partial new collection. Preserve its logs, inspect the failure, and use fresh names for a new attempt. If avoiding re-embedding becomes necessary, implement and validate a resume path with snapshot/model/configuration checks first. Do not bypass the existing-collection guard casually.

## 7. Inspect results and finish the remaining research stages

Expected successful outputs:

```text
reports/runs/<RUN_TAG>/comparison/predictions.jsonl
reports/runs/<RUN_TAG>/comparison/evaluation.json
reports/runs/<RUN_TAG>/run.log
reports/runs/<RUN_TAG>/environment.txt
```

`predictions.jsonl` is appended after each completed question; for the current 100 questions and seven configurations a successful run should have 700 records. A file with fewer records is partial. `evaluation.json` is written only after all configurations and questions complete.

Review candidate recall, newly added gold evidence, Recall@10, complete-evidence rate, recovered/lost gold IDs, latency, paired confidence intervals, and the full versus fully covered subsets. Inspect one/two-hop reachability and score/path traces for the diagnostic qids. Article-ID completeness does not prove that the selected representative chunks contain every needed clause or that an answer is correct.

Still outside the runner:

1. Independently review relation precision, including ambiguous and rejected cases. Add implementing/amending or semantic relationships only with supporting evidence.
2. Build a separately reported relational benchmark. `reports/relational_review_queue_20261113.json` contains 42 existing natural diagnostic questions awaiting review, not a finished benchmark. Add independently authored rule/exception, condition/procedure and cross-document cases, including failures. Label synthetic questions separately if added.
3. Freeze development and untouched test sets, preferably separated by document families. Do not tune on the existing 100 diagnostic questions or repeatedly on the final test set.
4. Evaluate generated answers and citation support with consistent generation settings and context budgets. The runner currently records answer correctness and citation support as `null`.
5. Write a new dated result report with actual measurements, input/model/code hashes, budgets, costs, confidence intervals and explicit limitations. Attribute corpus gains to ingestion; claim a graph contribution only if the controlled evidence supports it.

Preserve the original 2026-11-07 artifacts and the Mac 2026-11-13 snapshots throughout. Copy raw run directories separately when moving machines: they remain Git-ignored.
