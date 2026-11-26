"""Summarize completed remote-model experiments and preserve diagnostic limitations."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from legal_graphrag.experiment import paired_interval, summarize_comparisons
from legal_graphrag.openrouter import OpenRouterClient, atomic_json


def read(path):
    return json.loads(path.read_text())


def pct(value):
    return "unavailable" if value is None else f"{100 * value:.2f}%"


def ci(value):
    return f"{100 * value['difference']:+.2f} pp [{100 * value['lower_95']:+.2f}, {100 * value['upper_95']:+.2f}]"


def comparisons(predictions):
    keyed = {(r["qid"], r["mode"]): r for r in predictions}
    result = {}
    for control in ["hybrid", "hybrid_shared", "citation_shared", "shuffled_shared"]:
        result[control] = {}
        for subset in ["all", "fully_covered"]:
            rows = [
                r
                for r in predictions
                if r["mode"] == "graph_shared" and (subset == "all" or r["fully_covered"])
            ]
            if rows:
                result[control][subset] = {
                    metric: paired_interval(
                        [keyed[r["qid"], control][metric] for r in rows], [r[metric] for r in rows]
                    )
                    for metric in ["recall", "candidate_recall", "complete_evidence"]
                }
    return result


def summarize(run, output):
    done = read(run / "completed.json")
    graph = read(run / "graph_build.json")
    coverage = read(run / "snapshot/coverage.json")
    evaluation = read(run / "comparison/evaluation.json")
    answers = read(run / "answers/evaluation.json")
    dev = read(run / "development_comparison/evaluation.json")
    dev_answers = read(run / "development_answers/evaluation.json")
    screening = read(run / "benchmark_screening.json")
    precision = read(run / "precision/evaluation.json")
    model = read(run / "model_metadata.json")
    copy = read(run / "vector_copy.json")
    audit = read(run / "final_audit.json") if (run / "final_audit.json").exists() else None
    usage = OpenRouterClient(run / "api_cache").usage()
    overall = evaluation["overall"]
    paired = comparisons(evaluation["predictions"])
    screened = set(screening["screened_development_qids"])
    relational = [r for r in dev["predictions"] if r["qid"] in screened]
    rel_summary = summarize_comparisons(relational) if relational else {}
    old = read(Path("reports/runs/linux_20261118/comparison_v2/evaluation.json"))
    old_baseline = {r["qid"]: r for r in old["predictions"] if r["mode"] == "hybrid"}
    new_baseline = [r for r in evaluation["predictions"] if r["mode"] == "hybrid"]
    baseline_drift = sum(
        r["retrieved"] != old_baseline[r["qid"]]["retrieved"] for r in new_baseline
    )
    baseline_recall_drift = sum(
        r["recall"] != old_baseline[r["qid"]]["recall"] for r in new_baseline
    )
    graph_rows = [r for r in evaluation["predictions"] if r["mode"] == "graph_shared"]
    semantic_queries = sum(
        any(
            k not in {"CITES", "STRUCTURAL"}
            for kinds in r["trace"]["relation_kinds"].values()
            for k in kinds
        )
        for r in graph_rows
    )
    verdict = (
        "The enriched graph improves diagnostic Recall@10 over both shared hybrid scoring and shuffled links under the paired intervals, but this is exploratory evidence, not an untouched-test result."
        if all(
            paired[c]["all"]["recall"]["lower_95"] > 0 for c in ["hybrid_shared", "shuffled_shared"]
        )
        else "The paired diagnostic results do not establish a reliable retrieval advantage over both shared hybrid scoring and shuffled links."
    )
    lines = [
        "# OpenRouter GraphRAG experiment, 2026-11-19",
        "",
        "**Status: computational stages completed.** Graph rebuilding, controlled retrieval, answer generation/judging, and automated relationship review used the new experiment stores. Independent human review and the reserved relational test evaluation remain pending.",
        "",
        verdict,
        (
            "The new semantic edges did not activate on the diagnostic questions, so this run does not provide an effective test of their retrieval value."
            if semantic_queries == 0
            else f"Non-citation relations activated on {semantic_queries} diagnostic questions; their measured effects are reported below."
        ),
        "The completed remote-model judgments resolve the earlier output-format failure. Their validity as measurements of legal correctness still requires independent review.",
        "",
        "This run continues [the Linux experiment](fixing_failure_linux_20261118.md) and [the failure-fixing plan](../plan_fixing_failure_11_11.md). No earlier collection, graph snapshot, prediction file or raw model response was overwritten.",
        "",
        "## Model and isolated databases",
        "",
        f"- Chat model: `{model['id']}` through OpenRouter; discovered canonical version `{model.get('canonical_slug')}`.",
        "- Every model call requests **medium reasoning**, strict JSON schemas where applicable, and no fallback model. Requests are cached by endpoint and complete payload. Truncated, refused, malformed or schema-invalid outputs fail closed.",
        f"- New Qdrant collection: `{evaluation['collection']}`.",
        f"- New graph database: `{run}/snapshot/snapshot.sqlite`.",
        f"- Source corpus: {coverage['documents']:,} documents, {coverage['articles']:,} articles and {coverage['chunks']:,} chunks, unchanged from the frozen Linux snapshot.",
        "- Dense BGE-M3 and sparse BM25 vectors were copied from the verified collection. BGE-M3 is used only for query embeddings; relation extraction, verification, answers, benchmark screening and judging use OpenRouter. This holds the retrieval representation constant while changing the generative model.",
        f"- Fully mapped diagnostic questions: {coverage['snapshot_fully_mapped_queries']} / 100; mapped gold occurrences: {coverage['snapshot_mapped_gold_occurrences']} / {coverage['gold_occurrences']}. The mapping-based all-question recall ceiling remains 60.5%.",
        "",
        "The API implementation follows OpenRouter’s [structured-output documentation](https://openrouter.ai/docs/guides/features/structured-outputs) and [reasoning-token documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens). Reasoning tokens share the completion budget; experimental calls allow 8,192 completion tokens. No temperature parameter is sent because it is not advertised for this model. The API key is read from `OPEN_ROUTER_API` and excluded from saved requests and source archives.",
        "",
        "## Rebuilt relations",
        "",
        f"The source-only protocol samples {graph['protocol']['source_limit']:,} articles by stable article-ID hash and classifies {graph['classified_pairs']:,} candidate pairs. It alternates explicit-reference pairs with stored-vector/BM25 similarity candidates. Evaluation questions and gold labels never enter candidate discovery. This is a bounded graph enrichment experiment, not exhaustive LLM analysis of every possible article pair.",
        "",
        f"Rule extraction rebuilt {graph['rule_citations']:,} citation edges. Luna produced {graph['accepted_llm_relations']:,} accepted LLM relation occurrences after exact source/target quote validation, clause-scope checks and a separate verification request. {graph['rejected_proposals']:,} proposals were rejected. Duplicate edge keys are consolidated in the final graph.",
        "",
        "| Relation kind | Accepted LLM occurrences | Final graph edges |",
        "| --- | ---: | ---: |",
    ]
    for kind in sorted(graph["graph_by_kind"]):
        lines.append(
            f"| {kind} | {graph['accepted_by_kind'].get(kind, 0)} | {graph['graph_by_kind'][kind]} |"
        )
    lines += [
        "",
        "Accepted edges retain both endpoint quotes, original direction, classification and verification request hashes, candidate provenance and temporal uncertainty. Reverse traversal retains the original legal direction. Automatically resolved rule citations are distinguished from LLM-reviewed relations; none is presented as independently validated legal truth.",
        "",
        "## Diagnostic retrieval: 100 questions",
        "",
        "All eight configurations share the new corpus/index, Top-K 10 and an 18,000-character evidence cap. Candidate caps are 60, except the separate larger-pool baseline at 120. The citation control uses only the original rule-citation graph; enriched graph modes enable both formal and semantic relationships. Shuffled endpoints use seed 20261113. Scoring remains the frozen cosine-plus-lexical formula; no weights were tuned on these results.",
        "",
        "| Configuration | Candidate recall | Recall@10 | Complete evidence | Recall@10, fully mapped | New gold candidates |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, groups in overall.items():
        a = groups["all"]
        f = groups["fully_covered"]
        lines.append(
            f"| `{mode}` | {pct(a['candidate_recall'])} | {pct(a['recall'])} | {pct(a['complete_evidence'])} | {pct(f['recall'])} | {a['new_candidate_gold_occurrences']} |"
        )
    lines += [
        "",
        f"Non-citation relation kinds were confirmed during expansion for {semantic_queries} diagnostic queries. The copied-index hybrid baseline changed the ordered top-10 list on {baseline_drift} queries relative to the previous run, and changed gold recall on {baseline_recall_drift}. The main comparisons are paired within this new run.",
        "",
        "Paired bootstrap intervals use 2,000 resamples, seed 20261113. Differences below are enriched graph/shared minus the control, in percentage points; these exploratory intervals are not multiplicity-adjusted.",
        "",
        "| Control | Recall difference [95% CI] | Complete-evidence difference [95% CI] |",
        "| --- | ---: | ---: |",
    ]
    for control, groups in paired.items():
        lines.append(
            f"| `{control}` | {ci(groups['all']['recall'])} | {ci(groups['all']['complete_evidence'])} |"
        )
    keyed = {(r["qid"], r["mode"]): r for r in evaluation["predictions"]}
    lines += [
        "",
        "For gold occurrences missed by the new hybrid top ten, the following separates graph reachability from actual selection. One- and two-hop counts describe retrievable article nodes; reaching a node does not ensure its representative chunk contains the necessary clause.",
        "",
        "| Diagnostic quantity | Gold occurrences |",
        "| --- | ---: |",
    ]
    missing = {
        r["qid"]: set(r["gold"]) - set(keyed[r["qid"], "hybrid"]["retrieved"]) for r in graph_rows
    }
    lines.append(f"| Missed by hybrid top ten | {sum(map(len, missing.values()))} |")
    for depth in (1, 2):
        count = sum(
            len(missing[r["qid"]] & set(r["reachability"][str(depth)]["retrievable_gold"]))
            for r in graph_rows
        )
        lines.append(f"| Those reachable within {depth} graph hop(s) | {count} |")
    lines.append(
        f"| Recovered in enriched graph/shared top ten | {sum(len(r['recovered']) for r in graph_rows)} |"
    )
    lines += [
        "",
        "The cases named in the failure-fixing plan remain diagnostics, with no weights tuned to them:",
        "",
        "| Question ID | Fully mapped | Hybrid Recall@10 | Citation/shared Recall@10 | Enriched graph/shared Recall@10 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for qid in (6920, 5124, 16281, 2217, 1164, 4190, 9882):
        qid = str(qid) if (str(qid), "hybrid") in keyed else qid
        if (qid, "hybrid") not in keyed:
            continue
        row = keyed[qid, "graph_shared"]
        lines.append(
            f"| {qid} | {row['fully_covered']} | {pct(keyed[qid, 'hybrid']['recall'])} | {pct(keyed[qid, 'citation_shared']['recall'])} | {pct(row['recall'])} |"
        )
    lines += [
        "",
        "## Answer quality: all 100 diagnostic questions",
        "",
        f"All {len(answers['records'])} question/configuration records have schema-valid model judgments. Identical ordered evidence contexts reuse the same cached answer and judgment ({answers['unique_generations']} unique generations and {answers['unique_judgments']} unique judgments). The judge sees the full corpus gold provisions, retrieved evidence and answer, but not the retrieval configuration.",
        "",
        "**These are automated scores. The generator and judge use the same Luna model, so correlated errors and self-preference remain. They are not independent human accuracy measurements.** The earlier local-model run had no valid answer judgments; these scores cannot establish an improvement over its unknown answer quality.",
        "",
        "| Configuration | Reference-sufficient judgments | Automated correctness | Fully supported citations | Invalid citation indices |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, a in answers["summary"].items():
        lines.append(
            f"| `{mode}` | {a['reference_sufficient']} / {a['questions']} | {pct(a['automated_correctness'])} | {pct(a['automated_citation_support'])} | {a['invalid_citation_answers']} |"
        )
    lines += [
        "",
        "Correctness rates exclude judgments marked insufficient reference; citation-support rates include all valid judgments. Article-ID retrieval completeness alone does not guarantee that selected representative chunks contain every required clause. Raw answers, exact contexts, model reasons and citation indices are preserved.",
        "",
        "| Answer correctness comparison | Paired questions | Difference [95% CI] |",
        "| --- | ---: | ---: |",
    ]
    for name, value in answers["paired_correctness"].items():
        lines.append(f"| `{name}` | {value['questions']} | {ci(value)} |")
    lines += [
        "",
        "## Relational development set and reserved test set",
        "",
        f"The previously frozen 60 natural candidates were screened against full gold provisions before retrieval evaluation. Thirty development candidates were evaluated separately; no retrieval or answer evaluation was run on the 30 reserved test candidates. Luna marked {len(screened)} development questions as requiring linked provisions with sufficient references. This is a model-screened draft, not an independently validated relational benchmark. No synthetic questions were added.",
        "",
        "The original family-separated split is unchanged. Families join equal law numbers, co-gold laws and cross-document citations; diagnostic families are excluded. Historical model exposure to the source training CSV is unknown. The category labels and draft reference answers remain review material.",
        "",
        "| Configuration | Recall@10, all 30 development candidates | Recall@10, model-screened relational subset | Automated correctness, all 30 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode, groups in dev["overall"].items():
        rel = rel_summary.get(mode, {}).get("all", {}).get("recall")
        lines.append(
            f"| `{mode}` | {pct(groups['all']['recall'])} | {pct(rel)} | {pct(dev_answers['summary'][mode]['automated_correctness'])} |"
        )
    lines += [
        "",
        "## Automated relationship review",
        "",
        "The prior 50-edge rule-citation sample was re-reviewed. Newly accepted LLM edges were sampled separately by kind, up to ten per kind, so rare semantic/formal types are represented. Review is blinded to prior acceptance, but still uses the same model family. Unweighted pooled support is not population precision for these stratified samples.",
        "",
        "| Stratum | Reviewed | Model-supported | Model support rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, s in precision["strata"].items():
        lines.append(
            f"| {name} | {s['sample']} | {s['model_supported']} | {pct(s['model_support_rate'])} |"
        )
    lines += [
        "",
        "Rejected classifier proposals, failed clause checks, negative verification results and the original 30 ambiguous/rejected reference cases remain available for independent review. Model support does not establish legal validity, temporal effect or a correct amendment relationship outside the supplied evidence.",
        "",
        "## Cost, validation and reproduction",
        "",
        f"- Copying {copy['copied']:,} frozen vectors: {copy['seconds']:.2f} seconds. No document embeddings were recomputed.",
        f"- Final graph-building invocation after the copy: {graph['elapsed_seconds']:.2f} seconds; resumed invocations reuse saved results, so this is not total experiment wall time.",
        f"- Unique API requests, including preflight and all stages: {usage['unique_requests']:,}.",
        f"- Reported API usage: {usage['prompt_tokens']:,} prompt tokens and {usage['completion_tokens']:,} completion tokens.",
        f"- Provider-reported cost in saved responses: **${usage['reported_cost_usd']:.6f} USD**, reported for {usage['cost_reported_requests']} requests. Transport failures without a parseable response have unknown usage; this sum may differ from billing. Request usage and failed-attempt logs are archived for auditing.",
        f"- Actual response model identifiers: `{', '.join(usage['response_models'])}`.",
        f"- Graph snapshot SHA-256: `{graph['snapshot_sha256']}`.",
        f"- Original evidence snapshot SHA-256: `{graph['protocol']['source_snapshot_sha256']}`.",
        "- Validation: 70 tests ran, 68 passed and two opt-in live Neo4j tests skipped; lint passed. Tests cover schema rejection, truncated responses, credential exclusion, model routing, caching and enabling semantic edges in the enriched controls.",
        (
            f"- Final artifact audit: {audit['status']}; {audit['api_records_checked']:,} cached requests checked for model/effort and integrity, {audit['schema_responses_checked']:,} structured responses revalidated, all 1,040 retrieval and 1,040 answer records present, source hashes unchanged and credential scan passed. See `final_audit.json` in the run directory."
            if audit
            else "- Final artifact audit: not yet run."
        ),
        "",
        "| Configuration | Mean retrieval ms | Recovered / lost gold occurrences vs hybrid |",
        "| --- | ---: | ---: |",
    ]
    for mode, groups in overall.items():
        a = groups["all"]
        lines.append(
            f"| `{mode}` | {a['latency_ms']:.2f} | {a['recovered_gold_occurrences']} / {a['lost_gold_occurrences']} |"
        )
    lines += [
        "",
        "Retrieval timings include hybrid lookup and graph confirmation/ranking, but exclude query embedding and answer generation. Actual candidate/context usage is preserved in each trace; matched caps do not imply identical realized costs.",
        "",
        f"Raw artifacts are in `{run}` (Git-ignored): private SQLite database, copied-vector manifest, frozen protocols, every candidate/classification/verification, API request/response caches without credentials, retrieval traces, answers, judgments, benchmark screening, review packets, costs, environment and source archive. Copy this directory separately when transferring machines. The companion [JSON report](openrouter_experiment_20261119.json) contains detailed paired results and hashes.",
        "",
        "Reproduction commands:",
        "",
        "```bash",
        ".venv-linux/bin/python scripts/build_openrouter_graph.py \\",
        "  --source-snapshot reports/runs/linux_20261118/coverage_v2 \\",
        "  --source-collection legal_repair_linux_20261118_v2 \\",
        "  --collection NEW_COLLECTION --output NEW_RUN --sources 1000 --pairs 1000 --workers 4",
        ".venv-linux/bin/python scripts/evaluate_openrouter_experiment.py \\",
        "  --run NEW_RUN --corpus data/vlsp/corpus.csv \\",
        "  --review-root reports/runs/linux_20261118/review --workers 4",
        "```",
        "",
        "Keep the supplied key in the ignored `.env`. Existing collection names are rejected unless accompanied by this experiment’s matching copy checkpoint. Caches and per-query records permit restarting the same frozen run; mismatched protocols/code are rejected.",
        "",
        "**Remaining limitation:** independent review of source identity, relation precision, benchmark linkage and answer correctness is still required. The 30 reserved test candidates remain untouched by retrieval/answer evaluation. The current results support only the diagnostic and explicitly model-screened development claims reported here.",
        "",
    ]
    output.write_text("\n".join(lines))
    compact = {
        "status": done["status"],
        "conclusion": verdict,
        "run": str(run),
        "model": model["id"],
        "canonical_slug": model.get("canonical_slug"),
        "reasoning_effort": "medium",
        "collection": evaluation["collection"],
        "snapshot_sha256": graph["snapshot_sha256"],
        "graph": graph,
        "retrieval": overall,
        "paired_retrieval": paired,
        "answers": {k: v for k, v in answers.items() if k != "records"},
        "development_retrieval": dev["overall"],
        "screened_relational_retrieval": rel_summary,
        "development_answers": {k: v for k, v in dev_answers.items() if k != "records"},
        "screened_development_qids": sorted(screened),
        "relation_precision": {k: v for k, v in precision.items() if k != "records"},
        "api_usage": usage,
        "baseline_top10_drift_queries": baseline_drift,
        "baseline_recall_drift_queries": baseline_recall_drift,
        "non_citation_expansion_queries": semantic_queries,
        "independent_review": "pending",
        "final_audit": audit,
        "reserved_test_evaluated": False,
        "artifact_hashes": {
            str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                run / "graph_build.json",
                run / "comparison/evaluation.json",
                run / "answers/evaluation.json",
                run / "development_comparison/evaluation.json",
                run / "development_answers/evaluation.json",
                run / "precision/evaluation.json",
                run / "benchmark_screening.json",
            ]
        },
    }
    atomic_json(output.with_suffix(".json"), compact)
    atomic_json(run / "api_usage.json", usage)
    print(verdict)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    summarize(a.run, a.output)
