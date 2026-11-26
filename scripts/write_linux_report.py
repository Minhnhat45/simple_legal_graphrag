"""Render the completed Linux measurements without inventing pending human judgments."""
# Long Markdown table rows and prose literals are intentionally kept readable.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def percent(value):
    return "unavailable" if value is None else f"{100 * value:.2f}%"


def interval(value):
    return f"{100 * value['difference']:+.2f} pp [{100 * value['lower_95']:+.2f}, {100 * value['upper_95']:+.2f}]"


def ids(values):
    return ", ".join(values) or "—"


def write(run, output):
    retrieval = json.loads((run / "retrieval_summary.json").read_text())
    answers = json.loads((run / "answers/evaluation.json").read_text())
    precision_path = run / "relation_precision/evaluation.json"
    precision = (
        json.loads(precision_path.read_text())
        if precision_path.exists()
        else {
            "status": "failed_invalid_model_output; independent_review_pending",
            "strata": {},
            "weighted_automated_support": None,
            "human_precision": None,
            "cached_attempts": len(list((run / "relation_precision/cache").glob("*.json"))),
        }
    )
    if not precision_path.exists():
        packet = json.loads((run / "review/relation_review.json").read_text())
        precision["strata"] = {
            name: {
                "population": population,
                "sample": sum(r["stratum"] == name for r in packet["items"]),
                "automated_supported": None,
            }
            for name, population in packet["population"].items()
        }
    identity = json.loads((run / "identity_audit.json").read_text())
    protocol = json.loads((run / "review/answer_protocol.json").read_text())
    candidates = json.loads((run / "review/relational_candidates.json").read_text())
    rejected = json.loads((run / "review/rejected_relations.json").read_text())
    coverage = json.loads((run / "coverage_v2/coverage.json").read_text())
    environment = json.loads((run / "linux_environment.json").read_text())
    overall = retrieval["overall"]
    controls = retrieval["paired_controls"]
    graph_delta = controls["graph_shared_minus_hybrid_shared"]["all"]["recall"]
    shuffle_delta = controls["graph_shared_minus_shuffled_shared"]["all"]["recall"]
    supports = graph_delta["lower_95"] > 0 and shuffle_delta["lower_95"] > 0
    conclusion = (
        "The exploratory diagnostic intervals favor citation expansion over both matched shared "
        "scoring and shuffled links. This is not an untouched-test or independently reviewed finding."
        if supports
        else "The controlled diagnostic comparison does not establish a reliable graph advantage over "
        "both shared hybrid scoring and shuffled links."
    )
    lines = [
        "# Linux experiment results, 2026-11-18",
        "",
        "**Status: diagnostic retrieval completed; answer and relationship evaluation attempts failed "
        "model-output validation.** No experiment jobs remain running. Independent legal review "
        "and a validated relational test benchmark remain unfinished.",
        "",
        conclusion,
        "",
        "This run continues [the plan](../plan_fixing_failure_11_11.md) and "
        "[the Linux handoff](linux_experiment_handoff_20261113.md). "
        "The original 2026-11-07 artifacts and collection were preserved. No tuning used these results.",
        "",
        "## Corpus and identity coverage",
        "",
        "The original corpus, evaluation CSV, manifest and frozen evaluation match all four hashes "
        "in the handoff. All 2,051 source-document fingerprints match the original Linux manifest. "
        "The Mac SQLite snapshot was absent, so it was rebuilt in a new directory. The first rebuild "
        "reproduced every reported count.",
        "",
        "A tested normalization fix inserts the missing space before an uppercase sentence start "
        "after punctuation. It preserves letters, numbers, punctuation, source evidence and gold labels. "
        "It accepts only unique full-body matches within the same law number. "
        "It adds 4,919 mappings and rejects 119 previously unique mappings that become ambiguous "
        "(118 many-to-one IDs and one duplicate source match), for a net gain of 4,800.",
        "",
        "| Measure | Rebuilt handoff snapshot | Frozen Linux snapshot |",
        "| --- | ---: | ---: |",
        "| Documents | 2,051 | 2,051 |",
        "| Articles | 55,898 | 55,898 |",
        "| Chunks | 91,405 | 91,405 |",
        "| Validated original IDs | 31,307 | 36,107 |",
        "| Fully mapped questions | 47 / 100 | 59 / 100 |",
        "| Mapped gold occurrences | 53 / 113 | 66 / 113 |",
        "| Automatically resolved citations | 19,674 | 19,674 |",
        "",
        "The old chunk-containment rule and the new full-body rule cover 53 questions in common. "
        "Fifteen satisfy only the old rule and six only the new rule. Thus the original 68-question "
        "coverage count and the new 59-question count use different identity standards. "
        "This is not a measured retrieval regression. Forty-six documents still have no parsed articles. "
        "Ambiguous duplicates and substantive text/parser differences remain unresolved.",
        "",
        f"Under these accepted identity mappings, the all-question mean recall ceiling is "
        f"{percent(retrieval['mapping_recall_ceiling'])}, and the complete-evidence ceiling is "
        "59%. Unmapped identities may still have source text present; they cannot receive "
        "gold-ID retrieval credit without validated mapping.",
        "",
        "## Controlled retrieval results",
        "",
        "All seven configurations share the new BGE-M3 hybrid index, unchanged questions/gold labels, "
        "Top-K 10 and an 18,000-character source-text cap. Candidate caps are 60 except the explicitly "
        "larger 120-candidate baseline. These are matched upper bounds; realized counts differ. "
        "Shared scoring is the existing cosine-plus-token-overlap formula, not a trained cross-encoder. "
        "Query embeddings and actual hybrid pools are reused within each question. "
        "The successful output contains 700 records: 100 questions × seven configurations.",
        "",
        "| Configuration | Candidate recall | Recall@10 | Complete evidence | Recall@10, fully mapped | New gold candidates | Recovered / lost gold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, groups in overall.items():
        a, b = groups["all"], groups["fully_covered"]
        lines.append(
            f"| `{mode}` | {percent(a['candidate_recall'])} | {percent(a['recall'])} | "
            f"{percent(a['complete_evidence'])} | {percent(b['recall'])} | "
            f"{a['new_candidate_gold_occurrences']} | "
            f"{a['recovered_gold_occurrences']} / {a['lost_gold_occurrences']} |"
        )
    lines += [
        "",
        "New candidates and recovered/lost occurrences are relative to `hybrid`. "
        "Complete evidence means all gold article IDs occur among the selected chunks; "
        "it does not prove all required clauses were selected.",
        "",
        "Paired percentile bootstrap intervals use 2,000 resamples and the frozen seed. "
        "The following Recall@10 differences are percentage points, treatment minus control. "
        "They are exploratory and not adjusted for multiple comparisons.",
        "",
        "| Comparison | All 100 questions, difference [95% CI] | Fully mapped 59 |",
        "| --- | ---: | ---: |",
    ]
    for name, c in controls.items():
        lines.append(
            f"| `{name}` | {interval(c['all']['recall'])} | "
            f"{interval(c['fully_covered']['recall'])} |"
        )
    reach = retrieval["reachability"]
    lines += [
        "",
        f"Two-hop graph traversal from actual hybrid seeds reaches "
        f"{reach['gold_occurrences_reachable_beyond_hybrid_candidates']} gold occurrences "
        "outside the baseline candidate pool. Operational bounded expansion adds "
        f"{reach['gold_occurrences_added_by_graph']} of them as confirmed candidates. "
        "Reachability includes unrestricted one/two-hop diagnostics; deployed expansion also "
        "applies degree, confidence, confirmation and candidate limits.",
        "",
        "The sole new gold candidate is 56079 for qid 9882. It ranks 53rd with shared "
        "scoring, 44th with the path bonus and 49th with legacy graph scoring, so none selects "
        "it in the top 10. Graph expansion also displaces gold 53375 from the candidate pool "
        "for qid 567. The small legacy gain comes from qid 15959, whose recovered gold was "
        "already in the hybrid candidate pool. These cases support a narrow scoring/selection "
        "diagnosis, not a general benefit from newly reachable graph evidence.",
        "",
        "| Diagnostic qid | Mapped gold | Hybrid top-10 gold | Graph/shared top-10 gold | Reachable beyond hybrid candidates |",
        "| --- | --- | --- | --- | --- |",
    ]
    for q in retrieval["stages"]:
        if q["qid"] not in {"6920", "5124", "16281", "2217", "1164", "4190", "9882"}:
            continue
        lines.append(
            f"| {q['qid']} | {ids(q['mapped'])} | "
            f"{ids(a for a, r in q['gold_ranks']['hybrid'].items() if r)} | "
            f"{ids(a for a, r in q['gold_ranks']['graph_shared'].items() if r)} | "
            f"{ids(q['reachable_beyond_hybrid'])} |"
        )
    lines += [
        "",
        "## Sampled answer evaluation",
        "",
        "Before retrieval results were available, 20 diagnostic questions were selected by "
        "SHA-256 order with seed 20261118. Gemma generated answers for every configuration; "
        "Qwen was asked to judge correctness against full gold corpus provisions and support against the "
        "retrieved excerpts, blinded to configuration. Both use fixed temperature/seed and "
        "saved prompts. Identical ordered contexts reuse the exact answer and judgment. "
        "**All 140 answer judgments failed the required schema. Correctness and citation-support "
        "measurements are unavailable.** The 69 unique generation requests and 69 unique judge "
        "requests are preserved; shared contexts account for the 140 configuration records. "
        "The generator also produced truncated outputs in several cases. These failures must be "
        "repaired and rerun before drawing answer-quality conclusions.",
        "",
        "| Configuration | Judged / 20 | Reference sufficient | Automated correctness | Automated full citation support | Truncated answers |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, a in answers["summary"].items():
        lines.append(
            f"| `{mode}` | {a['judged']} | {a['reference_sufficient']} | "
            f"{percent(a['automated_correctness'])} | {percent(a['automated_full_citation_support'])} | "
            f"{a['truncated_answers']} |"
        )
    lines += [
        "",
        "Correctness denominators exclude judgments marked insufficient reference; "
        "support denominators include all valid judgments. Failed or oversized-reference "
        "judgments are retained separately, not silently counted as correct. "
        "Raw answers, cited-source checks, model outputs and validation errors are saved. "
        "There are no valid paired answer-quality intervals.",
        "",
        "## Relationship review and relational benchmark",
        "",
        "The 50-relation precision sample contains 25 within-document and 25 cross-document "
        "edges selected before retrieval outcomes. The first Qwen response was prose rather than "
        "the required JSON, and the audit stopped. No precision measurement was produced. "
        "The exact request, response and traceback are preserved.",
        "",
        "| Stratum | Population | Sample | Model-supported |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, s in precision["strata"].items():
        supported = s["automated_supported"]
        lines.append(
            f"| {name} | {s['population']} | {s['sample']} | "
            f"{supported if supported is not None else 'unavailable'} |"
        )
    lines += [
        "",
        f"The population-weighted automated support rate is "
        f"{percent(precision['weighted_automated_support'])}. The 50 sampled relations and "
        "30 rejected/ambiguous cases are ready for independent review. "
        "No new implementing, amending or semantic edges were accepted from model judgments.",
        "",
        "A separate 60-question natural review set was selected from 237 eligible multi-gold "
        "questions in the supplied training CSV, without selecting successful graph paths or "
        "requiring complete mapping. Thirty are development candidates and thirty are reserved "
        "test candidates. Law families join equal law numbers, co-gold laws and cross-document "
        "citations. The splits share no such family, and diagnostic families are excluded. "
        "Historical exposure of this training data is unknown; the reserved candidates were "
        "not run in this experiment. Reviewers must establish actual linkage requirements, "
        "rule/exception, condition/procedure and cross-document categories, gold identity and "
        "answers before this becomes a validated benchmark. No synthetic questions were added.",
        "",
        "## Cost, reproducibility and limits",
        "",
        f"- Snapshot build: {coverage['build_seconds']:.2f} seconds, including "
        f"{coverage['graph_build_seconds']:.2f} seconds for citation extraction.",
        f"- Full embedding and Qdrant ingestion: {retrieval['index_seconds']:.2f} seconds "
        f"({retrieval['index_seconds'] / 60:.2f} minutes), batch size 64.",
        f"- Answer generation: {answers['unique_generations']} unique requests, "
        f"{answers['unique_generation_seconds']:.2f} measured service seconds.",
        f"- Answer judgments: {answers['unique_judgments']} unique requests, "
        f"{answers['unique_judgment_seconds']:.2f} measured service seconds.",
        "- Hardware: NVIDIA RTX 5060 Ti, 16 GiB VRAM; Python 3.13.11. "
        "Package versions and model metadata are archived. No cloud API charges were incurred; "
        "electricity cost was not measured.",
        "",
        "| Configuration | Mean retrieval ms | Median ms | P95 ms | Mean confirmed articles | Mean source characters |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, d in retrieval["actual_budgets_and_latency"].items():
        latency = d["retrieval_ms"]
        lines.append(
            f"| `{mode}` | {latency['mean']:.2f} | {latency['median']:.2f} | "
            f"{latency['p95_nearest_rank']:.2f} | {d['confirmed_articles']['mean']:.2f} | "
            f"{d['context_chars']['mean']:.0f} |"
        )
    lines += [
        "",
        "Retrieval timings include the measured shared hybrid lookup and exclude query "
        "embedding. They are single-run local measurements, not a production load benchmark.",
        "",
        f"- Snapshot SHA-256: `{retrieval['snapshot_sha256']}`.",
        f"- BGE-M3 digest: `{retrieval['embedding_digest']}` (matches the Mac digest).",
        f"- Collection: `{retrieval['collection']}`.",
        "- Validation: 66 tests, 64 passed and two opt-in live Neo4j tests skipped; lint passed. "
        "A live three-chunk smoke experiment also passed interruption/resumption and the "
        "existing-collection guard. "
        "All snapshot citation quotes occur in their saved endpoint chunks. "
        "The completed retrieval output was checked for all expected records, unique top-K IDs "
        "and candidate/context caps.",
        "",
        "The initial run was interrupted by a server restart after 1,856 indexed chunks. "
        "Its partial collection and checkpoint are preserved; the final run uses a distinct "
        "snapshot and collection. The runner now validates snapshot/model/code/settings and "
        "existing point payloads on explicit `--resume`, and saves selected evidence for answer "
        "evaluation. Completed results cannot be overwritten through resume.",
        "",
        f"Raw artifacts are under `{run}` (Git-ignored). "
        "They include both rebuilt snapshots, predictions, score/path traces, model requests "
        "and responses, identity differences, review packets, frozen protocol, logs, package "
        "versions, code hashes and a source archive. Copy this directory separately when "
        "moving machines. The compact machine-readable companion is "
        "[fixing_failure_linux_20261118.json](fixing_failure_linux_20261118.json).",
        "",
        "To rerun, use the frozen snapshot with a new collection/output and "
        "`EMBED_BATCH_SIZE=64 .venv-linux/bin/python scripts/run_repaired_experiment.py`. "
        "Then use `summarize_repaired_experiment.py`, `run_answer_experiment.py` with the saved "
        "answer protocol and `--corpus data/vlsp/corpus.csv`, and `review_snapshot_relations.py` "
        "with the saved relation sample. The archived `finish_stages.sh` records the original "
        "command arguments. See the [data inventory](../data/README.md) for the current input paths.",
        "",
        "**Remaining work:** repair the generator/judge output failures and rerun the frozen "
        "answer evaluation; complete the citation audit. Independent source/legal review is needed to validate "
        "relationship precision, answer correctness and the relational benchmark. "
        "The reserved test set remains unevaluated pending that review. "
        "The diagnostic and automated results do not establish universal GraphRAG superiority.",
        "",
    ]
    output.write_text("\n".join(lines))
    compact = {
        "status": "diagnostic_retrieval_complete; model_evaluation_failed; independent_validation_pending",
        "conclusion": conclusion,
        "run_directory": str(run),
        "retrieval": retrieval,
        "answers": {
            **answers,
            "raw_status": answers["status"],
            "status": "failed_schema_validation; answer_quality_unavailable",
        },
        "relation_precision": precision,
        "identity_coverage_transitions": identity["coverage_transitions"],
        "answer_protocol": protocol,
        "relational_candidates": {
            "count": len(candidates["items"]),
            "eligible": candidates["eligible_candidates"],
            "review_status": "pending",
        },
        "rejected_relation_population": rejected["population"],
        "base_commit": environment["base_commit"],
        "artifact_hashes": {
            str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                run / "retrieval_summary.json",
                run / "answers/evaluation.json",
                run / "relation_precision/evaluation.json",
                run / "review/manifest.json",
            ]
            if p.exists()
        },
    }
    output.with_suffix(".json").write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    write(a.run, a.output)
