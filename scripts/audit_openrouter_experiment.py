"""Validate completed experimental artifacts without making model requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import jsonschema

from legal_graphrag.config import load_dotenv
from legal_graphrag.openrouter import atomic_json


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(run):
    load_dotenv()
    secret = os.getenv("OPEN_ROUTER_API", "").encode()
    assert secret, "Credential required only for local leak checking; no API calls are made"
    done = read(run / "completed.json")
    assert done["status"] == "computational_stages_complete"
    graph = read(run / "graph_build.json")
    protocol = read(run / "protocol.json")
    evaluation_protocol = read(run / "evaluation_protocol.json")
    assert sha(run / "snapshot/snapshot.sqlite") == graph["snapshot_sha256"]
    assert graph["snapshot_sha256"] == evaluation_protocol["graph_snapshot_sha256"]
    source = Path("reports/runs/linux_20261118/coverage_v2/snapshot.sqlite")
    assert sha(source) == protocol["source_snapshot_sha256"]
    for name, expected in protocol["code_hashes"].items():
        assert sha(Path(name)) == expected, f"Graph source changed: {name}"
    assert (
        sha(Path("scripts/evaluate_openrouter_experiment.py")) == evaluation_protocol["code_sha256"]
    )
    for name, expected in evaluation_protocol["supporting_code_sha256"].items():
        assert sha(Path("scripts") / name) == expected
    assert (
        len(list((run / "graph_pair_results").glob("*.json"))) == graph["classified_pairs"] == 1000
    )
    screen = read(run / "benchmark_screening.json")
    reserved = {r["qid"] for r in screen["rows"] if r["split"] == "test_reserved"}
    development = {r["qid"] for r in screen["rows"] if r["split"] == "development"}
    primary = set(evaluation_protocol["diagnostic_qids"])
    assert len(reserved) == len(development) == 30 and len(primary) == 100
    assert not (reserved & development or reserved & primary or development & primary)
    modes = set(evaluation_protocol["configs"])
    counts = {}
    references = set()
    for directory, answer_directory, qids in [
        ("comparison", "answers", primary),
        ("development_comparison", "development_answers", development),
    ]:
        rows = read(run / directory / "evaluation.json")["predictions"]
        answers = read(run / answer_directory / "evaluation.json")["records"]
        expected = {(qid, mode) for qid in qids for mode in modes}
        assert len(rows) == len(answers) == len(expected)
        assert {(r["qid"], r["mode"]) for r in rows} == expected
        assert {(r["qid"], r["mode"]) for r in answers} == expected
        for row in rows:
            assert len(row["retrieved"]) == len(set(row["retrieved"])) <= 10
            assert row["trace"]["context_chars"] <= 18000
        references.update(r[k] for r in answers for k in ["generation_request", "judgment_request"])
        counts[directory] = len(rows)
        counts[answer_directory] = len(answers)
    cache_keys = set()
    response_models = set()
    schema_requests = 0
    for path in sorted((run / "api_cache").glob("*.json")):
        record = read(path)
        identity = record["request"]
        key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        assert key == path.stem, "Request cache hash mismatch"
        cache_keys.add(key)
        payload = identity["payload"]
        assert identity["endpoint"] == "https://openrouter.ai/api/v1"
        assert payload["model"] == "openai/gpt-6-luna"
        assert payload["reasoning"] == {"effort": "medium", "exclude": True}
        assert payload["provider"] == {"require_parameters": True, "allow_fallbacks": False}
        response = record["response"]
        assert not response.get("error")
        response_models.add(response["model"])
        assert response["model"] in {"openai/gpt-6-luna", "openai/gpt-6-luna-20260922"}
        choice = response["choices"][0]
        assert choice["finish_reason"] == "stop"
        content = choice["message"]["content"]
        assert isinstance(content, str) and content.strip()
        if "response_format" in payload:
            jsonschema.validate(
                json.loads(content), payload["response_format"]["json_schema"]["schema"]
            )
            schema_requests += 1
    assert references <= cache_keys, "Answer references missing API records"
    for path in run.rglob("*"):
        if path.is_file() and path.suffix in {
            ".json",
            ".jsonl",
            ".log",
            ".md",
            ".py",
            ".sh",
            ".txt",
        }:
            assert secret not in path.read_bytes(), f"Credential leak detected in {path}"
    result = {
        "status": "passed",
        "records": counts,
        "api_records_checked": len(cache_keys),
        "schema_responses_checked": schema_requests,
        "actual_model_ids": sorted(response_models),
        "reserved_test_questions_excluded": len(reserved),
        "original_snapshot_unchanged": True,
        "source_code_hashes_match": True,
        "credential_scan_passed": True,
        "independent_human_review": "pending",
    }
    atomic_json(run / "final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    audit(parser.parse_args().run)
