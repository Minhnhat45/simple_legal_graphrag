"""Build an isolated, bounded LLM relation experiment without diagnostic gold input."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legal_graphrag.config import Settings
from legal_graphrag.extraction import chunk_body
from legal_graphrag.models import DocumentChunk, LegalDocument, QueryFilters
from legal_graphrag.openrouter import OpenRouterClient, atomic_json
from legal_graphrag.qdrant_store import POINT_NAMESPACE, QdrantStore
from legal_graphrag.semantic import (
    PROMPT,
    REVIEW_PROMPT,
    REVIEW_SCHEMA,
    explicit_article_relations,
    referenced_scope_matches,
    resolve_aliases,
    response_schema,
    validate_facts,
    validate_provision_relation,
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_snapshot(snapshot):
    db = sqlite3.connect(f"file:{snapshot / 'snapshot.sqlite'}?mode=ro", uri=True)
    documents = {
        did: LegalDocument(**json.loads(p))
        for did, p in db.execute("SELECT id,payload FROM documents")
    }
    chunks = {
        cid: DocumentChunk(**json.loads(p))
        for cid, aid, p in db.execute("SELECT id,article_id,payload FROM chunks")
    }
    mapping = dict(db.execute("SELECT id,article_id FROM chunks"))
    db.close()
    articles = defaultdict(list)
    for cid, aid in mapping.items():
        articles[aid].append(cid)
    return documents, chunks, mapping, articles


def copy_vectors(source_collection, collection, snapshot, run):
    """Copy stored dense AND sparse vectors, checking every payload against the snapshot."""
    from qdrant_client import models

    settings = replace(Settings.from_env(), qdrant_collection=collection, qdrant_hybrid=True)
    store = QdrantStore(settings)
    _, chunks, mapping, _ = load_snapshot(snapshot)
    expected = {str(uuid.uuid5(POINT_NAMESPACE, cid)): cid for cid in chunks}
    state_path = run / "vector_copy.json"
    identity = {
        "source_collection": source_collection,
        "collection": collection,
        "source_snapshot_sha256": sha(snapshot / "snapshot.sqlite"),
        "chunks": len(chunks),
    }
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise ValueError("Vector copy identity changed")
        if state["complete"]:
            if store.count() != len(chunks):
                raise ValueError("Completed copied collection changed")
            return store
    else:
        if store.collection_exists():
            raise ValueError("Refuse existing target collection")
        if store.client.count(source_collection, exact=True).count != len(chunks):
            raise ValueError("Source collection count differs from frozen snapshot")
        info = store.client.get_collection(source_collection)
        store.client.create_collection(
            collection_name=collection,
            vectors_config=info.config.params.vectors,
            sparse_vectors_config=info.config.params.sparse_vectors,
        )
        state = {
            "identity": identity,
            "offset": None,
            "copied": 0,
            "complete": False,
            "seconds": 0.0,
        }
        atomic_json(state_path, state)
    started = time.monotonic()
    previous = state["seconds"]
    while state["copied"] < len(chunks):
        points, offset = store.client.scroll(
            source_collection,
            offset=state["offset"],
            limit=256,
            with_payload=True,
            with_vectors=True,
        )
        converted = []
        for point in points:
            cid = expected.get(str(point.id))
            if cid is None or any(
                point.payload.get(k) != v
                for k, v in {
                    "chunk_id": cid,
                    "article_id": mapping[cid],
                    "text": chunks[cid].text,
                    "document_id": chunks[cid].document_id,
                }.items()
            ):
                raise ValueError("Source vector payload differs from immutable evidence")
            converted.append(
                models.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
            )
        store.client.upsert(collection_name=collection, points=converted, wait=True)
        state.update(
            offset=str(offset) if offset is not None else None,
            copied=state["copied"] + len(points),
            seconds=previous + time.monotonic() - started,
        )
        atomic_json(state_path, state)
        if state["copied"] % 4096 == 0:
            print(f"copied {state['copied']}/{len(chunks)} vectors", flush=True)
        if offset is None:
            break
    if store.count() != len(chunks):
        raise ValueError("Copied collection count mismatch")
    store.ensure_collection(1024)
    state["complete"] = True
    atomic_json(state_path, state)
    return store


def discover(store, chunks, mapping, articles, citations, *, source_limit, pair_limit):
    sources = sorted(articles, key=lambda aid: hashlib.sha256(aid.encode()).hexdigest())[
        :source_limit
    ]
    pairs = {}

    def add(a, b, cids, route, rank):
        if a == b:
            return
        key = tuple(sorted((a, b)))
        item = pairs.setdefault(
            key, {"articles": list(key), "chunks": [], "routes": [], "rank": rank}
        )
        item["chunks"] = list(dict.fromkeys(item["chunks"] + list(cids)))
        item["routes"] = sorted(set(item["routes"] + [route]))
        item["rank"] = min(item["rank"], rank)

    for r in citations:
        if r.source_id in sources or r.target_id in sources:
            add(r.source_id, r.target_id, [r.source_chunk_id, r.target_chunk_id], "reference", 0)
    store._sparse_encoder()
    for i, aid in enumerate(sources, 1):
        cid = articles[aid][0]
        point = store.client.retrieve(
            store.collection,
            ids=[str(uuid.uuid5(POINT_NAMESPACE, cid))],
            with_vectors=True,
            with_payload=False,
        )[0]
        vector = point.vector["dense"]
        hits = store.hybrid_query(
            chunk_body(chunks[cid].text),
            vector,
            filters=QueryFilters(),
            limit=24,
            candidate_limit=40,
        )
        for rank, hit in enumerate(hits, 1):
            target = mapping.get(hit.chunk_id)
            if target:
                add(aid, target, [cid, hit.chunk_id], "dense_bm25_hybrid", rank)
        if i % 100 == 0:
            print(f"candidate sources {i}/{len(sources)}", flush=True)

    def tie(item):
        return hashlib.sha256("|".join(item["articles"]).encode()).hexdigest()

    explicit = sorted((p for p in pairs.values() if "reference" in p["routes"]), key=tie)
    semantic = sorted(
        (p for p in pairs.values() if "reference" not in p["routes"]),
        key=lambda p: (p["rank"], tie(p)),
    )
    interleaved = []
    for i in range(max(len(explicit), len(semantic))):
        for group in (explicit, semantic):
            if i < len(group):
                interleaved.append(group[i])
    return {
        "source_articles": sources,
        "candidate_pairs": len(pairs),
        "reference_pairs": len(explicit),
        "similarity_only_pairs": len(semantic),
        "selection": "alternate reference and hybrid similarity; no questions or gold used",
        "pairs": interleaved[:pair_limit],
    }


def classify(pair, client, documents, chunks, articles, output):
    pair_key = hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest()
    path = output / f"{pair_key}.json"
    if path.exists():
        return json.loads(path.read_text())
    supplied = {
        aid: ([cid for cid in pair["chunks"] if cid in articles[aid]][:2] or articles[aid][:1])
        for aid in pair["articles"]
    }
    local = {cid: chunks[cid] for cids in supplied.values() for cid in cids}
    aliases = {f"A{i}": aid for i, aid in enumerate(supplied)}
    chunk_aliases = {f"C{i}": cid for i, cid in enumerate(local)}
    reverse = {cid: label for label, cid in chunk_aliases.items()}
    payload = [
        {
            "id": label,
            "section": local[supplied[aid][0]].section,
            "number": documents[local[supplied[aid][0]].document_id].number,
            "chunks": [
                {"id": reverse[cid], "body": chunk_body(local[cid].text)} for cid in supplied[aid]
            ],
        }
        for label, aid in aliases.items()
    ]
    raw, key = client.call(
        [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        schema=response_schema(aliases, chunk_aliases),
    )
    response = resolve_aliases(raw, aliases, chunk_aliases)
    result = {
        "pair": pair,
        "classification_request": key,
        "proposed": response["relations"],
        "accepted": [],
        "rejected": [],
        "facts": [],
    }
    for data in response["relations"]:
        relation = validate_provision_relation(data, local, supplied, documents)
        if relation is None:
            result["rejected"].append(
                {"proposal": data, "reason": "grounding_or_identity_validation_failed"}
            )
            continue
        if not referenced_scope_matches(relation, documents, local):
            result["rejected"].append(
                {"proposal": data, "reason": "target_quote_does_not_ground_referenced_clause"}
            )
            continue
        review_payload = {
            "kind": relation.kind,
            "source_quote": relation.evidence,
            "target_quote": relation.target_evidence,
            "source_law": documents[local[relation.source_chunk_id].document_id].number,
            "target_law": documents[local[relation.target_chunk_id].document_id].number,
        }
        review, review_key = client.call(
            [
                {"role": "system", "content": REVIEW_PROMPT},
                {"role": "user", "content": json.dumps(review_payload, ensure_ascii=False)},
            ],
            schema=REVIEW_SCHEMA,
        )
        if not review["supported"]:
            result["rejected"].append(
                {
                    "proposal": data,
                    "reason": "second_pass_rejected",
                    "review": review,
                    "review_request": review_key,
                }
            )
            continue
        provenance = {
            **json.loads(relation.provenance),
            "provider": "OpenRouter",
            "model": client.model,
            "reasoning_effort": "medium",
            "classification_request": key,
            "review_request": review_key,
            "review": review,
            "candidate": pair,
        }
        result["accepted"].append(
            asdict(
                replace(
                    relation,
                    method="openrouter_verified",
                    provenance=json.dumps(provenance, ensure_ascii=False),
                )
            )
        )
    result["facts"] = validate_facts(
        response["facts"],
        local,
        supplied,
        json.dumps({"provider": "OpenRouter", "model": client.model, "request_hash": key}),
    )
    atomic_json(path, result)
    return result


def build(source_snapshot, source_collection, collection, run, sources=1000, pairs=1000, workers=4):
    run.mkdir(parents=True, exist_ok=True)
    coverage = json.loads((source_snapshot / "coverage.json").read_text())
    if sha(source_snapshot / "snapshot.sqlite") != coverage["snapshot_sha256"]:
        raise ValueError("Source snapshot hash mismatch")
    protocol = {
        "source_snapshot_sha256": coverage["snapshot_sha256"],
        "source_collection": source_collection,
        "collection": collection,
        "model": "openai/gpt-6-luna",
        "reasoning_effort": "medium",
        "source_limit": sources,
        "pair_limit": pairs,
        "workers": workers,
        "discovery": "reference edges plus stored-vector/BM25 article similarity",
        "candidate_input": "source articles only; no diagnostic questions or gold labels",
        "code_hashes": {
            str(p): sha(p)
            for p in [Path(__file__), *sorted(Path("src/legal_graphrag").glob("*.py"))]
        },
    }
    if (run / "protocol.json").exists():
        if json.loads((run / "protocol.json").read_text()) != protocol:
            raise ValueError("Graph experiment protocol/code mismatch")
    else:
        atomic_json(run / "protocol.json", protocol)
    if (run / "graph_build.json").exists():
        return
    store = copy_vectors(source_collection, collection, source_snapshot, run)
    snapshot = run / "snapshot"
    snapshot.mkdir(exist_ok=True)
    if not (snapshot / "snapshot.sqlite").exists():
        shutil.copyfile(source_snapshot / "snapshot.sqlite", snapshot / "snapshot.sqlite")
    documents, chunks, mapping, articles = load_snapshot(source_snapshot)
    started = time.monotonic()
    citations = explicit_article_relations(list(documents.values()), list(chunks.values()), mapping)
    if (run / "candidates.json").exists():
        candidates = json.loads((run / "candidates.json").read_text())
    else:
        candidates = discover(
            store, chunks, mapping, articles, citations, source_limit=sources, pair_limit=pairs
        )
        atomic_json(run / "candidates.json", candidates)
    client = OpenRouterClient(run / "api_cache")
    output = run / "graph_pair_results"
    output.mkdir(exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bounded batches stop after a service failure instead of queueing the full API budget.
        for start in range(0, len(candidates["pairs"]), 16):
            batch = candidates["pairs"][start : start + 16]
            results.extend(
                pool.map(lambda p: classify(p, client, documents, chunks, articles, output), batch)
            )
            print(
                f"relation pairs {len(results)}/{len(candidates['pairs'])}; "
                f"accepted LLM relations {sum(len(r['accepted']) for r in results)}",
                flush=True,
            )
    accepted = {
        (r.source_id, r.target_id, r.kind, r.source_chunk_id, r.target_chunk_id): asdict(r)
        for r in citations
    }
    for item in results:
        for r in item["accepted"]:
            accepted[
                (
                    r["source_id"],
                    r["target_id"],
                    r["kind"],
                    r["source_chunk_id"],
                    r["target_chunk_id"],
                )
            ] = r
    db = sqlite3.connect(snapshot / "snapshot.sqlite")
    db.execute("DELETE FROM relations")  # This is the NEW private experiment database only.
    db.executemany(
        "INSERT INTO relations VALUES (?,?,?)",
        [
            (r["source_id"], r["target_id"], json.dumps(r, ensure_ascii=False))
            for r in accepted.values()
        ],
    )
    db.execute("CREATE TABLE IF NOT EXISTS model_facts (id TEXT PRIMARY KEY,payload TEXT)")
    db.executemany(
        "INSERT OR REPLACE INTO model_facts VALUES (?,?)",
        [(f["id"], json.dumps(f, ensure_ascii=False)) for r in results for f in r["facts"]],
    )
    db.commit()
    db.close()
    coverage.update(
        snapshot_sha256=sha(snapshot / "snapshot.sqlite"),
        source_snapshot_sha256=protocol["source_snapshot_sha256"],
        status="openrouter_graph_snapshot; copied_frozen_embeddings",
        graph_build_seconds=time.monotonic() - started,
        resolved_citations=sum(r["kind"] == "CITES" for r in accepted.values()),
    )
    atomic_json(snapshot / "coverage.json", coverage)
    shutil.copyfile(source_snapshot / "mapping_gaps.json", snapshot / "mapping_gaps.json")
    report = {
        "status": "complete",
        "protocol": protocol,
        "snapshot_sha256": coverage["snapshot_sha256"],
        "rule_citations": len(citations),
        "classified_pairs": len(results),
        "accepted_llm_relations": sum(len(r["accepted"]) for r in results),
        "accepted_by_kind": dict(Counter(r["kind"] for item in results for r in item["accepted"])),
        "graph_by_kind": dict(Counter(r["kind"] for r in accepted.values())),
        "rejected_proposals": sum(len(r["rejected"]) for r in results),
        "elapsed_seconds": time.monotonic() - started,
        "api_usage": client.usage(),
    }
    atomic_json(run / "graph_build.json", report)
    store.client.close()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-snapshot", required=True, type=Path)
    p.add_argument("--source-collection", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--sources", type=int, default=1000)
    p.add_argument("--pairs", type=int, default=1000)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    build(
        a.source_snapshot,
        a.source_collection,
        a.collection,
        a.output,
        a.sources,
        a.pairs,
        a.workers,
    )
