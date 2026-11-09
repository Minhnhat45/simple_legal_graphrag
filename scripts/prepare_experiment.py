"""Copy only text-validated existing embeddings into a NEW current-data hybrid collection.

The source collection is read-only. Current filtered texts determine eligibility;
original aid labels are carried unchanged. No stored training questions are read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_graphrag.config import Settings
from legal_graphrag.csv_loader import normalize_text, stream_dataset
from legal_graphrag.graph_store import GraphStore
from legal_graphrag.models import DocumentChunk
from legal_graphrag.qdrant_store import POINT_NAMESPACE
from legal_graphrag.structure import parse_structure


def main():
    from qdrant_client import QdrantClient, models

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data")
    parser.add_argument("--source-collection", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.source_collection == args.collection or Path(args.output).exists():
        parser.error("Use a new output path and a distinct collection")
    settings = Settings.from_env()
    client = QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=120
    )
    if client.collection_exists(args.collection):
        parser.error("Target collection already exists; no overwrite is supported")
    documents = list(stream_dataset(args.dataset))
    indexed = defaultdict(list)
    article_counts = Counter()
    for document in documents:
        for p in parse_structure(document):
            if p.level == "Article":
                indexed[document.number].append((document.id, p.id, normalize_text(p.text)))
                article_counts[document.id] += 1
    offset, accepted, rejected, aids = None, [], Counter(), defaultdict(set)
    while True:
        points, offset = client.scroll(
            args.source_collection,
            limit=256,
            offset=offset,
            with_payload=[
                "aid",
                "law_id",
                "chunk_text",
                "embedding_text",
                "chunk_index",
                "embedding_model",
                "embedding_model_digest",
            ],
            with_vectors=args.apply,
        )
        for point in points:
            p = point.payload
            body = normalize_text(p.get("chunk_text", ""))
            matches = [
                (did, aid)
                for did, aid, text in indexed.get(p.get("law_id"), [])
                if len(body) >= 30 and body in text
            ]
            if len(matches) != 1:
                rejected["unmatched" if not matches else "ambiguous"] += 1
                continue
            did, aid = matches[0]
            if p.get("embedding_model") not in {"bge-m3", "bge-m3:latest"}:
                rejected["wrong_model"] += 1
                continue
            aids[str(p["aid"])].add(aid)
            accepted.append(
                {
                    "source_point_id": str(point.id),
                    "document_id": did,
                    "article_id": aid,
                    "original_article_id": str(p["aid"]),
                    "text": p["chunk_text"],
                    "chunk_index": p["chunk_index"],
                    "vector": point.vector,
                    "model_digest": p.get("embedding_model_digest"),
                    "reembed": not p.get("embedding_text")
                    or not any(
                        normalize_text(p["embedding_text"]) in text
                        for docid, article, text in indexed[p["law_id"]]
                        if (docid, article) == (did, aid)
                    ),
                }
            )
        if offset is None:
            break
    # Conflicting old-to-current mappings are excluded, never guessed by article ordinal.
    accepted = [p for p in accepted if len(aids[p["original_article_id"]]) == 1]
    reverse = defaultdict(set)
    for p in accepted:
        reverse[p["article_id"]].add(p["original_article_id"])
    accepted = [p for p in accepted if len(reverse[p["article_id"]]) == 1]
    identity = {p["article_id"]: p["original_article_id"] for p in accepted}
    report = {
        "dataset": str(Path(args.dataset).resolve()),
        "source_collection": args.source_collection,
        "collection": args.collection,
        "documents": len(documents),
        "accepted_chunks": len(accepted),
        "mapped_articles": len(identity),
        "rejected": dict(rejected),
        "original_ids": identity,
        "current_document_hashes": {d.id: d.fingerprint for d in documents},
        "mapping_rule": "Exact normalized chunk body substring in exactly one current article; "
        "one-to-one original aid/current article mapping; no question fields.",
        "applied": args.apply,
        "reembedded_chunks": sum(p["reembed"] for p in accepted),
    }
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"original_ids", "current_document_hashes"}
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.apply:
        from fastembed import SparseTextEmbedding

        from legal_graphrag.extraction import extract_mentions
        from legal_graphrag.llm import OllamaClient

        tags = OllamaClient(settings)._request("/api/tags")["models"]
        digest = next(m["digest"] for m in tags if m["name"] == "bge-m3:latest")
        if not accepted or any(p["model_digest"] != digest for p in accepted):
            raise ValueError("Stored embedding digest does not match local bge-m3:latest")
        report["embedding_digest"] = digest
        sparse = SparseTextEmbedding(model_name="Qdrant/bm25", disable_stemmer=True, avg_len=256.0)
        graph = GraphStore(settings)
        graph.ensure_schema()
        dmap = {d.id: d for d in documents}
        used = {p["document_id"] for p in accepted}
        # Stable namespaced identities protect unrelated and previous graph work.
        namespace = "experiment:" + args.collection + ":"
        scoped = {did: replace(dmap[did], id=namespace + did) for did in used}
        graph.prepare_documents(
            list(scoped.values()), [m for d in scoped.values() for m in extract_mentions(d)]
        )
        client.create_collection(
            args.collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=len(accepted[0]["vector"]), distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        chunks = defaultdict(list)
        actual_mapping = {}
        chunk_mapping = {}
        start = time.monotonic()
        for i in range(0, len(accepted), 64):
            batch = accepted[i : i + 64]
            lexical = list(sparse.embed([p["text"] for p in batch]))
            rebuild = [p for p in batch if p["reembed"]]
            if rebuild:
                embeddings = OllamaClient(settings).embed([p["text"] for p in rebuild])
                for item, vector in zip(rebuild, embeddings, strict=True):
                    item["vector"] = vector
            points = []
            for p, lex in zip(batch, lexical, strict=True):
                doc = scoped[p["document_id"]]
                aid = namespace + p["article_id"]
                cid = namespace + p["source_point_id"]
                section = "Điều " + p["article_id"].split(":article:")[-1]
                chunk = DocumentChunk(cid, doc.id, p["chunk_index"], section, p["text"])
                chunks[doc.id].append(chunk)
                actual_mapping[aid] = p["original_article_id"]
                chunk_mapping[cid] = aid
                payload = asdict(doc)
                payload.pop("full_text")
                payload.update(
                    chunk_id=cid,
                    document_id=doc.id,
                    chunk_index=chunk.index,
                    section=section,
                    text=chunk.text,
                    article_id=aid,
                    original_article_id=p["original_article_id"],
                    document_number=doc.number,
                    embedding_model="bge-m3:latest",
                    embedding_model_digest=digest,
                    source_point_id=p["source_point_id"],
                    embedding_rebuilt_from_current_body=p["reembed"],
                )
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid5(POINT_NAMESPACE, cid)),
                        payload=payload,
                        vector={
                            "dense": p["vector"],
                            "bm25": models.SparseVector(
                                indices=lex.indices.tolist(), values=lex.values.tolist()
                            ),
                        },
                    )
                )
            client.upsert(args.collection, points, wait=True)
            print(f"copied={min(i + 64, len(accepted))}/{len(accepted)}", flush=True)
        # Upsert the complete chunk set (batch grouping above is not a hierarchy assumption).
        for doc in scoped.values():
            graph.upsert_chunks(chunks[doc.id])
            graph.mark_indexed(
                [{"id": doc.id, "index_hash": doc.fingerprint, "chunk_count": len(chunks[doc.id])}]
            )
            graph.upsert_structure(doc, chunks[doc.id])
        graph.link_article_chunks(chunk_mapping)
        report["original_ids"] = actual_mapping
        report["namespace"] = namespace
        report["copy_seconds"] = time.monotonic() - start
        graph.close()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    client.close()


if __name__ == "__main__":
    main()
