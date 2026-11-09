from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from .config import Settings
from .extraction import normalize_document_number, normalize_entity_key
from .models import DocumentChunk, GraphNeighbor, LegalDocument, LegalRelation, Mention


class GraphStore:
    METADATA_RELATIONSHIPS = (
        "HAS_IDENTIFIER",
        "ISSUED_BY",
        "IN_SECTOR",
        "IN_FIELD",
        "SIGNED_BY",
        "MENTIONS",
        "SHARES_CONCEPT",
    )

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        if client is None:
            from neo4j import GraphDatabase

            auth = (
                (settings.neo4j_username, settings.neo4j_password)
                if settings.neo4j_username
                else None
            )
            client = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        self.client = client
        self.database = settings.neo4j_database

    def close(self) -> None:
        self.client.close()

    def _query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        routing: str = "w",
    ) -> list[Any]:
        records, _, _ = self.client.execute_query(
            query,
            parameters_=params or {},
            database_=self.database,
            routing_=routing,
        )
        return records

    def _read(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        return self._query(query, params, routing="r")

    def reset(self) -> None:
        self._query("MATCH (n) DETACH DELETE n")

    def ensure_schema(self) -> None:
        indexes = (
            "CREATE INDEX IF NOT EXISTS FOR (p:Provision) ON (p.id)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.id)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Provision) ON (p.document_id)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
            "CREATE INDEX IF NOT EXISTS FOR (i:Instrument) ON (i.number_norm)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Agency) ON (a.key)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Sector) ON (s.key)",
            "CREATE INDEX IF NOT EXISTS FOR (f:Field) ON (f.key)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.key)",
            "CREATE INDEX IF NOT EXISTS FOR (c:LegalConcept) ON (c.key)",
        )
        for query in indexes:
            self._query(query)

    def get_index_hashes(self, document_ids: Sequence[str]) -> dict[str, str]:
        if not document_ids:
            return {}
        result = self._read(
            """
            UNWIND $ids AS document_id
            MATCH (d:Document {id: document_id})
            RETURN d.id, d.index_hash
            """,
            {"ids": list(document_ids)},
        )
        return {str(row[0]): str(row[1] or "") for row in result}

    def _delete_previous_document_state(self, document_ids: Sequence[str]) -> None:
        params = {"ids": list(document_ids)}
        self._query(
            """
            UNWIND $ids AS document_id
            MATCH (:Document {id: document_id})-[r:LEGAL_RELATION]-()
            DELETE r
            """,
            params,
        )
        self._query(
            """
            MATCH (p:Provision) WHERE p.document_id IN $ids DETACH DELETE p
        """,
            params,
        )
        for relationship in self.METADATA_RELATIONSHIPS:
            self._query(
                f"""
                UNWIND $ids AS document_id
                MATCH (d:Document {{id: document_id}})-[r:{relationship}]->()
                DELETE r
                """,
                params=params,
            )
        self._query(
            """
            UNWIND $ids AS document_id
            MATCH (d:Document {id: document_id})-[:HAS_CHUNK]->(c:Chunk)
            DETACH DELETE c
            """,
            params=params,
        )

    def prepare_documents(
        self,
        documents: Sequence[LegalDocument],
        mentions: Sequence[Mention],
    ) -> None:
        if not documents:
            return
        document_ids = [document.id for document in documents]
        self._delete_previous_document_state(document_ids)

        rows = [
            {
                "id": document.id,
                "title": document.title,
                "number": document.number,
                "number_norm": normalize_document_number(document.number),
                "document_type": document.document_type,
                "sector": document.sector,
                "issued_date": document.issued_date,
                "field": document.field,
                "effective_date": document.effective_date,
                "status": document.status,
                "expiry_date": document.expiry_date,
                "agency": document.agency,
                "signer_position": document.signer_position,
                "signer": document.signer,
                "content_hash": document.fingerprint,
            }
            for document in documents
        ]
        self._query(
            """
            UNWIND $rows AS row
            MERGE (d:Document {id: row.id})
            SET d.title = row.title,
                d.number = row.number,
                d.number_norm = row.number_norm,
                d.document_type = row.document_type,
                d.sector = row.sector,
                d.issued_date = row.issued_date,
                d.field = row.field,
                d.effective_date = row.effective_date,
                d.status = row.status,
                d.expiry_date = row.expiry_date,
                d.agency = row.agency,
                d.signer_position = row.signer_position,
                d.signer = row.signer,
                d.content_hash = row.content_hash,
                d.index_hash = '',
                d.index_status = 'building'
            MERGE (i:Instrument {number_norm: row.number_norm})
            SET i.number = row.number
            MERGE (d)-[:HAS_IDENTIFIER]->(i)
            """,
            params={"rows": rows},
        )

        agencies = [
            {
                "document_id": document.id,
                "key": normalize_entity_key(document.agency),
                "name": document.agency,
            }
            for document in documents
            if document.agency
        ]
        self._upsert_named_entities(agencies, "Agency", "ISSUED_BY")

        sectors = [
            {
                "document_id": document.id,
                "key": normalize_entity_key(document.sector),
                "name": document.sector,
            }
            for document in documents
            if document.sector
        ]
        self._upsert_named_entities(sectors, "Sector", "IN_SECTOR")

        fields = [
            {
                "document_id": document.id,
                "key": normalize_entity_key(document.field),
                "name": document.field,
            }
            for document in documents
            if document.field
        ]
        self._upsert_named_entities(fields, "Field", "IN_FIELD")

        signers = [
            {
                "document_id": document.id,
                "key": normalize_entity_key(document.signer),
                "name": document.signer,
                "position": document.signer_position,
            }
            for document in documents
            if document.signer
        ]
        if signers:
            self._query(
                """
                UNWIND $rows AS row
                MATCH (d:Document {id: row.document_id})
                MERGE (p:Person {key: row.key})
                SET p.name = row.name
                MERGE (d)-[r:SIGNED_BY]->(p)
                SET r.position = row.position
                """,
                params={"rows": signers},
            )

        mention_rows = [
            {
                "document_id": mention.document_id,
                "number": mention.number,
                "number_norm": mention.number_norm,
                "kinds": list(mention.kinds),
                "count": mention.count,
            }
            for mention in mentions
        ]
        if mention_rows:
            self._query(
                """
                UNWIND $rows AS row
                MATCH (d:Document {id: row.document_id})
                MERGE (i:Instrument {number_norm: row.number_norm})
                SET i.number = row.number
                MERGE (d)-[r:MENTIONS]->(i)
                SET r.kinds = row.kinds, r.count = row.count
                """,
                params={"rows": mention_rows},
            )

    def _upsert_named_entities(
        self,
        rows: Sequence[dict[str, str]],
        label: str,
        relationship: str,
    ) -> None:
        if not rows:
            return
        self._query(
            f"""
            UNWIND $rows AS row
            MATCH (d:Document {{id: row.document_id}})
            MERGE (entity:{label} {{key: row.key}})
            SET entity.name = row.name
            MERGE (d)-[:{relationship}]->(entity)
            """,
            params={"rows": list(rows)},
        )

    def upsert_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return
        rows = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "index": chunk.index,
                "section": chunk.section,
            }
            for chunk in chunks
        ]
        self._query(
            """
            UNWIND $rows AS row
            MATCH (d:Document {id: row.document_id})
            MERGE (c:Chunk {id: row.id})
            SET c.document_id = row.document_id,
                c.chunk_index = row.index,
                c.section = row.section
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            params={"rows": rows},
        )

    def mark_indexed(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        now = datetime.now(UTC).isoformat()
        values = [{**row, "indexed_at": now} for row in rows]
        self._query(
            """
            UNWIND $rows AS row
            MATCH (d:Document {id: row.id})
            SET d.index_hash = row.index_hash,
                d.index_status = 'ready',
                d.chunk_count = row.chunk_count,
                d.indexed_at = row.indexed_at
            """,
            params={"rows": values},
        )

    def expand(self, seed_document_ids: Sequence[str], *, limit: int) -> list[GraphNeighbor]:
        if not seed_document_ids:
            return []
        params = {"seed_ids": list(seed_document_ids), "limit": limit * 3}
        queries = (
            (
                "shared_metadata",
                0.2,
                """
                UNWIND $seed_ids AS seed_id
                MATCH (seed:Document {id: seed_id})
                      -[:ISSUED_BY|IN_SECTOR|IN_FIELD|SIGNED_BY]->(entity)
                      <-[:ISSUED_BY|IN_SECTOR|IN_FIELD|SIGNED_BY]-(related:Document)
                WHERE related.id <> seed.id
                RETURN seed.id, related.id, related.title, related.number,
                       count(DISTINCT entity)
                ORDER BY count(DISTINCT entity) DESC
                LIMIT $limit
                """,
            ),
            (
                "cites",
                1.0,
                """
                UNWIND $seed_ids AS seed_id
                MATCH (seed:Document {id: seed_id})-[:MENTIONS]->(i:Instrument)
                MATCH (i)<-[:HAS_IDENTIFIER]-(candidate:Document)
                WITH seed, i, collect(DISTINCT candidate) AS candidates
                WHERE size(candidates) = 1
                UNWIND candidates AS related
                WITH seed, related
                WHERE related.id <> seed.id
                RETURN seed.id, related.id, related.title, related.number, 1
                LIMIT $limit
                """,
            ),
            (
                "cited_by",
                0.9,
                """
                UNWIND $seed_ids AS seed_id
                MATCH (seed:Document {id: seed_id})-[:HAS_IDENTIFIER]->(i:Instrument)
                MATCH (i)<-[:HAS_IDENTIFIER]-(owner:Document)
                WITH seed, i, count(DISTINCT owner) AS owner_count
                WHERE owner_count = 1
                MATCH (i)<-[:MENTIONS]-(related:Document)
                WHERE related.id <> seed.id
                RETURN seed.id, related.id, related.title, related.number, 1
                LIMIT $limit
                """,
            ),
        )

        seed_set = set(seed_document_ids)
        combined: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "title": "",
                "number": "",
                "score": 0.0,
                "relations": set(),
                "seeds": set(),
                "evidence": [],
            }
        )
        for relation, base_score, query in queries:
            result = self._read(query, params)
            for row in result:
                seed_id, document_id, title, number, strength = row
                document_id = str(document_id)
                if document_id in seed_set:
                    continue
                item = combined[document_id]
                item["title"] = str(title or "")
                item["number"] = str(number or "")
                score = base_score * max(1.0, float(strength or 1))
                item["score"] = min(1.0, float(item["score"]) + score)
                item["relations"].add(relation)
                item["seeds"].add(str(seed_id))

        verified = self._read(
            """
            UNWIND $seed_ids AS seed_id
            MATCH (seed:Document {id: seed_id})-[r:LEGAL_RELATION]-(related:Document)
            WHERE seed.index_status = 'ready' AND related.index_status = 'ready'
            WITH seed, related, r, endNode(r) AS target
            MATCH (target)-[:HAS_IDENTIFIER]->(identifier:Instrument)
            MATCH (identifier)<-[:HAS_IDENTIFIER]-(owner:Document)
            WITH seed, related, r, count(DISTINCT owner) AS owners
            WHERE owners = 1
            RETURN seed.id, related.id, related.title, related.number,
                   startNode(r).id, endNode(r).id, properties(r)
            ORDER BY related.id, r.kind
            """,
            params,
        )
        for seed_id, document_id, title, number, source_id, target_id, props in verified:
            item = combined[str(document_id)]
            item["title"], item["number"] = str(title or ""), str(number or "")
            item["score"] = 1.0
            item["relations"].add(props["kind"])
            item["seeds"].add(str(seed_id))
            relation = LegalRelation(
                source_id=source_id,
                target_id=target_id,
                kind=props["kind"],
                source_chunk_id=props["source_chunk_id"],
                evidence=props["evidence"],
                method=props["method"],
                target_chunk_id=props.get("target_chunk_id", ""),
                provenance=props.get("provenance", ""),
            )
            if relation not in item["evidence"]:
                item["evidence"].append(relation)

        neighbors = [
            GraphNeighbor(
                document_id=document_id,
                title=str(item["title"]),
                number=str(item["number"]),
                graph_score=float(item["score"]),
                relations=tuple(sorted(item["relations"])),
                seed_document_ids=tuple(sorted(item["seeds"])),
                evidence=tuple(item["evidence"]),
            )
            for document_id, item in combined.items()
        ]
        return sorted(
            neighbors, key=lambda item: (bool(item.evidence), item.graph_score), reverse=True
        )[:limit]

    def stats(self) -> dict[str, int]:
        labels = {
            "documents": "Document",
            "chunks": "Chunk",
            "instruments": "Instrument",
            "agencies": "Agency",
            "sectors": "Sector",
            "fields": "Field",
            "people": "Person",
            "legal_concepts": "LegalConcept",
        }
        output: dict[str, int] = {}
        for key, label in labels.items():
            result = self._read(f"MATCH (n:{label}) RETURN count(n)")
            output[key] = int(result[0][0])
        result = self._read("MATCH ()-[r:LEGAL_RELATION]->() RETURN count(r)")
        output["legal_relations"] = int(result[0][0])
        return output

    def health(self) -> bool:
        result = self._read("RETURN 1")
        return bool(result and result[0][0] == 1)

    def relation_catalog(self) -> dict[str, dict[str, Any]]:
        rows = self._read("""
            MATCH (d:Document)
            RETURN d.id, d.index_status, d.index_hash, d.number, d.chunk_count
        """)
        return {
            row[0]: dict(
                zip(("status", "index_hash", "number", "chunk_count"), row[1:], strict=True)
            )
            for row in rows
        }

    def identifier_owners(self) -> dict[str, list[str]]:
        return {
            row[0]: list(row[1])
            for row in self._read("""
            MATCH (d:Document)-[:HAS_IDENTIFIER]->(i:Instrument)
            RETURN i.number_norm, collect(DISTINCT d.id)
        """)
        }

    def entity_candidates(self, source_id: str, *, limit: int) -> list[tuple[str, int]]:
        return [
            (row[0], int(row[1]))
            for row in self._read(
                """
            MATCH (:Document {id: $id})-[:ISSUED_BY|IN_FIELD|IN_SECTOR|SHARES_CONCEPT]->(entity)
            CALL (entity) {
                MATCH (entity)<-[:ISSUED_BY|IN_FIELD|IN_SECTOR|SHARES_CONCEPT]-(target:Document)
                WHERE target.id <> $id AND target.index_status = 'ready'
                RETURN target ORDER BY target.id LIMIT $limit
            }
            RETURN target.id, count(DISTINCT entity) AS overlap
            ORDER BY overlap DESC, target.id LIMIT $limit
        """,
                {"id": source_id, "limit": limit},
            )
        ]

    def upsert_relations(self, relations: Sequence[LegalRelation]) -> None:
        if not relations:
            return
        self._query(
            """
            UNWIND $rows AS row
            MATCH (source:Document {id: row.source_id})
                  -[:HAS_CHUNK]->(:Chunk {id: row.source_chunk_id})
            MATCH (target:Document {id: row.target_id})
            WHERE source.index_status = 'ready' AND target.index_status = 'ready'
            MATCH (target)-[:HAS_IDENTIFIER]->(i:Instrument)
            MATCH (i)<-[:HAS_IDENTIFIER]-(owner:Document)
            WITH source, target, row, count(DISTINCT owner) AS owners
            WHERE owners = 1
            MERGE (source)-[r:LEGAL_RELATION {kind: row.kind}]->(target)
            SET r.source_chunk_id = row.source_chunk_id,
                r.target_chunk_id = row.target_chunk_id, r.evidence = row.evidence,
                r.method = row.method, r.provenance = row.provenance
        """,
            {"rows": [asdict(relation) for relation in relations]},
        )

    def upsert_reviewed_concept(
        self,
        document_id: str,
        chunk: DocumentChunk,
        phrase: str,
        *,
        reviewed_by: str,
    ) -> None:
        """Explicit opt-in for reviewed phrases; never promote TF-IDF features."""
        from .extraction import chunk_body

        if (
            chunk.document_id != document_id
            or not phrase.strip()
            or not reviewed_by.strip()
            or phrase not in chunk_body(chunk.text)
        ):
            raise ValueError("A reviewed concept requires a matching source chunk and exact phrase")
        self._query(
            """
            MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(:Chunk {id: $chunk_id})
            MERGE (concept:LegalConcept {key: $key})
            SET concept.name = $phrase
            MERGE (d)-[r:SHARES_CONCEPT {source_chunk_id: $chunk_id}]->(concept)
            SET r.evidence = $phrase, r.reviewed_by = $reviewed_by
        """,
            {
                "id": document_id,
                "chunk_id": chunk.id,
                "key": normalize_entity_key(phrase),
                "phrase": phrase,
                "reviewed_by": reviewed_by,
            },
        )

    def upsert_structure(self, document: LegalDocument, chunks: Sequence[DocumentChunk]) -> None:
        """Add hierarchy to ready documents without re-embedding or changing legacy IDs."""
        from .structure import map_chunks, parse_structure, structure_hash

        provisions = parse_structure(document)
        signature = structure_hash(document)
        # Invalidate this document's own derived hierarchy only when its text changes.
        self._query(
            """
            MATCH (d:Document {id: $id})
            WHERE d.structure_hash IS NOT NULL AND d.structure_hash <> $hash
            MATCH (p:Provision {document_id: $id}) DETACH DELETE p
        """,
            {"id": document.id, "hash": signature},
        )
        for level in ("Chapter", "Article", "Clause", "Point"):
            rows = [asdict(p) for p in provisions if p.level == level]
            if not rows:
                continue
            self._query(
                f"""
                UNWIND $rows AS row
                MATCH (d:Document {{id: row.document_id}})
                MERGE (p:Provision:{level} {{id: row.id}})
                SET p += row, p.structure_hash = $hash
                MERGE (d)-[:HAS_PROVISION]->(p)
            """,
                {"rows": rows, "hash": signature},
            )
        self._query(
            """
            UNWIND $rows AS row
            MATCH (child:Provision {id: row.id})
            MATCH (parent) WHERE parent.id = row.parent_id
                AND (parent:Document OR parent:Provision)
            MERGE (parent)-[:CONTAINS]->(child)
        """,
            {"rows": [asdict(p) for p in provisions]},
        )
        mapping = map_chunks(document, list(chunks))
        self._query(
            """
            UNWIND $rows AS row
            MATCH (a:Article {id: row.article_id}), (c:Chunk {id: row.chunk_id})
            MERGE (a)-[:HAS_CHUNK]->(c)
            SET c.article_id = a.id
        """,
            {"rows": [{"chunk_id": cid, "article_id": aid} for cid, aid in mapping.items()]},
        )
        self._query(
            "MATCH (d:Document {id: $id}) SET d.structure_hash = $hash",
            {"id": document.id, "hash": signature},
        )

    def upsert_provision_relations(self, relations) -> None:
        import hashlib

        rows = []
        for relation in relations:
            row = asdict(relation)
            row["key"] = hashlib.sha256(
                "\x1f".join(
                    (
                        relation.source_id,
                        relation.target_id,
                        relation.kind,
                        relation.source_chunk_id,
                        relation.target_chunk_id,
                    )
                ).encode()
            ).hexdigest()
            rows.append(row)
        self._query(
            """
            UNWIND $rows AS row
            MATCH (s:Article {id: row.source_id})-[:HAS_CHUNK]->(:Chunk {id: row.source_chunk_id})
            MATCH (t:Article {id: row.target_id})-[:HAS_CHUNK]->(:Chunk {id: row.target_chunk_id})
            MATCH (sd:Document {id: s.document_id}), (td:Document {id: t.document_id})
            WHERE sd.index_status = 'ready' AND td.index_status = 'ready'
            MERGE (s)-[r:PROVISION_RELATION {key: row.key}]->(t)
            SET r += row, r.source_hash = s.structure_hash, r.target_hash = t.structure_hash
        """,
            {"rows": rows},
        )

    def upsert_legal_facts(self, facts: list[dict]) -> None:
        """Grounded concepts/entities and normative spans remain attached to their article."""
        self._query(
            """
            UNWIND $rows AS row
            MATCH (a:Article {id: row.article_id})-[:HAS_CHUNK]->(:Chunk {id: row.chunk_id})
            MERGE (f:LegalFact {id: row.id}) SET f += row
            MERGE (a)-[r:EXPRESSES {kind: row.kind}]->(f)
            SET r.evidence = row.evidence, r.source_chunk_id = row.chunk_id
            FOREACH (_ IN CASE WHEN row.kind IN ['DEFINES', 'REGULATES'] THEN [1] ELSE [] END |
                MERGE (c:LegalConcept {key: row.concept_key}) SET c.name = row.name
                MERGE (a)-[link:CONCEPT_RELATION {kind: row.kind, fact_id: row.id}]->(c)
                SET link.evidence = row.evidence, link.source_chunk_id = row.chunk_id,
                    link.provenance = row.provenance, link.source_id = a.id,
                    link.target_id = c.key, link.source_hash = a.structure_hash,
                    link.valid_from = '', link.valid_to = '', link.temporal_status = 'unknown')
        """,
            {"rows": facts},
        )

    def expand_articles(self, seed_article_ids, *, config, as_of: str = ""):
        """Bounded BFS. Traverse either way, retaining the edge's actual legal direction."""
        from .models import ArticleCandidate, ProvisionRelation

        if config.mode == "M0" or not seed_article_ids:
            return []
        frontier = {aid: (1.0, ()) for aid in seed_article_ids[: config.seed_limit]}
        visited = set(frontier)
        output = {}
        for depth in range(config.depth):
            if not frontier or len(output) >= config.candidate_budget:
                break
            params = {
                "ids": sorted(frontier),
                "kinds": config.kinds,
                "confidence": config.confidence,
                "per_node": config.per_node_limit,
                "as_of": as_of,
                "weights": config.weights,
            }
            rows = (
                self._read(
                    """
                UNWIND $ids AS seed_id
                MATCH (seed:Article {id: seed_id})
                CALL (seed) {
                    MATCH (seed)-[r:PROVISION_RELATION]-(a:Article)
                    WHERE r.kind IN $kinds AND r.confidence >= $confidence
                        AND r.source_hash = startNode(r).structure_hash
                        AND r.target_hash = endNode(r).structure_hash
                        AND ($as_of = '' OR (r.temporal_status = 'known'
                            AND (r.valid_from = '' OR r.valid_from <= $as_of)
                            AND (r.valid_to = '' OR $as_of < r.valid_to)))
                    MATCH (d:Document {id: a.document_id}) WHERE d.index_status = 'ready'
                    RETURN a, r ORDER BY r.confidence * coalesce($weights[r.kind], 0) DESC,
                        a.id, r.key LIMIT $per_node
                }
                MATCH (a)-[:HAS_CHUNK]->(c:Chunk)
                RETURN seed.id, a.id, collect(DISTINCT c.id), properties(r)
            """,
                    params,
                )
                if config.kinds
                else []
            )
            if config.mode in {"M1", "M4"}:
                rows += self._read(
                    """
                    UNWIND $ids AS seed_id
                    MATCH (seed:Article {id: seed_id})
                    CALL (seed) {
                        MATCH (parent)-[:CONTAINS]->(seed), (parent)-[:CONTAINS]->(a:Article)
                        WHERE a.id <> seed.id AND EXISTS { MATCH (a)-[:HAS_CHUNK]->(:Chunk) }
                        RETURN a ORDER BY abs(a.start - seed.start), a.id LIMIT $per_node
                    }
                    MATCH (a)-[:HAS_CHUNK]->(c:Chunk)
                    RETURN seed.id, a.id, collect(DISTINCT c.id), null
                """,
                    params,
                )
            next_frontier = {}
            proposals = []
            for seed, aid, chunk_ids, props in rows:
                if aid in visited:
                    continue
                previous_score, path = frontier[seed]
                if props:
                    relation = ProvisionRelation(
                        **{
                            k: props[k]
                            for k in ProvisionRelation.__dataclass_fields__
                            if k in props
                        }
                    )
                    score = (
                        previous_score * config.weights.get(relation.kind, 0) * relation.confidence
                    )
                    path = (*path, relation)
                else:
                    score = previous_score * config.weights.get("STRUCTURAL", 0)
                if depth:
                    score *= config.decay
                proposals.append(
                    ArticleCandidate(aid, tuple(sorted(chunk_ids)), score, path, props is None)
                )
            for item in sorted(proposals, key=lambda x: (-x.score, x.article_id)):
                if item.article_id in next_frontier or item.score <= 0:
                    continue
                output[item.article_id] = item
                next_frontier[item.article_id] = (item.score, item.path)
                if len(output) >= config.candidate_budget:
                    break
            visited.update(next_frontier)
            frontier = next_frontier
        return list(output.values())

    def chunk_article_mapping(self, document_ids=None):
        rows = self._read(
            """
            MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk)
            WHERE $ids IS NULL OR a.document_id IN $ids
            RETURN c.id, collect(DISTINCT a.id)
        """,
            {"ids": document_ids},
        )
        return {cid: aids[0] for cid, aids in rows if len(aids) == 1}

    def concept_candidates(self, article_ids, *, limit):
        """Previously grounded concepts discover candidates, never prove a relation."""
        return self._read(
            """
            UNWIND $ids AS aid
            MATCH (s:Article {id: aid})-[sr:CONCEPT_RELATION]->(concept:LegalConcept)
            CALL (s, concept, sr) {
                MATCH (t:Article)-[tr:CONCEPT_RELATION]->(concept)
                WHERE t.id <> s.id
                RETURN t, tr ORDER BY t.id LIMIT $limit
            }
            RETURN s.id, t.id, sr.source_chunk_id, tr.source_chunk_id
        """,
            {"ids": list(article_ids), "limit": limit},
        )

    def link_article_chunks(self, mapping):
        """Attach externally text-validated mappings, including repeated-heading occurrences."""
        self._query(
            """
            UNWIND $rows AS row
            MATCH (a:Article {id: row.article_id}), (c:Chunk {id: row.chunk_id})
            WHERE a.document_id = c.document_id
            MERGE (a)-[:HAS_CHUNK]->(c) SET c.article_id = a.id
        """,
            {"rows": [{"chunk_id": cid, "article_id": aid} for cid, aid in mapping.items()]},
        )
