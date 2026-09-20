from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from .config import Settings
from .extraction import normalize_document_number, normalize_entity_key
from .models import DocumentChunk, GraphNeighbor, LegalDocument, Mention


class GraphStore:
    METADATA_RELATIONSHIPS = (
        "HAS_IDENTIFIER",
        "ISSUED_BY",
        "IN_SECTOR",
        "IN_FIELD",
        "SIGNED_BY",
        "MENTIONS",
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
            "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
            "CREATE INDEX IF NOT EXISTS FOR (i:Instrument) ON (i.number_norm)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Agency) ON (a.key)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Sector) ON (s.key)",
            "CREATE INDEX IF NOT EXISTS FOR (f:Field) ON (f.key)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.key)",
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

        neighbors = [
            GraphNeighbor(
                document_id=document_id,
                title=str(item["title"]),
                number=str(item["number"]),
                graph_score=float(item["score"]),
                relations=tuple(sorted(item["relations"])),
                seed_document_ids=tuple(sorted(item["seeds"])),
            )
            for document_id, item in combined.items()
        ]
        return sorted(neighbors, key=lambda item: item.graph_score, reverse=True)[:limit]

    def stats(self) -> dict[str, int]:
        labels = {
            "documents": "Document",
            "chunks": "Chunk",
            "instruments": "Instrument",
            "agencies": "Agency",
            "sectors": "Sector",
            "fields": "Field",
            "people": "Person",
        }
        output: dict[str, int] = {}
        for key, label in labels.items():
            result = self._read(f"MATCH (n:{label}) RETURN count(n)")
            output[key] = int(result[0][0])
        return output

    def health(self) -> bool:
        result = self._read("RETURN 1")
        return bool(result and result[0][0] == 1)
