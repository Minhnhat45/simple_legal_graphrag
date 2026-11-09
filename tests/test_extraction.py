from __future__ import annotations

import unittest

from legal_graphrag.extraction import extract_mentions, normalize_document_number
from legal_graphrag.models import LegalDocument


def make_document(text: str) -> LegalDocument:
    return LegalDocument(
        id="d1",
        title="Văn bản mẫu",
        full_text=text,
        number="01/2026/QĐ-UBND",
        document_type="Quyết định",
        sector="Khoa học và công nghệ",
        issued_date="2026-01-01",
        field="Quản lý khoa học",
        effective_date="2026-01-02",
        status="Còn hiệu lực",
        expiry_date="",
        agency="UBND tỉnh Mẫu",
        signer_position="Chủ tịch",
        signer="Nguyễn Văn A",
    )


class ExtractionTests(unittest.TestCase):
    def test_normalizes_number_spacing_and_case(self) -> None:
        self.assertEqual(
            normalize_document_number(" 46 / 2008 / qđ - ubnd "),
            "46/2008/QĐ-UBND",
        )

    def test_extracts_and_aggregates_legal_mentions(self) -> None:
        mentions = extract_mentions(
            make_document(
                "Căn cứ Luật số 29/2013/QH13. Bãi bỏ Quyết định số 46/2008/QĐ-UBND. "
                "Quyết định 46/2008/QĐ-UBND được nhắc lại. "
                "Số hiệu của văn bản này là 01/2026/QĐ-UBND."
            )
        )
        by_number = {item.number_norm: item for item in mentions}
        self.assertIn("29/2013/QH13", by_number)
        self.assertEqual(by_number["46/2008/QĐ-UBND"].count, 2)
        self.assertIn("repeals", by_number["46/2008/QĐ-UBND"].kinds)
        self.assertNotIn("01/2026/QĐ-UBND", by_number)

    def test_dates_are_not_mistaken_for_document_numbers(self) -> None:
        mentions = extract_mentions(make_document("Ban hành ngày 01/02/2026."))
        self.assertEqual(mentions, [])


class RelationEvidenceTests(unittest.TestCase):
    def test_reference_offsets_exclude_repeated_headers(self) -> None:
        from legal_graphrag.extraction import chunk_body, extract_reference_spans
        from legal_graphrag.models import DocumentChunk

        text = (
            "Title mentions 29/2013/QH13\nsecond title line\n"
            "Số hiệu: 03/2026/QĐ-UBND\nMục: Điều 1\n"
            "Bãi bỏ Quyết định số 01 / 2026 / QĐ-UBND."
        )
        chunk = DocumentChunk("source:0", "source", 0, "Điều 1", text)
        refs = extract_reference_spans(chunk, "03/2026/QĐ-UBND")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].number_norm, "01/2026/QĐ-UBND")
        self.assertEqual(text[refs[0].start : refs[0].end], refs[0].text)
        self.assertTrue(chunk_body(text).startswith("Bãi bỏ"))

    def test_verifier_rejects_malformed_wrong_target_and_header_evidence(self) -> None:
        import json

        from legal_graphrag.models import DocumentChunk
        from legal_graphrag.relation_builder import Candidate, validate_relation

        body = "Bãi bỏ Quyết định số 01/2026/QĐ-UBND."
        chunk = DocumentChunk(
            "s:0",
            "s",
            0,
            "Điều 1",
            "Misleading header\nSố hiệu: 03/2026/QĐ-UBND\nMục: Điều 1\n" + body,
        )
        pair = Candidate("s", "t")
        valid = dict(
            source_id="s", target_id="t", kind="REPEALS", source_chunk_id="s:0", evidence=body
        )

        def verify(data):
            return validate_relation(json.dumps(data), pair, [chunk], [], "01/2026/QĐ-UBND")

        self.assertEqual(verify(valid).kind, "REPEALS")
        for changes in (
            {"target_id": "other"},
            {"kind": "DEFINES"},
            {"source_chunk_id": "t:0"},
            {"target_chunk_id": "fake"},
            {"evidence": "Misleading header"},
            {"evidence": "Invented quote"},
            {"evidence": "01/2026/QĐ-UBND"},
            {"kind": "AMENDS"},
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(verify({**valid, **changes}))
        for data in ([], None, {"kind": "NONE"}):
            self.assertIsNone(verify(data))
        self.assertIsNone(validate_relation("not JSON", pair, [chunk], [], "01/2026/QĐ-UBND"))


if __name__ == "__main__":
    unittest.main()


class SemanticVerificationTests(unittest.TestCase):
    def setUp(self):
        from dataclasses import replace

        from legal_graphrag.models import DocumentChunk

        self.documents = {
            d.id: d
            for d in [
                make_document(""),
                replace(make_document(""), id="d2", number="02/2026/QĐ-UBND"),
            ]
        }
        self.chunks = {
            "s": DocumentChunk(
                "s", "d1", 0, "Điều 1", "Phải có giấy phép kinh doanh để hoạt động."
            ),
            "t": DocumentChunk(
                "t", "d2", 0, "Điều 2", "Hoạt động kinh doanh cần giấy phép kinh doanh."
            ),
        }
        self.articles = {"d1:article:1": ["s"], "d2:article:2": ["t"]}
        self.data = dict(
            source_id="d1:article:1",
            target_id="d2:article:2",
            source_chunk_id="s",
            target_chunk_id="t",
            kind="SETS_CONDITION_FOR",
            evidence=self.chunks["s"].text,
            target_evidence=self.chunks["t"].text,
            confidence=0.9,
            bridge={
                "source_quote": "giấy phép kinh doanh",
                "target_quote": "giấy phép kinh doanh",
                "explanation": "Nguồn quy định giấy phép là điều kiện cần cho hoạt động ở đích.",
            },
        )

    def verify(self, **updates):
        from legal_graphrag.semantic import validate_provision_relation

        return validate_provision_relation(
            {**self.data, **updates}, self.chunks, self.articles, self.documents
        )

    def test_semantic_needs_both_quotes_correct_endpoints_and_bridge(self):
        self.assertIsNotNone(self.verify())
        for update in (
            {"target_evidence": "invented"},
            {"source_chunk_id": "t"},
            {"bridge": {}},
            {"confidence": float("nan")},
            {"confidence": True},
            {"confidence": 0.2},
            {"kind": "NONE"},
            {"kind": "DEFINES"},
        ):
            with self.subTest(update=update):
                self.assertIsNone(self.verify(**update))

    def test_similarity_never_creates_formal_amendments_or_repeals(self):
        for kind in ("AMENDS", "ADDS_TO", "REPEALS", "REPLACES", "CITES"):
            self.assertIsNone(self.verify(kind=kind))

    def test_temporal_overlap_and_unknown_dates(self):
        from dataclasses import replace

        relation = self.verify()
        self.assertEqual(relation.valid_from, "2026-01-02")
        self.documents["d2"] = replace(self.documents["d2"], effective_date="Đã biết")
        self.assertEqual(self.verify().temporal_status, "unknown")
        self.documents["d2"] = replace(
            self.documents["d2"], effective_date="2025-01-01", expiry_date="2025-12-31"
        )
        self.assertIsNone(self.verify())

    def test_concepts_and_facts_require_exact_grounded_spans(self):
        from legal_graphrag.semantic import validate_facts

        fact = dict(
            article_id="d1:article:1",
            chunk_id="s",
            kind="OBLIGATION",
            name="giấy phép kinh doanh",
            evidence=self.chunks["s"].text,
        )
        self.assertEqual(len(validate_facts([fact], self.chunks, self.articles, "{}")), 1)
        self.assertEqual(
            validate_facts([{**fact, "kind": "DEFINES"}], self.chunks, self.articles, "{}"), []
        )
        self.assertEqual(
            validate_facts([{**fact, "name": "invented"}], self.chunks, self.articles, "{}"), []
        )


class VerifierContractTests(unittest.TestCase):
    def test_alias_mapping_never_guesses_unknown_or_wrong_typed_ids(self):
        from legal_graphrag.semantic import resolve_aliases

        response = {
            "relations": [
                {
                    "source_id": "A0",
                    "target_id": ["A1"],
                    "source_chunk_id": "C0",
                    "target_chunk_id": "unknown",
                }
            ]
        }
        result = resolve_aliases(response, {"A0": "real-article"}, {"C0": "real-chunk"})
        self.assertEqual(result["relations"][0]["source_id"], "real-article")
        self.assertEqual(result["relations"][0]["target_id"], "")
        self.assertEqual(result["relations"][0]["target_chunk_id"], "")

    def test_ollama_schema_does_not_change_legacy_chat_contract(self):
        from unittest.mock import patch

        from legal_graphrag.config import Settings
        from legal_graphrag.llm import OllamaClient, OllamaEmptyResponse, OllamaError

        client = OllamaClient(Settings())
        with patch.object(client, "_request", return_value={"message": {"content": "{}"}}) as call:
            client.chat([], json_mode=True)
            self.assertEqual(call.call_args.args[1]["format"], "json")
            self.assertNotIn("num_predict", call.call_args.args[1]["options"])
            schema = {"type": "object"}
            client.chat([], json_schema=schema, max_tokens=2400)
            self.assertEqual(call.call_args.args[1]["format"], schema)
            self.assertEqual(call.call_args.args[1]["options"]["num_predict"], 2400)
            client.chat([], json_schema=schema, max_tokens=2048, think=True)
            self.assertTrue(call.call_args.args[1]["think"])
        with (
            patch.object(client, "_request", return_value={"message": {"content": ""}}),
            self.assertRaises(OllamaEmptyResponse),
        ):
            client.chat([])
        with (
            patch.object(client, "_request", side_effect=OllamaError("connection failed")),
            self.assertRaisesRegex(OllamaError, "connection failed"),
        ):
            client.chat([])


class ScopeVerificationTests(unittest.TestCase):
    def test_reference_to_director_duties_cannot_use_creditor_rights_as_target(self):
        from dataclasses import replace

        from legal_graphrag.models import DocumentChunk, ProvisionRelation
        from legal_graphrag.semantic import referenced_scope_matches

        source = make_document("Điều 18. Những người tại khoản 3 và khoản 4 Điều 5 phải nộp đơn.")
        target = replace(
            make_document(
                "Điều 5. Quyền và nghĩa vụ. "
                "1. Chủ nợ có quyền nộp đơn. 2. Người lao động có quyền nộp đơn. "
                "3. Người đại diện có nghĩa vụ nộp đơn. 4. Chủ doanh nghiệp phải nộp đơn."
            ),
            id="d2",
        )
        chunks = {
            "s": DocumentChunk("s", source.id, 0, "Điều 18", source.full_text),
            "t": DocumentChunk("t", target.id, 0, "Điều 5", target.full_text),
        }
        relation = ProvisionRelation(
            "d1:article:18",
            "d2:article:5",
            "SPECIFIES",
            "s",
            "t",
            "Những người tại khoản 3 và khoản 4 Điều 5 phải nộp đơn.",
            "Chủ nợ có quyền nộp đơn.",
            0.95,
            "{}",
        )
        docs = {d.id: d for d in [source, target]}
        self.assertFalse(referenced_scope_matches(relation, docs, chunks))
        self.assertTrue(
            referenced_scope_matches(
                replace(relation, target_evidence="Người đại diện có nghĩa vụ nộp đơn."),
                docs,
                chunks,
            )
        )


class ArticleCitationTests(unittest.TestCase):
    def test_model_json_accepts_final_fence_but_rejects_ambiguous_or_truncated_output(self):
        from legal_graphrag.semantic import parse_model_json

        expected = {"relations": [], "facts": []}
        import json

        body = json.dumps(expected)
        self.assertEqual(parse_model_json(body), expected)
        self.assertEqual(parse_model_json("Commentary.```json\n" + body + "\n```"), expected)
        for raw in (
            "```json\n" + body,
            "```json\n" + body + "\n``` trailing prose",
            "```json\n{}\n```\n```json\n" + body + "\n```",
            'Text {"relations": []}',
        ):
            with self.assertRaises(ValueError):
                parse_model_json(raw)

    def test_exact_self_citation_keeps_direction_and_never_promotes_repeal(self):
        from legal_graphrag.chunking import chunk_document
        from legal_graphrag.semantic import explicit_article_relations
        from legal_graphrag.structure import map_chunks

        doc = make_document(
            "Điều 1. Bãi bỏ quy định tại Điều 2 của Quyết định này. Điều 2. Quy định kiểm thử."
        )
        chunks = chunk_document(doc, max_chars=1000, overlap=20)
        result = explicit_article_relations([doc], chunks, map_chunks(doc, chunks))
        self.assertEqual(len(result), 1)
        self.assertEqual(
            (result[0].source_id, result[0].target_id, result[0].kind),
            ("d1:article:1", "d1:article:2", "CITES"),
        )
        self.assertEqual(result[0].method, "exact_article_reference")

    def test_named_external_law_without_a_number_is_not_treated_as_self_reference(self):
        from legal_graphrag.chunking import chunk_document
        from legal_graphrag.semantic import explicit_article_relations
        from legal_graphrag.structure import map_chunks

        doc = make_document("Điều 1. Theo Điều 2 của Luật Đất đai. Điều 2. Quy định kiểm thử.")
        chunks = chunk_document(doc, max_chars=1000, overlap=20)
        self.assertEqual(explicit_article_relations([doc], chunks, map_chunks(doc, chunks)), [])
