"""Protect the semantic-edge ablation and shared comparison budgets."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_openrouter_experiment import configurations


class CloudConfigurationTests(unittest.TestCase):
    def test_semantic_relations_enabled_only_in_enriched_graph_modes(self):
        configs = configurations()
        for name in ["graph_shared", "graph_shared_path", "graph_legacy", "shuffled_shared"]:
            self.assertIn("SPECIFIES", configs[name].kinds)
            self.assertIn("EXCEPTION_TO", configs[name].kinds)
        self.assertNotIn("SPECIFIES", configs["citation_shared"].kinds)
        self.assertEqual(configs["hybrid"].kinds, [])
        for name, c in configs.items():
            self.assertEqual(c.context_chars, 18000)
            self.assertEqual(c.candidate_budget, 120 if name == "hybrid_larger" else 60)
        self.assertEqual(configs["graph_shared"].scoring, configs["hybrid_shared"].scoring)
        self.assertEqual(configs["shuffled_shared"].scoring, configs["graph_shared"].scoring)
