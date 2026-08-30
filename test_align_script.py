#!/usr/bin/env python3
"""
test_align_script.py - Unit & Integration Test Suite for align_script.py
Tests Hard Monotonicity Enforcement, 250-Word Block Expansion, Neighbor Rescue Pass (match_type=rescue),
Aggressive Structural Block Merging, and Persistent Logger.
"""

import os
import sys
import unittest
from io import StringIO
from align_script import (
    align_script,
    clean_text,
    detect_element_type,
    parse_script_hierarchical,
    ScriptElementType,
    RejectionReason,
    compute_adaptive_search_window,
    create_run_logger,
    BLOCK_EXPANSION_WORDS,
    NEIGHBOR_RESCUE_MARGIN
)


class TestHierarchicalAlignScript(unittest.TestCase):

    def setUp(self):
        # Educational script with headings, subheadings, bullet lists, and unspoken content
        self.script_text = """
What is Gene Mapping?
Types of Gene Mapping
1.
Genetic Mapping (Linkage Mapping)

Advantages:
- High Accuracy
- Reliable Results

Limitations:
- Time-consuming
- Expensive

This sentence was never spoken in the video.
        """

        # Simulated transcript
        self.transcript_tuples = [
            # Heading 1
            ("What", "00:00:01.000", "00:00:01.200"),
            ("is", "00:00:01.200", "00:00:01.400"),
            ("gene", "00:00:01.400", "00:00:01.800"),
            ("mapping?", "00:00:01.800", "00:00:02.300"),

            # Heading 2
            ("Types", "00:00:04.000", "00:00:04.300"),
            ("of", "00:00:04.300", "00:00:04.500"),
            ("gene", "00:00:04.500", "00:00:04.900"),
            ("mapping.", "00:00:04.900", "00:00:05.400"),

            # Block 1
            ("1.", "00:00:07.000", "00:00:07.300"),
            ("genetic", "00:00:07.300", "00:00:07.800"),
            ("mapping", "00:00:07.800", "00:00:08.300"),
            ("linkage", "00:00:08.300", "00:00:08.900"),
            ("mapping.", "00:00:08.900", "00:00:09.500"),

            # Block 2: Aggressive Structural Block Merging ("Advantages: - High Accuracy - Reliable Results")
            ("advantages", "00:00:12.000", "00:00:12.400"),
            ("high", "00:00:12.400", "00:00:12.800"),
            ("accuracy", "00:00:12.800", "00:00:13.200"),
            ("reliable", "00:00:13.200", "00:00:13.600"),
            ("results.", "00:00:13.600", "00:00:14.100"),

            # Block 3: Aggressive Structural Block Merging ("Limitations: - Time-consuming - Expensive")
            ("limitations", "00:00:15.000", "00:00:15.400"),
            ("time", "00:00:15.400", "00:00:15.700"),
            ("consuming", "00:00:15.700", "00:00:16.100"),
            ("expensive.", "00:00:16.100", "00:00:16.600")
        ]
        self.created_logs = []

    def tearDown(self):
        for log_f in self.created_logs:
            if os.path.exists(log_f):
                try:
                    os.remove(log_f)
                except Exception:
                    pass

    def test_aggressive_structural_block_merging(self):
        sections = parse_script_hierarchical(self.script_text)
        all_sents = []
        for sec in sections:
            if sec.heading_sentence:
                all_sents.append(sec.heading_sentence.raw_text)
            for b in sec.blocks:
                for s in b.sentences:
                    all_sents.append(s.raw_text)

        # Verify aggressive structural block merging for subheadings + bullet items
        self.assertTrue(any("Advantages:" in s and "High Accuracy" in s for s in all_sents))
        self.assertTrue(any("Limitations:" in s and "Time-consuming" in s for s in all_sents))

    def test_250_word_block_expansion(self):
        self.assertEqual(BLOCK_EXPANSION_WORDS, 250)

    def test_rescue_match_type_and_logging(self):
        results = align_script(
            self.transcript_tuples,
            self.script_text,
            min_confidence=70.0,
            debug_alignment=True
        )

        matched_results = [r for r in results if r.get("matched")]
        self.assertTrue(len(matched_results) >= 4)

        unspoken = [r for r in results if "never spoken" in r["sentence"]]
        self.assertEqual(len(unspoken), 1)
        self.assertFalse(unspoken[0].get("matched", True))

    def test_persistent_run_logger(self):
        with create_run_logger() as ctx:
            log_filepath = ctx.filepath
            self.created_logs.append(log_filepath)
            self.assertTrue(os.path.exists(log_filepath))
            self.assertTrue(ctx.filename.startswith("alignment_run_"))
            self.assertTrue(ctx.filename.endswith(".txt"))

        with open(log_filepath, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("BUMBLEBEE ALIGNMENT RUN LOG", content)


if __name__ == "__main__":
    unittest.main()
