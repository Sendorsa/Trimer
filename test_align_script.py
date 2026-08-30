#!/usr/bin/env python3
"""
test_align_script.py - Unit & Integration Test Suite for align_script.py
Tests Section Heading Hard Anchors, Block Region Expansion, Neighbor Rescue Pass,
and Persistent Run Logging.
"""

import os
import sys
import glob
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
    BLOCK_EXPANSION_MARGIN,
    NEIGHBOR_RESCUE_MARGIN
)


class TestCandidateDiscoveryAlignScript(unittest.TestCase):

    def setUp(self):
        # Educational script with headings, blocks, and unspoken content
        self.script_text = """
What is Gene Mapping?
Types of Gene Mapping
1.
Genetic Mapping (Linkage Mapping)

Advantages:
High Accuracy

Limitations:
Time Consuming

Applications:
Whole Genome Sequencing

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

            # Block 1: "1. Genetic Mapping (Linkage Mapping)"
            ("1.", "00:00:07.000", "00:00:07.300"),
            ("genetic", "00:00:07.300", "00:00:07.800"),
            ("mapping", "00:00:07.800", "00:00:08.300"),
            ("linkage", "00:00:08.300", "00:00:08.900"),
            ("mapping.", "00:00:08.900", "00:00:09.500"),

            # Block 2: "Advantages: High Accuracy"
            ("advantages", "00:00:12.000", "00:00:12.400"),
            ("high", "00:00:12.400", "00:00:12.800"),
            ("accuracy.", "00:00:12.800", "00:00:13.400"),

            # Block 3: "Limitations: Time Consuming"
            ("limitations", "00:00:15.000", "00:00:15.400"),
            ("time", "00:00:15.400", "00:00:15.800"),
            ("consuming.", "00:00:15.800", "00:00:16.300"),

            # Block 4: "Applications: Whole Genome Sequencing"
            ("applications", "00:00:18.000", "00:00:18.400"),
            ("whole", "00:00:18.400", "00:00:18.800"),
            ("genome", "00:00:18.800", "00:00:19.200"),
            ("sequencing.", "00:00:19.200", "00:00:19.700")
        ]
        self.created_logs = []

    def tearDown(self):
        for log_f in self.created_logs:
            if os.path.exists(log_f):
                try:
                    os.remove(log_f)
                except Exception:
                    pass

    def test_persistent_run_logger(self):
        with create_run_logger() as ctx:
            log_filepath = ctx.filepath
            self.created_logs.append(log_filepath)
            self.assertTrue(os.path.exists(log_filepath))
            self.assertTrue(ctx.filename.startswith("alignment_run_"))
            self.assertTrue(ctx.filename.endswith(".txt"))
            
            print("Test persistent stdout write")
            print("Test persistent stderr write", file=sys.stderr)

        with open(log_filepath, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("BUMBLEBEE ALIGNMENT RUN LOG", content)
        self.assertIn("Test persistent stdout write", content)
        self.assertIn("Test persistent stderr write", content)
        self.assertIn("RUNTIME SUMMARY", content)

    def test_section_anchor_creation_and_restriction(self):
        results = align_script(
            self.transcript_tuples,
            self.script_text,
            min_confidence=70.0,
            debug_alignment=False
        )

        self.assertTrue(len(results) > 0)
        matched_results = [r for r in results if r.get("matched")]
        self.assertTrue(len(matched_results) >= 6)

    def test_block_expansion_region_calculation(self):
        self.assertEqual(BLOCK_EXPANSION_MARGIN, 150)
        sections = parse_script_hierarchical(self.script_text)
        align_script(self.transcript_tuples, self.script_text, min_confidence=70.0)
        for sec in sections:
            for b in sec.blocks:
                if b.start_idx is not None:
                    expected_exp_start = max(0, b.start_idx - 150)
                    self.assertTrue(b.expanded_start_idx >= expected_exp_start)

    def test_neighbor_rescue_pass_success_and_failure(self):
        self.assertEqual(NEIGHBOR_RESCUE_MARGIN, 20)
        results = align_script(
            self.transcript_tuples,
            self.script_text,
            min_confidence=70.0,
            debug_alignment=True
        )

        unspoken = [r for r in results if "never spoken" in r["sentence"]]
        self.assertEqual(len(unspoken), 1)
        self.assertFalse(unspoken[0].get("matched", True))
        self.assertEqual(unspoken[0].get("rejection_reason"), RejectionReason.BELOW_THRESHOLD.value)


if __name__ == "__main__":
    unittest.main()
