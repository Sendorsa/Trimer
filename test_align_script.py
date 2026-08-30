#!/usr/bin/env python3
"""
test_align_script.py - Unit & Integration Test Suite for align_script.py
Tests Section Heading Anchors, Block Region Expansion, and Neighbor-Based Rescue Pass
"""

import unittest
from align_script import (
    align_script,
    clean_text,
    detect_element_type,
    parse_script_hierarchical,
    ScriptElementType,
    RejectionReason,
    compute_adaptive_search_window,
    BLOCK_EXPANSION_MARGIN,
    NEIGHBOR_RESCUE_MARGIN
)


class TestCandidateDiscoveryAlignScript(unittest.TestCase):

    def setUp(self):
        # Sample educational script
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

            # Block 1
            ("1.", "00:00:07.000", "00:00:07.300"),
            ("genetic", "00:00:07.300", "00:00:07.800"),
            ("mapping", "00:00:07.800", "00:00:08.300"),
            ("linkage", "00:00:08.300", "00:00:08.900"),
            ("mapping.", "00:00:08.900", "00:00:09.500"),

            # Block 2
            ("advantages", "00:00:12.000", "00:00:12.400"),
            ("high", "00:00:12.400", "00:00:12.800"),
            ("accuracy.", "00:00:12.800", "00:00:13.400"),

            # Block 3
            ("limitations", "00:00:15.000", "00:00:15.400"),
            ("time", "00:00:15.400", "00:00:15.800"),
            ("consuming.", "00:00:15.800", "00:00:16.300"),

            # Block 4
            ("applications", "00:00:18.000", "00:00:18.400"),
            ("whole", "00:00:18.400", "00:00:18.800"),
            ("genome", "00:00:18.800", "00:00:19.200"),
            ("sequencing.", "00:00:19.200", "00:00:19.700")
        ]

    def test_block_expansion_margins(self):
        self.assertEqual(BLOCK_EXPANSION_MARGIN, 150)
        self.assertEqual(NEIGHBOR_RESCUE_MARGIN, 20)

    def test_section_anchors_and_block_expansion(self):
        sections = parse_script_hierarchical(self.script_text)
        results = align_script(
            self.transcript_tuples,
            self.script_text,
            min_confidence=70.0,
            debug_alignment=True
        )

        self.assertTrue(len(results) > 0)
        matched_results = [r for r in results if r.get("matched")]
        self.assertTrue(len(matched_results) >= 6)

    def test_neighbor_rescue_pass_logging(self):
        results = align_script(
            self.transcript_tuples,
            self.script_text,
            min_confidence=70.0,
            debug_alignment=True
        )

        unspoken = [r for r in results if "never spoken" in r["sentence"]]
        self.assertEqual(len(unspoken), 1)
        self.assertFalse(unspoken[0].get("matched", True))


if __name__ == "__main__":
    unittest.main()
