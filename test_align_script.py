#!/usr/bin/env python3
"""
test_align_script.py - Unit & Integration Test Suite for align_script.py Diagnostics & Adaptive Search
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
    BASE_SEARCH_WINDOW,
    MAX_SEARCH_WINDOW
)


class TestDiagnosticsAlignScript(unittest.TestCase):

    def setUp(self):
        # Educational presentation script
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

            # Structural Block 1: "1. Genetic Mapping (Linkage Mapping)"
            ("1.", "00:00:07.000", "00:00:07.300"),
            ("genetic", "00:00:07.300", "00:00:07.800"),
            ("mapping", "00:00:07.800", "00:00:08.300"),
            ("linkage", "00:00:08.300", "00:00:08.900"),
            ("mapping.", "00:00:08.900", "00:00:09.500"),

            # Structural Block 2: "Advantages: High Accuracy"
            ("advantages", "00:00:12.000", "00:00:12.400"),
            ("high", "00:00:12.400", "00:00:12.800"),
            ("accuracy.", "00:00:12.800", "00:00:13.400"),

            # Structural Block 3: "Limitations: Time Consuming"
            ("limitations", "00:00:15.000", "00:00:15.400"),
            ("time", "00:00:15.400", "00:00:15.800"),
            ("consuming.", "00:00:15.800", "00:00:16.300"),

            # Structural Block 4: "Applications: Whole Genome Sequencing"
            ("applications", "00:00:18.000", "00:00:18.400"),
            ("whole", "00:00:18.400", "00:00:18.800"),
            ("genome", "00:00:18.800", "00:00:19.200"),
            ("sequencing.", "00:00:19.200", "00:00:19.700")
        ]

    def test_adaptive_search_window_calculation(self):
        w1 = compute_adaptive_search_window(False, False, False)
        self.assertEqual(w1, BASE_SEARCH_WINDOW)

        w2 = compute_adaptive_search_window(True, True, True)
        self.assertEqual(w2, BASE_SEARCH_WINDOW + 450)
        self.assertTrue(w2 <= MAX_SEARCH_WINDOW)

    def test_structural_block_recovery(self):
        sections = parse_script_hierarchical(self.script_text)
        all_sents = []
        for sec in sections:
            if sec.heading_sentence:
                all_sents.append(sec.heading_sentence.raw_text)
            for b in sec.blocks:
                for s in b.sentences:
                    all_sents.append(s.raw_text)

        # Verify grouped structural labels
        self.assertTrue(any("Advantages: High Accuracy" in s for s in all_sents))
        self.assertTrue(any("Limitations: Time Consuming" in s for s in all_sents))
        self.assertTrue(any("Applications: Whole Genome Sequencing" in s for s in all_sents))

    def test_alignment_diagnostics_and_failure_analytics(self):
        results = align_script(
            self.transcript_tuples,
            self.script_text,
            min_confidence=70.0,
            debug_alignment=True
        )

        self.assertTrue(len(results) > 0)

        # Verify rejection reason recorded for unspoken sentence
        unspoken = [r for r in results if "never spoken" in r["sentence"]]
        self.assertEqual(len(unspoken), 1)
        self.assertFalse(unspoken[0].get("matched", True))
        self.assertIn("rejection_reason", unspoken[0])
        self.assertIn(unspoken[0]["rejection_reason"], (RejectionReason.NO_CANDIDATES.value, RejectionReason.BELOW_THRESHOLD.value))


if __name__ == "__main__":
    unittest.main()
