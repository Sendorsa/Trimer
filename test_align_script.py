#!/usr/bin/env python3
"""
test_align_script.py - Unit & Integration Test Suite for Updated align_script.py
"""

import unittest
from align_script import (
    align_script,
    clean_text,
    detect_element_type,
    parse_script_hierarchical,
    ScriptElementType,
    COMMON_ASR_NORMALIZATIONS
)


class TestHierarchicalAlignScript(unittest.TestCase):

    def setUp(self):
        # Educational lecture sample script with headings, subheadings, bullets, and numbers
        self.script_text = """
What is Gene Mapping?
Types of Gene Mapping
1.
Genetic Mapping (Linkage Mapping)

Advantages:
High Accuracy
- Fast sequencing

This sentence was never spoken in the video.
        """

        # Simulated transcript from transcript.py
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

            # Merged Numbered Item: "1. Genetic Mapping (Linkage Mapping)"
            ("1.", "00:00:07.000", "00:00:07.300"),
            ("genetic", "00:00:07.300", "00:00:07.800"),
            ("mapping", "00:00:08.300", "00:00:08.900"),
            ("linkage", "00:00:08.900", "00:00:09.500"),
            ("mapping.", "00:00:09.500", "00:00:10.000"),

            # Merged Subheading + Content: "Advantages: High Accuracy"
            ("advantages", "00:00:12.000", "00:00:12.400"),
            ("high", "00:00:12.400", "00:00:12.800"),
            ("accuracy.", "00:00:12.800", "00:00:13.400"),

            # Bullet: "- Fast sequencing" (With ASR error: "sangal" -> "sanger")
            ("fast", "00:00:15.000", "00:00:15.300"),
            ("sangal", "00:00:15.300", "00:00:15.800"),
            ("sequencing.", "00:00:15.800", "00:00:16.400")
        ]

    def test_element_type_detection(self):
        self.assertEqual(detect_element_type("What is Gene Mapping?"), ScriptElementType.HEADING)
        self.assertEqual(detect_element_type("Advantages:"), ScriptElementType.SUBHEADING)
        self.assertEqual(detect_element_type("1."), ScriptElementType.NUMBERED_ITEM)
        self.assertEqual(detect_element_type("- High Accuracy"), ScriptElementType.BULLET)
        self.assertEqual(detect_element_type("This is a regular lecture sentence."), ScriptElementType.SENTENCE)

    def test_structural_merging(self):
        sections = parse_script_hierarchical(self.script_text)
        all_sents = []
        for sec in sections:
            if sec.heading_sentence:
                all_sents.append(sec.heading_sentence.raw_text)
            for b in sec.blocks:
                for s in b.sentences:
                    all_sents.append(s.raw_text)

        # Check that "1." was merged with "Genetic Mapping (Linkage Mapping)"
        self.assertTrue(any("1. Genetic Mapping" in s for s in all_sents))
        # Check that "Advantages:" was merged with "High Accuracy"
        self.assertTrue(any("Advantages: High Accuracy" in s for s in all_sents))

    def test_asr_normalization(self):
        cleaned = clean_text("fast sangal sequencing")
        self.assertIn("sanger", cleaned)

    def test_end_to_end_hierarchical_alignment(self):
        results = align_script(
            self.transcript_tuples,
            self.script_text,
            min_confidence=70.0
        )

        self.assertTrue(len(results) > 0)

        # Check that confidence levels exist (HIGH, MEDIUM, LOW)
        matched_results = [r for r in results if r.get("matched") is not False]
        self.assertTrue(len(matched_results) > 0)
        for r in matched_results:
            self.assertIn("confidence_level", r)
            self.assertIn(r["confidence_level"], ("HIGH", "MEDIUM", "LOW"))

        # Unspoken sentence test
        unspoken = [r for r in results if "never spoken" in r["sentence"]]
        self.assertEqual(len(unspoken), 1)
        self.assertFalse(unspoken[0].get("matched", True))


if __name__ == "__main__":
    unittest.main()
