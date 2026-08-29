#!/usr/bin/env python3
"""
test_align_script.py - Unit & Integration Test Suite for align_script.py with Stage 3 Phonetic Rescue
"""

import unittest
from align_script import (
    align_script,
    clean_text,
    phonetic_normalize,
    detect_element_type,
    parse_script_hierarchical,
    ScriptElementType,
    COMMON_ASR_NORMALIZATIONS
)


class TestPhoneticRescueAlignScript(unittest.TestCase):

    def setUp(self):
        # Sample script with misheard domain terms
        self.script_text = """
The unit used is called a centimorgan.
We performed Sanger sequencing.
Follow the roadmap for gene mapping.
Recombination occurs during crossing over.
This sentence was never spoken in the video.
        """

        # Simulated transcript where Whisper misheard target terms phonetically
        self.transcript_tuples = [
            # 1. centimorgan -> centi morgan
            ("The", "00:00:01.000", "00:00:01.200"),
            ("unit", "00:00:01.200", "00:00:01.400"),
            ("used", "00:00:01.400", "00:00:01.600"),
            ("is", "00:00:01.600", "00:00:01.800"),
            ("called", "00:00:01.800", "00:00:02.100"),
            ("a", "00:00:02.100", "00:00:02.200"),
            ("centi", "00:00:02.200", "00:00:02.600"),
            ("morgan.", "00:00:02.600", "00:00:03.100"),

            # 2. Sanger -> Sangal
            ("We", "00:00:05.000", "00:00:05.200"),
            ("performed", "00:00:05.200", "00:00:05.600"),
            ("Sangal", "00:00:05.600", "00:00:06.100"),
            ("sequencing.", "00:00:06.100", "00:00:06.800"),

            # 3. roadmap -> load map
            ("Follow", "00:00:09.000", "00:00:09.300"),
            ("the", "00:00:09.300", "00:00:09.500"),
            ("load", "00:00:09.500", "00:00:09.800"),
            ("map", "00:00:09.800", "00:00:10.100"),
            ("for", "00:00:10.100", "00:00:10.300"),
            ("gene", "00:00:10.300", "00:00:10.600"),
            ("mapping.", "00:00:10.600", "00:00:11.100"),

            # 4. crossing over -> cross over
            ("Recombination", "00:00:13.000", "00:00:13.600"),
            ("occurs", "00:00:13.600", "00:00:14.000"),
            ("during", "00:00:14.000", "00:00:14.300"),
            ("cross", "00:00:14.300", "00:00:14.700"),
            ("over.", "00:00:14.700", "00:00:15.200")
        ]

    def test_phonetic_normalization(self):
        p1 = phonetic_normalize("Sanger sequencing")
        p2 = phonetic_normalize("Sangal sequencing")
        self.assertIn("SKNSNK", p1)
        self.assertIn("SKNSNK", p2)

    def test_phonetic_rescue_matches(self):
        results = align_script(
            self.transcript_tuples,
            self.script_text,
            min_confidence=85.0,
            phonetic_threshold=80.0
        )

        self.assertEqual(len(results), 5)

        # Matched rescued sentences
        matched_results = [r for r in results if r.get("matched")]
        self.assertTrue(len(matched_results) >= 4)

        for s in matched_results:
            self.assertIn(s.get("match_type"), ("normal", "phonetic"))

        # Unspoken sentence (false positive prevention)
        s5 = results[4]
        self.assertFalse(s5.get("matched", True))


if __name__ == "__main__":
    unittest.main()
