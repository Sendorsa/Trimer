#!/usr/bin/env python3
"""
test_align_script.py - Unit & Integration Test Suite for align_script.py
Tests ASR Normalization Layer (apply_asr_normalizations, COMMON_ASR_NORMALIZATIONS),
Section Heading Hard Anchors, Expanded Block Regions, and Neighbor Rescue Pass.
"""

import os
import sys
import unittest
import rapidfuzz
from io import StringIO
from align_script import (
    align_script,
    clean_text,
    apply_asr_normalizations,
    detect_element_type,
    parse_script_hierarchical,
    ScriptElementType,
    RejectionReason,
    compute_adaptive_search_window,
    create_run_logger,
    COMMON_ASR_NORMALIZATIONS,
    BLOCK_EXPANSION,
    NEIGHBOR_RESCUE_MARGIN
)


class TestASRNormalizationAlignScript(unittest.TestCase):

    def test_asr_normalization_phrases(self):
        # 1. road map -> roadmap
        t1 = apply_asr_normalizations("In simple words it acts like a road map")
        self.assertIn("roadmap", t1)
        self.assertNotIn("road map", t1)

        # 2. load map -> roadmap
        t2 = apply_asr_normalizations("In simple words it acts like a load map")
        self.assertIn("roadmap", t2)
        self.assertNotIn("load map", t2)

        # 3. sangal -> sanger
        t3 = apply_asr_normalizations("We performed sangal sequencing")
        self.assertIn("sanger", t3)
        self.assertNotIn("sangal", t3)

        # 4. nuclear types -> nucleotides
        t4 = apply_asr_normalizations("DNA is made of nuclear types")
        self.assertIn("nucleotides", t4)
        self.assertNotIn("nuclear types", t4)

    def test_asr_normalization_higher_similarity_score(self):
        ground_truth = "follow the roadmap for sequencing"
        raw_asr = "follow the load map for sequencing"
        normalized_asr = apply_asr_normalizations(raw_asr)

        raw_score = rapidfuzz.fuzz.ratio(ground_truth, raw_asr)
        normalized_score = rapidfuzz.fuzz.ratio(ground_truth, normalized_asr)

        # Verify normalized string produces higher similarity score than raw string
        self.assertTrue(normalized_score > raw_score)


class TestPhase1CandidateDiscoveryAlignScript(unittest.TestCase):

    def setUp(self):
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

        self.transcript_tuples = [
            ("What", "00:00:01.000", "00:00:01.200"),
            ("is", "00:00:01.200", "00:00:01.400"),
            ("gene", "00:00:01.400", "00:00:01.800"),
            ("mapping?", "00:00:01.800", "00:00:02.300"),
            ("Types", "00:00:04.000", "00:00:04.300"),
            ("of", "00:00:04.300", "00:00:04.500"),
            ("gene", "00:00:04.500", "00:00:04.900"),
            ("mapping.", "00:00:04.900", "00:00:05.400"),
            ("1.", "00:00:07.000", "00:00:07.300"),
            ("genetic", "00:00:07.300", "00:00:07.800"),
            ("mapping", "00:00:07.800", "00:00:08.300"),
            ("linkage", "00:00:08.300", "00:00:08.900"),
            ("mapping.", "00:00:08.900", "00:00:09.500"),
            ("advantages", "00:00:12.000", "00:00:12.400"),
            ("high", "00:00:12.400", "00:00:12.800"),
            ("accuracy", "00:00:12.800", "00:00:13.200"),
            ("reliable", "00:00:13.200", "00:00:13.600"),
            ("results.", "00:00:13.600", "00:00:14.100"),
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

    def test_section_anchor_creation_and_inheritance(self):
        sections = parse_script_hierarchical(self.script_text)
        self.assertTrue(len(sections) > 0)
        
        results = align_script(
            self.transcript_tuples,
            self.script_text,
            min_confidence=70.0,
            debug_alignment=True
        )

        matched = [r for r in results if r.get("matched")]
        self.assertTrue(len(matched) >= 4)

    def test_expanded_block_region_computation(self):
        self.assertEqual(BLOCK_EXPANSION, 150)
        sections = parse_script_hierarchical(self.script_text)
        align_script(self.transcript_tuples, self.script_text, min_confidence=70.0)
        for sec in sections:
            for b in sec.blocks:
                if b.start_idx is not None:
                    expected_exp_start = max(0, b.start_idx - BLOCK_EXPANSION)
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
