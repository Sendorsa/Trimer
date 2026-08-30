#!/usr/bin/env python3
"""
test_align_script.py - Unit & Integration Test Suite for align_script.py
Tests Context-Aware Alignment Scoring (roadmap vs load map, Sanger vs sangal,
nucleotides vs nuclear types, perfect match), Section Heading Hard Anchors,
Expanded Block Regions, and Neighbor Rescue Pass.
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
    score_candidate_context_aware,
    extract_context_words,
    ScriptSentence,
    TranscriptWord,
    BLOCK_EXPANSION,
    NEIGHBOR_RESCUE_MARGIN
)


class TestContextAwareScoringAlignScript(unittest.TestCase):

    def _make_script_sent(self, text, sent_id=0):
        words = clean_text(text).split()
        return ScriptSentence(
            sentence_id=sent_id,
            raw_text=text,
            clean_text=clean_text(text),
            words=words,
            word_count=len(words)
        )

    def _make_transcript_words(self, text_words):
        t_words = []
        for idx, w in enumerate(text_words):
            cw = clean_text(w)
            t_words.append(TranscriptWord(
                raw_word=w,
                clean_word=cw,
                start_sec=float(idx),
                end_sec=float(idx) + 0.5,
                start_fmt=f"00:00:0{idx}.000",
                end_fmt=f"00:00:0{idx}.500",
                index=idx
            ))
        return t_words

    def test_case1_roadmap_vs_load_map_with_context(self):
        script_prev = self._make_script_sent("Follow the instructions carefully.", 0)
        script_curr = self._make_script_sent("Please follow the roadmap for gene sequencing.", 1)
        script_next = self._make_script_sent("This completes the process.", 2)
        all_sents = [script_prev, script_curr, script_next]

        # Transcript mishears "roadmap" as "load map"
        cand_text = "follow the instructions carefully please follow the load map for gene sequencing this completes the process".split()
        t_words = self._make_transcript_words(cand_text)

        # Candidate slice for "please follow the load map for gene sequencing" (indices 4 to 12)
        _, _, ctx_score, _, _, final_score = score_candidate_context_aware(
            script_curr, 1, all_sents, 4, 12, t_words
        )

        # Verify context score is high due to surrounding matching context
        self.assertTrue(ctx_score >= 80.0)
        self.assertTrue(final_score >= 70.0)

    def test_case2_sanger_vs_sangal_with_context(self):
        script_prev = self._make_script_sent("DNA sequencing has evolved.", 0)
        script_curr = self._make_script_sent("We performed Sanger sequencing on all samples.", 1)
        script_next = self._make_script_sent("Results were accurate.", 2)
        all_sents = [script_prev, script_curr, script_next]

        # Transcript mishears "Sanger" as "sangal"
        cand_text = "dna sequencing has evolved we performed sangal sequencing on all samples results were accurate".split()
        t_words = self._make_transcript_words(cand_text)

        _, _, ctx_score, _, _, final_score = score_candidate_context_aware(
            script_curr, 1, all_sents, 4, 11, t_words
        )

        self.assertTrue(ctx_score >= 80.0)
        self.assertTrue(final_score >= 70.0)

    def test_case3_nucleotides_vs_nuclear_types_with_context(self):
        script_prev = self._make_script_sent("Gene sequencing is important.", 0)
        script_curr = self._make_script_sent("It determines the exact order of nucleotides in DNA.", 1)
        script_next = self._make_script_sent("This enables mapping.", 2)
        all_sents = [script_prev, script_curr, script_next]

        # Transcript mishears "nucleotides" as "nuclear types"
        cand_text = "gene sequencing is important it determines the exact order of nuclear types in dna this enables mapping".split()
        t_words = self._make_transcript_words(cand_text)

        _, _, ctx_score, _, _, final_score = score_candidate_context_aware(
            script_curr, 1, all_sents, 4, 13, t_words
        )

        self.assertTrue(ctx_score >= 80.0)
        self.assertTrue(final_score >= 70.0)

    def test_case4_perfect_transcript_match(self):
        script_prev = self._make_script_sent("Introduction to genomics.", 0)
        script_curr = self._make_script_sent("Gene mapping identifies marker locations.", 1)
        script_next = self._make_script_sent("Conclusion follows.", 2)
        all_sents = [script_prev, script_curr, script_next]

        cand_text = "introduction to genomics gene mapping identifies marker locations conclusion follows".split()
        t_words = self._make_transcript_words(cand_text)

        _, _, ctx_score, _, _, final_score = score_candidate_context_aware(
            script_curr, 1, all_sents, 3, 7, t_words
        )

        self.assertTrue(ctx_score >= 90.0)
        self.assertTrue(final_score >= 85.0)


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


if __name__ == "__main__":
    unittest.main()
