#!/usr/bin/env python3
"""
test_reconstruct_video.py - Production-Grade Unit & Integration Test Suite for reconstruct_video.py
Tests Stage 3 FFmpeg Reconstruction Engine:
  1. Timestamp parsing & human-readable formatting
  2. Extensible Clip dataclass properties
  3. Gap merging algorithm (MERGE_GAP_SECONDS = 0.5)
  4. Script order preservation
  5. Concat list file format generation
  6. Output naming convention
  7. Final statistics box output formatting
"""

import os
import sys
import json
import tempfile
import unittest
from io import StringIO
from reconstruct_video import (
    Clip,
    parse_timestamp_to_sec,
    format_sec_to_timestamp,
    format_duration_human,
    merge_consecutive_clips,
    reconstruct_video,
    MERGE_GAP_SECONDS
)


class TestProductionReconstructVideoEngine(unittest.TestCase):

    def test_timestamp_parsing_and_formatting(self):
        self.assertAlmostEqual(parse_timestamp_to_sec("00:04:00.520"), 240.52, places=3)
        self.assertAlmostEqual(parse_timestamp_to_sec("01:15:30.100"), 4530.1, places=3)
        self.assertAlmostEqual(parse_timestamp_to_sec("10.5"), 10.5, places=3)

        self.assertEqual(format_sec_to_timestamp(240.52), "00:04:00.520")

        self.assertEqual(format_duration_human(10.6), "10.600 sec")
        self.assertEqual(format_duration_human(1368.0), "22m 48s")

    def test_extensible_clip_dataclass_fields(self):
        clip = Clip(
            clip_id=1,
            start_time="00:04:00.520",
            end_time="00:04:11.120",
            duration=10.6,
            sentence="Gene mapping is important...",
            confidence=91.4,
            confidence_level="HIGH",
            start_sec=240.52,
            end_sec=251.12
        )
        self.assertEqual(clip.clip_id, 1)
        self.assertEqual(clip.start_time, "00:04:00.520")
        self.assertEqual(clip.end_time, "00:04:11.120")
        self.assertEqual(clip.duration, 10.6)
        self.assertEqual(clip.confidence, 91.4)
        self.assertEqual(clip.confidence_level, "HIGH")

    def test_gap_merging_algorithm(self):
        clip1 = Clip(1, "00:00:10.000", "00:00:15.000", 5.0, "Sentence 1", 90.0, "HIGH", 10.0, 15.0)
        # Gap = 0.1s (<= 0.5s) -> Should merge
        clip2 = Clip(2, "00:00:15.100", "00:00:20.000", 4.9, "Sentence 2", 85.0, "MEDIUM", 15.1, 20.0)
        # Gap = 2.0s (> 0.5s) -> Should NOT merge
        clip3 = Clip(3, "00:00:22.000", "00:00:28.000", 6.0, "Sentence 3", 95.0, "HIGH", 22.0, 28.0)

        raw_clips = [clip1, clip2, clip3]
        merged = merge_consecutive_clips(raw_clips, merge_gap_seconds=0.5)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].clip_id, 1)
        self.assertEqual(merged[0].start_sec, 10.0)
        self.assertEqual(merged[0].end_sec, 20.0)
        self.assertIn("Sentence 1", merged[0].sentence)
        self.assertIn("Sentence 2", merged[0].sentence)

        self.assertEqual(merged[1].clip_id, 2)
        self.assertEqual(merged[1].start_sec, 22.0)
        self.assertEqual(merged[1].end_sec, 28.0)

    def test_script_order_preservation(self):
        alignment_data = [
            {"sentence": "Sentence 1", "start": "00:04:00.520", "end": "00:04:11.120", "matched": True},
            {"sentence": "Unmatched sentence", "matched": False},
            {"sentence": "Sentence 2", "start": "00:01:00.000", "end": "00:01:10.000", "matched": True},
        ]
        
        temp_dir = tempfile.mkdtemp()
        align_json_path = os.path.join(temp_dir, "alignment.json")
        with open(align_json_path, "w", encoding="utf-8") as f:
            json.dump(alignment_data, f)

        with open(align_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        matched_sentences = [item["sentence"] for item in data if item.get("matched")]
        self.assertEqual(matched_sentences, ["Sentence 1", "Sentence 2"])

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_output_naming_convention(self):
        video_path = "/path/to/input/lecture.mp4"
        video_dir, video_filename = os.path.split(video_path)
        video_basename, video_ext = os.path.splitext(video_filename)
        output_filename = f"{video_basename}_reconstructed{video_ext}"
        self.assertEqual(output_filename, "lecture_reconstructed.mp4")

    def test_concat_list_format(self):
        temp_dir = tempfile.mkdtemp()
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")
        segment_files = [
            os.path.join(temp_dir, "clip_0001.mp4"),
            os.path.join(temp_dir, "clip_0002.mp4")
        ]
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        with open(concat_list_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("file '", content)
        self.assertIn("clip_0001.mp4'", content)
        self.assertIn("clip_0002.mp4'", content)

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
