#!/usr/bin/env python3
"""
main.py - Bumblebee Entry Point

Connects Stage 1 (in-memory speech-to-text transcript generation) directly
with Stage 3 (in-memory script alignment engine with Phonetic Rescue).

Usage:
    python3 main.py <video_file_path> <script_file_path> [--phonetic-threshold 85.0] [--debug]
"""

import sys
import json
import argparse
from transcript import generate_transcript
from align_script import align_script


def main():
    parser = argparse.ArgumentParser(description="Bumblebee Video Trimming Pipeline Entry Point")
    parser.add_argument("video_path", help="Path to input video/audio file")
    parser.add_argument("script_path", help="Path to ground-truth script text file")
    parser.add_argument("--phonetic-threshold", type=float, default=85.0, help="Phonetic rescue minimum score threshold (default: 85.0)")
    parser.add_argument("--phonetic-top-candidates", type=int, default=5, help="Number of top candidates evaluated for phonetic rescue (default: 5)")
    parser.add_argument("--min-confidence", type=float, default=70.0, help="Minimum normal confidence threshold (default: 70.0)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging output")

    args = parser.parse_args()

    try:
        # Stage 1: Generate transcript in memory
        transcript = generate_transcript(args.video_path)

        # Stage 2 & 3: Align script using in-memory transcript with Phonetic Rescue
        results = align_script(
            transcript,
            args.script_path,
            min_confidence=args.min_confidence,
            phonetic_threshold=args.phonetic_threshold,
            phonetic_top_candidates=args.phonetic_top_candidates,
            debug=args.debug
        )

        # Print alignment output JSON
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error executing Bumblebee pipeline: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
