#!/usr/bin/env python3
"""
main.py - Bumblebee Entry Point

Connects Stage 1 (in-memory speech-to-text transcript generation) directly
with Stage 2 (in-memory script alignment engine with Persistent Logger).

Usage:
    python3 main.py <video_file_path> <script_file_path> [--debug-alignment]
"""

import sys
import json
import argparse
from transcript import generate_transcript
from align_script import align_script, create_run_logger, DEBUG_ALIGNMENT


def main():
    parser = argparse.ArgumentParser(description="Bumblebee Video Trimming Pipeline Entry Point")
    parser.add_argument("video_path", help="Path to input video/audio file")
    parser.add_argument("script_path", help="Path to ground-truth script text file")
    parser.add_argument("--min-confidence", type=float, default=70.0, help="Minimum normal confidence threshold (default: 70.0)")
    parser.add_argument("--debug-alignment", action="store_true", default=DEBUG_ALIGNMENT, help="Enable detailed alignment diagnostics & failure analytics")

    args = parser.parse_args()

    # Initialize persistent Tee logger for alignment run
    with create_run_logger() as logger:
        try:
            # Stage 1: Generate transcript in memory
            transcript = generate_transcript(args.video_path)

            # Stage 2: Align script using in-memory transcript
            results = align_script(
                transcript,
                args.script_path,
                min_confidence=args.min_confidence,
                debug_alignment=args.debug_alignment
            )

            # Print alignment output JSON
            print(json.dumps(results, indent=2))
        except Exception as e:
            print(f"Error executing Bumblebee pipeline: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
