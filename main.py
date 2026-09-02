#!/usr/bin/env python3
"""
main.py - Bumblebee Video Trimming Pipeline Entry Point

Connects Stage 1 (transcript.py), Stage 2 (align_script.py), and Stage 3 (reconstruct_video.py) in memory.

Usage:
    python3 main.py <video_file_path> <script_file_path> [--reconstruct] [--debug-alignment]
"""

import os
import sys
import json
import argparse
from transcript import generate_transcript
from align_script import align_script, create_run_logger, DEBUG_ALIGNMENT
from reconstruct_video import reconstruct_video, MERGE_GAP_SECONDS


def main():
    parser = argparse.ArgumentParser(description="Bumblebee Video Trimming Pipeline Entry Point")
    parser.add_argument("video_path", help="Path to input video/audio file")
    parser.add_argument("script_path", help="Path to ground-truth script text file")
    parser.add_argument("--min-confidence", type=float, default=70.0, help="Minimum normal confidence threshold (default: 70.0)")
    parser.add_argument("--merge-gap", type=float, default=MERGE_GAP_SECONDS, help="Gap threshold in seconds to merge consecutive clips (default: 0.5)")
    parser.add_argument("--reconstruct", action="store_true", help="Execute Stage 3 FFmpeg Reconstruction Engine to produce trimmed output video")
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

            # Stage 3: Reconstruct video if requested
            if args.reconstruct:
                video_dir = os.path.dirname(args.video_path)
                alignment_json_path = os.path.join(video_dir if video_dir else ".", "alignment.json")
                with open(alignment_json_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

                reconstruct_video(
                    video_path=args.video_path,
                    alignment_json_path=alignment_json_path,
                    merge_gap_seconds=args.merge_gap
                )

        except Exception as e:
            print(f"Error executing Bumblebee pipeline: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
