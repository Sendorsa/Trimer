#!/usr/bin/env python3
"""
main.py - Bumblebee Entry Point

Connects Stage 1 (in-memory speech-to-text transcript generation) directly
with Stage 2 (in-memory script alignment engine).

Usage:
    python3 main.py <video_file_path> <script_file_path>
"""

import sys
import json
from transcript import generate_transcript
from align_script import align_script


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 main.py <video_file_path> <script_file_path>", file=sys.stderr)
        print("Example: python3 main.py input_video.mp4 script.txt", file=sys.stderr)
        sys.exit(1)

    video_path = sys.argv[1]
    script_path = sys.argv[2]

    try:
        # Stage 1: Generate transcript in memory
        transcript = generate_transcript(video_path)
       
        # Stage 2: Align script using in-memory transcript
        results = align_script(transcript, script_path)

        # Print alignment output
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error executing Bumblebee pipeline: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
