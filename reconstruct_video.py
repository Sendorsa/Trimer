#!/usr/bin/env python3
"""
reconstruct_video.py - Bumblebee Stage 3 Production-Grade FFmpeg Video Reconstruction Engine

Consumes alignment results (alignment.json) from Stage 2 and the original input video,
extracting matched script regions using FFmpeg stream copy (-c copy) and concatenating
them into a clean, reconstructed video file preserving original quality, container format, and metadata.

Features:
  1. Extensible Clip Dataclass (clip_id, start_time, end_time, duration, sentence, confidence, confidence_level)
  2. Human-Readable Timestamp Terminal Logging (HH:MM:SS.mmm)
  3. Strict Script Chronology Preservation ( alignment.json ordering)
  4. Consecutive Clip Region Merging (MERGE_GAP_SECONDS = 0.5)
  5. FFmpeg Extraction using argument list subprocess calls (no shell strings)
  6. Demuxer Concatenation (concat_list.txt) with automatic re-encoding fallback logging
  7. Container & Codec Preservation (.mp4, .mov, .mkv)
  8. Output location placement beside source video
  9. Final Statistics Summary Box (INPUT DURATION, OUTPUT DURATION, REMOVED, MATCHED/UNMATCHED, EXTRACTED/MERGED REGIONS)
 10. CLI Interface (--video, --alignment, --merge-gap, --output-dir)
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# Configurable Default Gap Merging Threshold
MERGE_GAP_SECONDS = 0.5


@dataclass
class Clip:
    """
    Extensible metadata container for extracted script video clips.
    Designed for future filtering (by confidence, section, heading, highlights, reels).
    """
    clip_id: int
    start_time: str                  # "HH:MM:SS.mmm"
    end_time: str                    # "HH:MM:SS.mmm"
    duration: float                  # Seconds
    sentence: str
    confidence: Optional[float] = None
    confidence_level: Optional[str] = None
    start_sec: float = 0.0
    end_sec: float = 0.0


def parse_timestamp_to_sec(timestamp_str: str) -> float:
    """
    Timestamp handling: Parses HH:MM:SS.mmm timestamp string into float seconds.
    Supports flexible formats (H:M:S, M:S, S).
    """
    try:
        parts = timestamp_str.split(":")
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds
        elif len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        else:
            return float(parts[0])
    except Exception:
        return 0.0


def format_sec_to_timestamp(seconds: float) -> str:
    """Formats float seconds into HH:MM:SS.mmm human-readable timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def format_duration_human(seconds: float) -> str:
    """Formats float seconds into human-readable duration (e.g. '22m 48s' or '10.600 sec')."""
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{seconds:.3f} sec"


def get_video_duration(video_path: str) -> float:
    """Retrieves total video duration in seconds using ffprobe argument list subprocess."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        # Fallback to ffmpeg -i parsing if ffprobe is unavailable
        cmd_ff = ["ffmpeg", "-i", video_path]
        res = subprocess.run(cmd_ff, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in res.stderr.splitlines():
            if "Duration:" in line:
                dur_str = line.split("Duration:")[1].split(",")[0].strip()
                return parse_timestamp_to_sec(dur_str)
        return 0.0


def merge_consecutive_clips(clips: List[Clip], merge_gap_seconds: float = MERGE_GAP_SECONDS) -> List[Clip]:
    """
    Merge Logic (Requirement 4):
    Merges consecutive clip regions if the gap between them is <= merge_gap_seconds.
    Prevents jumpy video output by creating seamless contiguous extraction regions.
    """
    if not clips:
        return []

    merged: List[Clip] = []
    curr = clips[0]

    for next_clip in clips[1:]:
        gap = next_clip.start_sec - curr.end_sec
        # If consecutive or overlapping within gap threshold, merge them into one extraction region
        if gap <= merge_gap_seconds and next_clip.start_sec >= curr.start_sec:
            merged_duration = next_clip.end_sec - curr.start_sec
            merged_sentence = curr.sentence + " " + next_clip.sentence
            curr = Clip(
                clip_id=curr.clip_id,
                start_time=curr.start_time,
                end_time=next_clip.end_time,
                duration=merged_duration,
                sentence=merged_sentence,
                confidence=max(curr.confidence or 0.0, next_clip.confidence or 0.0),
                confidence_level=curr.confidence_level,
                start_sec=curr.start_sec,
                end_sec=next_clip.end_sec
            )
        else:
            merged.append(curr)
            curr = next_clip

    merged.append(curr)

    # Re-assign sequential clip IDs to merged extraction regions
    for idx, c in enumerate(merged, 1):
        c.clip_id = idx

    return merged


def reconstruct_video(
    video_path: str,
    alignment_json_path: str,
    merge_gap_seconds: float = MERGE_GAP_SECONDS,
    output_dir: Optional[str] = None
) -> str:
    """
    Production-Grade Stage 3 Video Reconstruction Engine:
    1. Loads alignment JSON and extracts matched script clips in chronological script order.
    2. Merges consecutive segments separated by <= merge_gap_seconds.
    3. Extracts individual video segments using FFmpeg stream copy (-c copy).
    4. Concatenates segments into final reconstructed video using FFmpeg demuxer.
    5. Displays human-readable terminal logs and final statistics box.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")
    if not os.path.exists(alignment_json_path):
        raise FileNotFoundError(f"Alignment JSON file not found: {alignment_json_path}")

    with open(alignment_json_path, "r", encoding="utf-8") as f:
        alignment_data = json.load(f)

    # Requirement 7: Format Preservation - Auto-detect container extension
    video_dir, video_filename = os.path.split(video_path)
    video_basename, video_ext = os.path.splitext(video_filename)

    # Requirement 9: Output location placement beside source video
    target_dir = output_dir if output_dir else video_dir
    if not target_dir:
        target_dir = "."
    os.makedirs(target_dir, exist_ok=True)

    output_filename = f"{video_basename}_reconstructed{video_ext}"
    output_filepath = os.path.join(target_dir, output_filename)

    # Requirement 2 & 3: Read matched sentences in script order (source of truth)
    raw_clips: List[Clip] = []
    matched_sentences_count = 0
    unmatched_sentences_count = 0

    for idx, item in enumerate(alignment_data, 1):
        if item.get("matched") is True:
            matched_sentences_count += 1
            start_fmt = item.get("start", "00:00:00.000")
            end_fmt = item.get("end", "00:00:00.000")
            s_sec = parse_timestamp_to_sec(start_fmt)
            e_sec = parse_timestamp_to_sec(end_fmt)
            dur = max(0.0, e_sec - s_sec)

            raw_clips.append(Clip(
                clip_id=len(raw_clips) + 1,
                start_time=start_fmt,
                end_time=end_fmt,
                duration=dur,
                sentence=item.get("sentence", ""),
                confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
                confidence_level=item.get("confidence_level"),
                start_sec=s_sec,
                end_sec=e_sec
            ))
        else:
            unmatched_sentences_count += 1

    extracted_regions_count = len(raw_clips)

    if extracted_regions_count == 0:
        print("Warning: No matched sentences found in alignment JSON. Reconstruction aborted.", file=sys.stderr)
        return ""

    # Requirement 4: Merge consecutive clip segments within gap threshold
    merged_clips = merge_consecutive_clips(raw_clips, merge_gap_seconds=merge_gap_seconds)
    merged_regions_count = len(merged_clips)

    # Requirement 1: Human Readable Terminal Logging for clips
    print("\n" + "=" * 60)
    print("BUMBLEBEE STAGE 3 — FFMPEG RECONSTRUCTION ENGINE")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp(prefix="bumblebee_clips_")
    segment_files: List[str] = []

    try:
        for clip in merged_clips:
            # Human-readable HH:MM:SS.mmm format in terminal logs
            print(
                f"\n[CLIP {clip.clip_id:03d}]\n\n"
                f"Sentence:\n{clip.sentence}\n\n"
                f"Start:\n{clip.start_time}\n\n"
                f"End:\n{clip.end_time}\n\n"
                f"Duration:\n{format_duration_human(clip.duration)}"
            )

            clip_filename = f"clip_{clip.clip_id:04d}{video_ext}"
            clip_path = os.path.join(temp_dir, clip_filename)

            # Requirement 5 & 8: FFmpeg extraction via argument list subprocess (-c copy)
            extract_cmd = [
                "ffmpeg", "-y",
                "-ss", clip.start_time,
                "-to", clip.end_time,
                "-i", video_path,
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                clip_path
            ]
            subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            segment_files.append(clip_path)

        # Requirement 6: Generate concat_list.txt for FFmpeg concat demuxer
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        # Requirement 6 & 8: Concatenate using FFmpeg demuxer with stream copy
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_filepath
        ]
        res = subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Requirement 8: Fallback to re-encode if stream copy fails on non-keyframe boundaries
        if res.returncode != 0:
            print("\n[FFMPEG CONCAT] Stream-copy failed due to non-keyframe segment boundaries. Falling back to fast re-encoding to guarantee output validity...", file=sys.stderr)
            concat_fallback_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                output_filepath
            ]
            subprocess.run(concat_fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Requirement 10: Final Statistics Summary Box
    input_dur = get_video_duration(video_path)
    output_dur = sum(c.duration for c in merged_clips)
    removed_dur = max(0.0, input_dur - output_dur)

    print("\n" + "=" * 50)
    print("\nINPUT DURATION:\n" + format_duration_human(input_dur))
    print("\nOUTPUT DURATION:\n" + format_duration_human(output_dur))
    print("\nREMOVED:\n" + format_duration_human(removed_dur))
    print(f"\nMATCHED SENTENCES:\n{matched_sentences_count}")
    print(f"\nUNMATCHED SENTENCES:\n{unmatched_sentences_count}")
    print(f"\nEXTRACTED REGIONS:\n{extracted_regions_count}")
    print(f"\nMERGED REGIONS:\n{merged_regions_count}\n")
    print("=" * 50 + "\n")
    print(f"[RECONSTRUCTION COMPLETE] Saved to: {output_filepath}\n")

    return output_filepath


def main():
    parser = argparse.ArgumentParser(description="Bumblebee Stage 3 Production-Grade FFmpeg Reconstruction Engine")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--alignment", required=True, help="Path to alignment JSON file")
    parser.add_argument("--merge-gap", type=float, default=MERGE_GAP_SECONDS, help="Gap threshold in seconds to merge consecutive clips (default: 0.5)")
    parser.add_argument("--output-dir", help="Optional output directory for reconstructed video")

    args = parser.parse_args()

    try:
        reconstruct_video(
            video_path=args.video,
            alignment_json_path=args.alignment,
            merge_gap_seconds=args.merge_gap,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"Error during video reconstruction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
