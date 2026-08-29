#!/usr/bin/env python3
"""
align_script.py - Bumblebee Stage 2 Hierarchical Script Alignment Engine

Aligns noisy word-level transcripts produced by transcript.py with a ground-truth script file.

Features:
  1. Structural Parsing & Merging (Heading, Subheading, Bullet, Numbered Item classification)
  2. Hierarchical Alignment Engine (Section -> Block -> Sentence Alignment)
  3. Monotonic Chronological Alignment (prevents backward timestamp jumps)
  4. 4-Part RapidFuzz Similarity Scoring (ratio, token_set_ratio, partial_ratio, coverage)
  5. Ordered Word Sequence Matching (LCS Order Bonus)
  6. ASR Error Normalization Layer (extensible dictionary mapping)
  7. Calibrated Confidence Levels (HIGH, MEDIUM, LOW)
  8. Rare-Word Anchored Inverted Index Search (scalable to 10,000+ words)
"""

import os
import re
import sys
import json
import string
import warnings
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import Counter

# Suppress FP16 CPU warnings
warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")

try:
    import rapidfuzz
except ImportError:
    print("Error: 'rapidfuzz' package is required. Install it using:\n  pip install rapidfuzz", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# Common ASR error replacements for domain speech
COMMON_ASR_NORMALIZATIONS: Dict[str, str] = {
    "road map": "roadmap",
    "load map": "roadmap",
    "sangal": "sanger",
    "nuclear types": "nucleotides",
    "gene mapping": "genemapping",
}


class ScriptElementType(Enum):
    HEADING = auto()
    SUBHEADING = auto()
    BULLET = auto()
    NUMBERED_ITEM = auto()
    SENTENCE = auto()
    STRUCTURAL = auto()


@dataclass
class TranscriptWord:
    raw_word: str
    clean_word: str
    start_sec: float
    end_sec: float
    start_fmt: str
    end_fmt: str
    index: int


@dataclass
class ScriptSentence:
    sentence_id: int
    raw_text: str
    clean_text: str
    words: List[str]
    word_count: int
    element_type: ScriptElementType = ScriptElementType.SENTENCE


@dataclass
class ScriptBlock:
    block_id: int
    raw_text: str
    clean_text: str
    sentences: List[ScriptSentence]
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None


@dataclass
class ScriptSection:
    section_id: int
    heading_sentence: Optional[ScriptSentence]
    heading_text: str
    blocks: List[ScriptBlock]
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None


def parse_ffmpeg_timestamp(timestamp_str: str) -> float:
    """Parses HH:MM:SS.mmm timestamp string into float seconds."""
    try:
        parts = timestamp_str.split(":")
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds
    except Exception:
        return 0.0


def clean_text(text: str, normalizations: Optional[Dict[str, str]] = None) -> str:
    """
    Normalizes text by converting to lowercase, removing punctuation,
    collapsing spaces, and applying ASR error normalizations.
    """
    translator = str.maketrans("", "", string.punctuation)
    text_clean = text.translate(translator).lower()
    text_clean = " ".join(text_clean.split())

    norm_dict = COMMON_ASR_NORMALIZATIONS if normalizations is None else normalizations
    for err, fix in norm_dict.items():
        if err in text_clean:
            text_clean = text_clean.replace(err, fix)

    return text_clean


def detect_element_type(line: str) -> ScriptElementType:
    """
    Classifies a script line into a ScriptElementType enum.
    """
    trimmed = line.strip()
    if not trimmed:
        return ScriptElementType.STRUCTURAL

    # Heading / Subheading detection
    if re.match(r'^(what is|types of|applications of|steps in|introduction|conclusion|overview)', trimmed, re.I):
        return ScriptElementType.HEADING
    if re.match(r'^(advantages|disadvantages|summary|overview|notes):?$', trimmed, re.I):
        return ScriptElementType.SUBHEADING

    # Numbered item (e.g., "1.", "2)", "(3)")
    if re.match(r'^(\d+[\.\)]|\(\d+\))\s*$', trimmed) or re.match(r'^(\d+[\.\)]|\(\d+\))\s+\w+', trimmed):
        return ScriptElementType.NUMBERED_ITEM

    # Bullet point (e.g., "- High Accuracy", "* Item")
    if re.match(r'^[\-\*\•]\s+', trimmed):
        return ScriptElementType.BULLET

    # Short standalone label
    words = trimmed.split()
    if len(words) < 3 and trimmed.endswith(':'):
        return ScriptElementType.SUBHEADING
    if len(words) < 2:
        return ScriptElementType.STRUCTURAL

    return ScriptElementType.SENTENCE


def split_sentences_safely(text: str) -> List[str]:
    """
    Splits text on sentence boundaries without separating list prefixes (e.g. '1.').
    """
    chunks = re.split(r'(?<=[.!?])\s+', text)
    result = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        clean_chk = chunk.strip()
        if i < len(chunks) - 1 and (re.match(r'^(\d+[\.\)]|\(\d+\))$', clean_chk) or len(clean_chk) <= 2):
            result.append(f"{chunk} {chunks[i+1]}")
            i += 2
        else:
            result.append(chunk)
            i += 1
    return result


def normalize_transcript(
    transcript_tuples: list,
    normalizations: Optional[Dict[str, str]] = None
) -> Tuple[List[TranscriptWord], Dict[str, int], Dict[str, List[int]]]:
    """
    STAGE 1: Normalizes transcript words and builds word_frequency & inverted_index.
    """
    words_list: List[TranscriptWord] = []
    word_frequency: Dict[str, int] = {}
    inverted_index: Dict[str, List[int]] = {}

    for idx, item in enumerate(transcript_tuples):
        raw_word, start_fmt, end_fmt = item[0], item[1], item[2]
        clean_w = clean_text(raw_word, normalizations)

        start_sec = parse_ffmpeg_timestamp(start_fmt)
        end_sec = parse_ffmpeg_timestamp(end_fmt)

        t_word = TranscriptWord(
            raw_word=raw_word,
            clean_word=clean_w,
            start_sec=start_sec,
            end_sec=end_sec,
            start_fmt=start_fmt,
            end_fmt=end_fmt,
            index=idx
        )
        words_list.append(t_word)

        if clean_w:
            word_frequency[clean_w] = word_frequency.get(clean_w, 0) + 1
            if clean_w not in inverted_index:
                inverted_index[clean_w] = []
            inverted_index[clean_w].append(idx)

    return words_list, word_frequency, inverted_index


def parse_script_hierarchical(
    script_content: str,
    normalizations: Optional[Dict[str, str]] = None
) -> List[ScriptSection]:
    """
    STAGES 1 & 2 (Structural Parsing & Merging):
    Splits script into sentences, detects element types, merges short structural
    elements (< 3 words), and builds a 3-tier hierarchy: Section -> Block -> Sentence.
    """
    raw_lines = re.split(r'\n+', script_content)
    parsed_sentences: List[ScriptSentence] = []
    sent_id = 0

    pending_prefix = ""

    for line in raw_lines:
        trimmed = line.strip()
        if not trimmed:
            continue

        elem_type = detect_element_type(trimmed)
        words = trimmed.split()

        # Structural Merging: If short element (< 3 words), merge with next content
        if len(words) < 3 and elem_type in (ScriptElementType.NUMBERED_ITEM, ScriptElementType.SUBHEADING, ScriptElementType.STRUCTURAL):
            pending_prefix = (pending_prefix + " " + trimmed).strip()
            continue

        full_raw = (pending_prefix + " " + trimmed).strip() if pending_prefix else trimmed
        pending_prefix = ""

        # Sub-split long lines into sentences safely
        sub_sentences = split_sentences_safely(full_raw)
        for s in sub_sentences:
            s_trimmed = s.strip()
            if not s_trimmed:
                continue
            cleaned = clean_text(s_trimmed, normalizations)
            c_words = cleaned.split()
            if not c_words:
                continue

            sentence_obj = ScriptSentence(
                sentence_id=sent_id,
                raw_text=s_trimmed,
                clean_text=cleaned,
                words=c_words,
                word_count=len(c_words),
                element_type=detect_element_type(s_trimmed)
            )
            parsed_sentences.append(sentence_obj)
            sent_id += 1

    if pending_prefix:
        cleaned = clean_text(pending_prefix, normalizations)
        c_words = cleaned.split()
        if c_words:
            parsed_sentences.append(ScriptSentence(
                sentence_id=sent_id,
                raw_text=pending_prefix,
                clean_text=cleaned,
                words=c_words,
                word_count=len(c_words),
                element_type=ScriptElementType.STRUCTURAL
            ))

    # Group sentences into Blocks and Sections
    sections: List[ScriptSection] = []
    curr_section_heading: Optional[ScriptSentence] = None
    curr_section_blocks: List[ScriptBlock] = []
    curr_block_sentences: List[ScriptSentence] = []
    block_id = 0
    section_id = 0

    def finalize_block():
        nonlocal block_id, curr_block_sentences, curr_section_blocks
        if not curr_block_sentences:
            return
        b_raw = " ".join([s.raw_text for s in curr_block_sentences])
        b_clean = " ".join([s.clean_text for s in curr_block_sentences])
        curr_section_blocks.append(ScriptBlock(
            block_id=block_id,
            raw_text=b_raw,
            clean_text=b_clean,
            sentences=list(curr_block_sentences)
        ))
        block_id += 1
        curr_block_sentences = []

    def finalize_section():
        nonlocal section_id, curr_section_heading, curr_section_blocks, sections
        finalize_block()
        if not curr_section_blocks and not curr_section_heading:
            return
        h_text = curr_section_heading.raw_text if curr_section_heading else f"Section {section_id + 1}"
        sections.append(ScriptSection(
            section_id=section_id,
            heading_sentence=curr_section_heading,
            heading_text=h_text,
            blocks=list(curr_section_blocks)
        ))
        section_id += 1
        curr_section_heading = None
        curr_section_blocks = []

    for sentence in parsed_sentences:
        if sentence.element_type == ScriptElementType.HEADING:
            finalize_section()
            curr_section_heading = sentence
        else:
            curr_block_sentences.append(sentence)
            if len(curr_block_sentences) >= 4:
                finalize_block()

    finalize_section()
    return sections


def compute_lcs_order_ratio(script_words: List[str], cand_words: List[str]) -> float:
    """
    Computes Longest Common Subsequence (LCS) ratio to reward ordered word matches.
    """
    if not script_words or not cand_words:
        return 0.0
    m, n = len(script_words), len(cand_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if script_words[i - 1] == cand_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    return (lcs_len / float(m)) * 100.0


def score_candidate_advanced(
    sentence: ScriptSentence,
    candidate_words: List[TranscriptWord],
    max_pause_seconds: float = 1.5,
    pause_weight: float = 5.0
) -> Tuple[float, float, float, float]:
    """
    STAGE 4: Improved 4-Part RapidFuzz Similarity & Multiset Coverage Scoring.

    Similarity = 0.35 * fuzz.ratio
               + 0.35 * fuzz.token_set_ratio
               + 0.20 * fuzz.partial_ratio
               + 0.10 * coverage

    Final Ranking Score = 0.6 * similarity + 0.4 * coverage + order_bonus - pause_penalty
    """
    cand_clean_words = [w.clean_word for w in candidate_words if w.clean_word]
    cand_clean_text = " ".join(cand_clean_words)

    # 1. Multi-metric RapidFuzz Scores
    ratio_score = float(rapidfuzz.fuzz.ratio(sentence.clean_text, cand_clean_text))
    token_set_score = float(rapidfuzz.fuzz.token_set_ratio(sentence.clean_text, cand_clean_text))
    partial_score = float(rapidfuzz.fuzz.partial_ratio(sentence.clean_text, cand_clean_text))

    # 2. Multiset Word Coverage
    script_counts = Counter(sentence.words)
    cand_counts = Counter(cand_clean_words)
    matched_words_count = sum(min(count, cand_counts.get(word, 0)) for word, count in script_counts.items())
    coverage_score = (matched_words_count / sentence.word_count) * 100.0 if sentence.word_count > 0 else 0.0

    # 3. Composite Similarity (0-100)
    similarity = (
        0.35 * ratio_score +
        0.35 * token_set_score +
        0.20 * partial_score +
        0.10 * coverage_score
    )

    # 4. Ordered Match Sequence Bonus (LCS)
    order_ratio = compute_lcs_order_ratio(sentence.words, cand_clean_words)
    order_bonus = 0.05 * order_ratio

    # 5. Continuous Pause Penalty
    pause_penalty = 0.0
    for i in range(len(candidate_words) - 1):
        curr_word = candidate_words[i]
        next_word = candidate_words[i + 1]
        gap = next_word.start_sec - curr_word.end_sec
        if gap > max_pause_seconds:
            pause_penalty += (gap - max_pause_seconds) * pause_weight

    # Ranking Score
    ranking_score = (0.6 * similarity) + (0.4 * coverage_score) + order_bonus - pause_penalty

    return similarity, coverage_score, pause_penalty, ranking_score


def generate_candidate_windows_bounded(
    sentence: ScriptSentence,
    transcript_words: List[TranscriptWord],
    word_frequency: Dict[str, int],
    inverted_index: Dict[str, List[int]],
    min_search_idx: int = 0,
    max_search_idx: Optional[int] = None,
    max_anchors: int = 3
) -> List[Tuple[int, int]]:
    """
    STAGE 3: Rare-Word Anchored Candidate Generation bounded by search region.
    """
    total_words = len(transcript_words)
    if total_words == 0:
        return []

    if max_search_idx is None or max_search_idx >= total_words:
        max_search_idx = total_words - 1

    min_search_idx = max(0, min_search_idx)
    if min_search_idx > max_search_idx:
        min_search_idx = max(0, max_search_idx - 20)

    # Valid words in sentence that appear in transcript
    valid_words = [w for w in sentence.words if w in word_frequency]

    if not valid_words:
        window_size = sentence.word_count
        res = []
        for i in range(min_search_idx, max_search_idx + 1):
            e_idx = min(max_search_idx, i + window_size - 1)
            res.append((i, e_idx))
        return res

    # Select rarest words
    sorted_words = sorted(valid_words, key=lambda w: word_frequency[w])
    rarest_words = sorted_words[:max_anchors]

    anchor_indices = set()
    for w in rarest_words:
        for idx in inverted_index.get(w, []):
            if min_search_idx <= idx <= max_search_idx:
                anchor_indices.add(idx)

    if not anchor_indices:
        # Fallback to bounded range scan
        window_size = sentence.word_count
        return [(i, min(max_search_idx, i + window_size - 1)) for i in range(min_search_idx, max_search_idx + 1)]

    candidate_windows = set()
    L = sentence.word_count
    padding = max(3, int(0.5 * L) + 2)

    for anchor in anchor_indices:
        min_s = max(min_search_idx, anchor - L - padding)
        max_s = min(max_search_idx, anchor + padding)

        for s_idx in range(min_s, max_s + 1):
            for length_delta in range(-padding, padding + 1):
                e_idx = s_idx + L + length_delta - 1
                if min_search_idx <= s_idx <= e_idx <= max_search_idx:
                    candidate_windows.add((s_idx, e_idx))

    return list(candidate_windows)


def calibrate_confidence_level(confidence: float) -> str:
    """
    STAGE 5: Confidence Calibration Level.
    """
    if confidence >= 90.0:
        return "HIGH"
    elif confidence >= 75.0:
        return "MEDIUM"
    else:
        return "LOW"


def align_script(
    transcript_tuples: list,
    script_path: str,
    max_pause_seconds: float = 1.5,
    pause_weight: float = 5.0,
    tie_threshold: float = 2.0,
    min_confidence: float = 70.0,
    monotonic_overlap: int = 20,
    normalizations: Optional[Dict[str, str]] = None
) -> List[dict]:
    """
    Main Hierarchical Alignment Function.

    Args:
        transcript_tuples: List of (word, start_fmt, end_fmt) tuples in memory.
        script_path: Ground-truth script file path or raw script text.
        max_pause_seconds: Max gap allowed before pause penalty.
        pause_weight: Penalty multiplier for long pauses.
        tie_threshold: Threshold to prefer later occurrence for retakes.
        min_confidence: Minimum score required for a match.
        monotonic_overlap: Allowed backwards search overlap window (default 20).
        normalizations: Custom ASR error replacement dictionary.

    Returns:
        List of dict alignment results for each sentence in the script.
    """
    script_text = script_path
    if isinstance(script_path, str) and os.path.exists(script_path):
        with open(script_path, 'r', encoding='utf-8') as f:
            script_text = f.read()

    # Stage 1: Normalize Transcript & Build Inverted Index
    transcript_words, word_frequency, inverted_index = normalize_transcript(transcript_tuples, normalizations)
    total_transcript_words = len(transcript_words)

    # Stage 2: Parse Script into Sections -> Blocks -> Sentences
    sections = parse_script_hierarchical(script_text, normalizations)

    # Collect all sentences across sections for monotonic alignment tracking
    all_sentences: List[ScriptSentence] = []
    for sec in sections:
        if sec.heading_sentence:
            all_sentences.append(sec.heading_sentence)
        for b in sec.blocks:
            for s in b.sentences:
                all_sentences.append(s)

    alignment_results = []
    last_matched_end_idx = 0

    iterator = tqdm(all_sentences, desc="Aligning script sentences", unit="sentence", file=sys.stderr) if tqdm else all_sentences

    for sentence in iterator:
        # Monotonic Alignment Constraint: Start search at (last_matched_end_idx - monotonic_overlap)
        min_search_idx = max(0, last_matched_end_idx - monotonic_overlap)

        # Stage 3: Candidate Window Generation
        candidate_windows = generate_candidate_windows_bounded(
            sentence, transcript_words, word_frequency, inverted_index,
            min_search_idx=min_search_idx, max_search_idx=total_transcript_words - 1
        )

        best_candidate = None
        best_ranking_score = float('-inf')

        # Stage 4: Multi-metric Scoring
        for s_idx, e_idx in candidate_windows:
            cand_slice = transcript_words[s_idx : e_idx + 1]
            sim, cov, penalty, rank_score = score_candidate_advanced(
                sentence, cand_slice, max_pause_seconds, pause_weight
            )

            if best_candidate is None:
                best_ranking_score = rank_score
                best_candidate = (s_idx, e_idx, sim, cov, penalty, rank_score)
            else:
                score_diff = rank_score - best_ranking_score
                if score_diff > tie_threshold:
                    best_ranking_score = rank_score
                    best_candidate = (s_idx, e_idx, sim, cov, penalty, rank_score)
                elif abs(score_diff) <= tie_threshold:
                    # Retake Policy: Prefer later occurrence (higher start_idx)
                    if s_idx > best_candidate[0]:
                        best_ranking_score = rank_score
                        best_candidate = (s_idx, e_idx, sim, cov, penalty, rank_score)

        if best_candidate is None:
            alignment_results.append({
                "sentence": sentence.raw_text,
                "matched": False
            })
            continue

        s_idx, e_idx, sim, cov, penalty, rank_score = best_candidate
        confidence = max(0.0, min(100.0, rank_score))

        if confidence < min_confidence:
            alignment_results.append({
                "sentence": sentence.raw_text,
                "matched": False
            })
        else:
            cand_slice = transcript_words[s_idx : e_idx + 1]
            matched_text = " ".join([w.raw_word for w in cand_slice])
            start_fmt = cand_slice[0].start_fmt
            end_fmt = cand_slice[-1].end_fmt

            # Update last matched end index for monotonic tracking
            last_matched_end_idx = max(last_matched_end_idx, e_idx)

            alignment_results.append({
                "sentence": sentence.raw_text,
                "start": start_fmt,
                "end": end_fmt,
                "confidence": round(confidence, 2),
                "confidence_level": calibrate_confidence_level(confidence),
                "matched_text": matched_text,
                "start_idx": s_idx,
                "end_idx": e_idx
            })

    return alignment_results


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 align_script.py <script_file_path> <video_or_transcript_file>", file=sys.stderr)
        print("Example: python3 align_script.py script.txt lecture.mp4", file=sys.stderr)
        sys.exit(1)

    script_file = sys.argv[1]
    media_or_json_file = sys.argv[2]

    try:
        if media_or_json_file.endswith('.json'):
            with open(media_or_json_file, 'r', encoding='utf-8') as f:
                transcript_tuples = json.load(f)
        else:
            from transcript import generate_transcript
            transcript_tuples = generate_transcript(media_or_json_file)

        results = align_script(transcript_tuples, script_file)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error during alignment: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
