#!/usr/bin/env python3
"""
align_script.py - Bumblebee Stage 2 Alignment Engine with Persistent Tee-Style Logger

Aligns noisy word-level transcripts produced by transcript.py with a ground-truth script file.

Features:
  1. Persistent Run Logging (automatically creates alignment_run_YYYYMMDD_HHMMSS.txt)
  2. Tee-Style Stream Redirector (simultaneous terminal & log file output)
  3. Section Heading Hard Anchors (prevents section drift across large transcripts)
  4. Block Region Expansion (+/- 150 words search region expansion for sentences)
  5. Neighbor-Based Rescue Pass (+/- 20 words local search between prev/next matched bounds)
  6. Debug Instrumentation Logging (Block Alignment, Expanded Regions, Rejections, Rescue Pass)
  7. Monotonic Chronological Alignment (prevents backward timestamp jumps)
  8. 4-Part RapidFuzz Similarity Scoring (ratio, token_set_ratio, partial_ratio, coverage)
  9. ASR Error Normalization Layer (extensible dictionary mapping)
 10. Alignment Failure Analytics Summary
 11. Calibrated Confidence Levels (HIGH, MEDIUM, LOW)
"""

import os
import re
import sys
import json
import time
import string
import argparse
import datetime
import warnings
from enum import Enum, auto
from dataclasses import dataclass
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


# Configuration Constants
DEBUG_ALIGNMENT = False
BASE_SEARCH_WINDOW = 150
MAX_SEARCH_WINDOW = 600
BLOCK_EXPANSION_MARGIN = 150
NEIGHBOR_RESCUE_MARGIN = 20

# Common ASR error replacements for domain speech
COMMON_ASR_NORMALIZATIONS: Dict[str, str] = {
    "road map": "roadmap",
    "load map": "roadmap",
    "sangal": "sanger",
    "nuclear types": "nucleotides",
    "gene mapping": "genemapping",
}


# =========================================================================
# PERSISTENT TEE-STYLE RUN LOGGER (Python Standard Library Only)
# =========================================================================

class TeeStream:
    """Tee adapter that writes text to both a terminal stream and a log file simultaneously."""
    def __init__(self, original_stream, file_obj):
        self.original_stream = original_stream
        self.file_obj = file_obj

    def write(self, data):
        self.original_stream.write(data)
        self.file_obj.write(data)
        self.file_obj.flush()

    def flush(self):
        self.original_stream.flush()
        self.file_obj.flush()


class RunLoggerContext:
    """Context manager for automatic Tee stream redirection and cleanup."""
    def __init__(self, log_file, filepath, filename, start_time_sec):
        self.log_file = log_file
        self.filepath = filepath
        self.filename = filename
        self.start_time_sec = start_time_sec
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr

    def __enter__(self):
        sys.stdout = TeeStream(self.orig_stdout, self.log_file)
        sys.stderr = TeeStream(self.orig_stderr, self.log_file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time_sec
        if exc_type is not None:
            self.log_file.write(f"\n[EXCEPTION IN RUN]\n{exc_type.__name__}: {exc_val}\n")
        
        self.log_file.write(f"\n[RUNTIME SUMMARY]\nTotal Execution Time: {duration:.2f} seconds\n")
        self.log_file.flush()
        
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr
        self.log_file.close()


def create_run_logger(log_dir: str = ".") -> RunLoggerContext:
    """
    Creates a timestamped text log file (alignment_run_YYYYMMDD_HHMMSS.txt)
    and returns a context manager that redirects stdout and stderr simultaneously.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"alignment_run_{timestamp}.txt"
    filepath = os.path.join(log_dir, filename)

    print(f"[LOG FILE]\n{filename}", file=sys.stderr)

    log_file = open(filepath, "w", encoding="utf-8")

    start_str = now.strftime("%Y-%m-%d %H:%M:%S")
    log_file.write("=" * 60 + "\n")
    log_file.write(f"BUMBLEBEE ALIGNMENT RUN LOG\n")
    log_file.write(f"Start Time: {start_str}\n")
    log_file.write(f"Log File:   {filename}\n")
    log_file.write("=" * 60 + "\n\n")
    log_file.flush()

    return RunLoggerContext(log_file, filepath, filename, time.time())


class ScriptElementType(Enum):
    HEADING = auto()
    SUBHEADING = auto()
    BULLET = auto()
    NUMBERED_ITEM = auto()
    SENTENCE = auto()
    STRUCTURAL = auto()


class RejectionReason(Enum):
    NONE = "NONE"
    NO_CANDIDATES = "NO_CANDIDATES"
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    MONOTONIC_REJECTED = "MONOTONIC_REJECTED"
    OUTSIDE_WINDOW = "OUTSIDE_WINDOW"
    BLOCK_ALIGNMENT_FAILED = "BLOCK_ALIGNMENT_FAILED"


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
    block_id: Optional[int] = None
    section_name: Optional[str] = None


@dataclass
class ScriptBlock:
    block_id: int
    raw_text: str
    clean_text: str
    sentences: List[ScriptSentence]
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    expanded_start_idx: Optional[int] = None
    expanded_end_idx: Optional[int] = None


@dataclass
class ScriptSection:
    section_id: int
    heading_sentence: Optional[ScriptSentence]
    heading_text: str
    blocks: List[ScriptBlock]
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    heading_anchor_idx: Optional[int] = None


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
    """Classifies a script line into a ScriptElementType enum."""
    trimmed = line.strip()
    if not trimmed:
        return ScriptElementType.STRUCTURAL

    if re.match(r'^(what is|types of|applications of|steps in|introduction|conclusion|overview)', trimmed, re.I):
        return ScriptElementType.HEADING
    if re.match(r'^(advantages|disadvantages|limitations|applications|summary|overview|notes):?$', trimmed, re.I):
        return ScriptElementType.SUBHEADING

    if re.match(r'^(\d+[\.\)]|\(\d+\))\s*$', trimmed) or re.match(r'^(\d+[\.\)]|\(\d+\))\s+\w+', trimmed):
        return ScriptElementType.NUMBERED_ITEM

    if re.match(r'^[\-\*\•]\s+', trimmed):
        return ScriptElementType.BULLET

    words = trimmed.split()
    if len(words) < 3 and trimmed.endswith(':'):
        return ScriptElementType.SUBHEADING
    if len(words) < 2:
        return ScriptElementType.STRUCTURAL

    return ScriptElementType.SENTENCE


def split_sentences_safely(text: str) -> List[str]:
    """Splits text on sentence boundaries without separating list prefixes (e.g. '1.')."""
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
    """STAGE 1: Normalizes transcript words and builds word_frequency & inverted_index."""
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
    elements (< 3 words), and constructs ScriptBlock and ScriptSection objects.
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
        block_obj = ScriptBlock(
            block_id=block_id,
            raw_text=b_raw,
            clean_text=b_clean,
            sentences=list(curr_block_sentences)
        )
        for s in curr_block_sentences:
            s.block_id = block_id
            if curr_section_heading:
                s.section_name = curr_section_heading.raw_text
            else:
                s.section_name = f"Section {section_id + 1}"
        curr_section_blocks.append(block_obj)
        block_id += 1
        curr_block_sentences = []

    def finalize_section():
        nonlocal section_id, curr_section_heading, curr_section_blocks, sections
        finalize_block()
        if not curr_section_blocks and not curr_section_heading:
            return
        h_text = curr_section_heading.raw_text if curr_section_heading else f"Section {section_id + 1}"
        if curr_section_heading:
            curr_section_heading.section_name = h_text
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
    """Computes Longest Common Subsequence (LCS) ratio to reward ordered word matches."""
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

    ratio_score = float(rapidfuzz.fuzz.ratio(sentence.clean_text, cand_clean_text))
    token_set_score = float(rapidfuzz.fuzz.token_set_ratio(sentence.clean_text, cand_clean_text))
    partial_score = float(rapidfuzz.fuzz.partial_ratio(sentence.clean_text, cand_clean_text))

    script_counts = Counter(sentence.words)
    cand_counts = Counter(cand_clean_words)
    matched_words_count = sum(min(count, cand_counts.get(word, 0)) for word, count in script_counts.items())
    coverage_score = (matched_words_count / sentence.word_count) * 100.0 if sentence.word_count > 0 else 0.0

    similarity = (
        0.35 * ratio_score +
        0.35 * token_set_score +
        0.20 * partial_score +
        0.10 * coverage_score
    )

    order_ratio = compute_lcs_order_ratio(sentence.words, cand_clean_words)
    order_bonus = 0.05 * order_ratio

    pause_penalty = 0.0
    for i in range(len(candidate_words) - 1):
        curr_word = candidate_words[i]
        next_word = candidate_words[i + 1]
        gap = next_word.start_sec - curr_word.end_sec
        if gap > max_pause_seconds:
            pause_penalty += (gap - max_pause_seconds) * pause_weight

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
    """STAGE 3: Rare-Word Anchored Candidate Generation bounded by search region."""
    total_words = len(transcript_words)
    if total_words == 0:
        return []

    if max_search_idx is None or max_search_idx >= total_words:
        max_search_idx = total_words - 1

    min_search_idx = max(0, min_search_idx)
    if min_search_idx > max_search_idx:
        min_search_idx = max(0, max_search_idx - 20)

    valid_words = [w for w in sentence.words if w in word_frequency]

    if not valid_words:
        window_size = sentence.word_count
        res = []
        for i in range(min_search_idx, max_search_idx + 1):
            e_idx = min(max_search_idx, i + window_size - 1)
            res.append((i, e_idx))
        return res

    sorted_words = sorted(valid_words, key=lambda w: word_frequency[w])
    rarest_words = sorted_words[:max_anchors]

    anchor_indices = set()
    for w in rarest_words:
        for idx in inverted_index.get(w, []):
            if min_search_idx <= idx <= max_search_idx:
                anchor_indices.add(idx)

    if not anchor_indices:
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


def compute_adaptive_search_window(
    prev_unmatched: bool,
    block_conf_low: bool,
    section_changed: bool
) -> int:
    """Dynamically expands search window from BASE_SEARCH_WINDOW (150) up to MAX_SEARCH_WINDOW (600)."""
    window = BASE_SEARCH_WINDOW
    if prev_unmatched:
        window += 150
    if block_conf_low:
        window += 150
    if section_changed:
        window += 150
    return min(MAX_SEARCH_WINDOW, window)


def calibrate_confidence_level(confidence: float) -> str:
    """STAGE 5: Confidence Calibration Level."""
    if confidence >= 90.0:
        return "HIGH"
    elif confidence >= 75.0:
        return "MEDIUM"
    else:
        return "LOW"


def evaluate_candidate_windows(
    sentence: ScriptSentence,
    candidate_windows: List[Tuple[int, int]],
    transcript_words: List[TranscriptWord],
    max_pause_seconds: float = 1.5,
    pause_weight: float = 5.0,
    tie_threshold: float = 2.0
) -> Tuple[Optional[Tuple], float]:
    """
    Evaluates candidate windows using 4-part scoring & retake preference policy.
    Returns: (best_candidate_tuple, confidence_score)
    """
    if not candidate_windows:
        return None, 0.0

    evaluated_candidates = []
    for s_idx, e_idx in candidate_windows:
        cand_slice = transcript_words[s_idx : e_idx + 1]
        sim, cov, penalty, rank_score = score_candidate_advanced(
            sentence, cand_slice, max_pause_seconds, pause_weight
        )
        evaluated_candidates.append((s_idx, e_idx, sim, cov, penalty, rank_score, cand_slice))

    evaluated_candidates.sort(key=lambda c: c[5], reverse=True)
    best_candidate = evaluated_candidates[0]
    best_ranking_score = best_candidate[5]

    for cand in evaluated_candidates[1:10]:
        s_idx, rank_score = cand[0], cand[5]
        score_diff = rank_score - best_ranking_score
        if score_diff > tie_threshold:
            best_ranking_score = rank_score
            best_candidate = cand
        elif abs(score_diff) <= tie_threshold:
            if s_idx > best_candidate[0]:
                best_ranking_score = rank_score
                best_candidate = cand

    normal_confidence = max(0.0, min(100.0, best_candidate[5]))
    return best_candidate, normal_confidence


def align_script(
    transcript_tuples: list,
    script_path: str,
    max_pause_seconds: float = 1.5,
    pause_weight: float = 5.0,
    tie_threshold: float = 2.0,
    min_confidence: float = 70.0,
    monotonic_overlap: int = 20,
    normalizations: Optional[Dict[str, str]] = None,
    debug_alignment: bool = DEBUG_ALIGNMENT
) -> List[dict]:
    """
    Main Hierarchical Alignment Function with Section Heading Hard Anchors,
    Block Region Expansion, Neighbor-Based Rescue Pass, and Persistent Diagnostics.
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

    # =========================================================================
    # IMPROVEMENT 1: SECTION HEADING HARD ANCHORS
    # What is a section anchor?
    # A section anchor is the verified transcript word index (section_anchor_idx)
    # where the section heading title was spoken.
    #
    # Why does it reduce drift?
    # In large lecture transcripts (10,000+ words), sentences far down in the
    # script can drift backward if early candidates match common generic words.
    # Binding all blocks in a section to start at >= max(0, section_anchor_idx - 50)
    # acts as a hard checkpoint that anchors the entire section's candidate search.
    # =========================================================================
    for sec in sections:
        sec_words = []
        if sec.heading_sentence:
            sec_words.extend(sec.heading_sentence.words)
            h_anchors = [idx for w in sec.heading_sentence.words if w in inverted_index for idx in inverted_index[w]]
            if h_anchors:
                sec.heading_anchor_idx = min(h_anchors)

        for b in sec.blocks:
            sec_words.extend(b.clean_text.split())

        sec_anchors = [idx for w in set(sec_words) if w in inverted_index for idx in inverted_index[w]]
        if sec_anchors:
            sec.start_idx = min(sec_anchors)
            sec.end_idx = max(sec_anchors)
        else:
            sec.start_idx = sec.heading_anchor_idx if sec.heading_anchor_idx is not None else 0
            sec.end_idx = max(0, total_transcript_words - 1)

        # Enforce hard section anchor bound
        if sec.heading_anchor_idx is not None:
            sec.start_idx = max(sec.start_idx, max(0, sec.heading_anchor_idx - 50))

        # =========================================================================
        # IMPROVEMENT 2: BLOCK REGION EXPANSION (+/- 150 words)
        # Why is expansion needed?
        # Coarse block alignment might find a tight word range [start_idx, end_idx],
        # but surrounding sentences in the paragraph may spill over slightly before
        # or after.
        #
        # Why +/- 150 buffer?
        # A buffer of +/- 150 words covers ~30-45 seconds of speech padding,
        # ensuring all sentences in the block remain discoverable without triggering
        # expensive global transcript scans.
        # =========================================================================
        for b in sec.blocks:
            b_words = b.clean_text.split()
            b_anchors = [idx for w in set(b_words) if w in inverted_index for idx in inverted_index[w]]
            if b_anchors:
                b.start_idx = min(b_anchors)
                b.end_idx = max(b_anchors)
            else:
                b.start_idx = sec.start_idx
                b.end_idx = sec.end_idx

            if sec.heading_anchor_idx is not None:
                b.start_idx = max(b.start_idx, max(0, sec.heading_anchor_idx - 50))

            # Expand block search region by +/- 150 words
            b.expanded_start_idx = max(0, b.start_idx - BLOCK_EXPANSION_MARGIN)
            b.expanded_end_idx = min(total_transcript_words - 1, b.end_idx + BLOCK_EXPANSION_MARGIN)

            if debug_alignment:
                print(
                    f"\n[BLOCK ALIGNMENT]\n"
                    f"Block {b.block_id}\n"
                    f"Region: [{b.start_idx}-{b.end_idx}]\n"
                    f"Expanded: [{b.expanded_start_idx}-{b.expanded_end_idx}]",
                    file=sys.stderr
                )

    # Collect all sentences across sections for alignment loop
    all_sentences: List[ScriptSentence] = []
    sentence_block_map: Dict[int, ScriptBlock] = {}
    for sec in sections:
        if sec.heading_sentence:
            all_sentences.append(sec.heading_sentence)
        for b in sec.blocks:
            for s in b.sentences:
                all_sentences.append(s)
                sentence_block_map[s.sentence_id] = b

    alignment_results = [None] * len(all_sentences)
    last_matched_end_idx = 0
    prev_unmatched = False
    last_section_name = None

    failure_analytics = {
        RejectionReason.NO_CANDIDATES.value: 0,
        RejectionReason.BELOW_THRESHOLD.value: 0,
        RejectionReason.MONOTONIC_REJECTED.value: 0,
        RejectionReason.OUTSIDE_WINDOW.value: 0,
        RejectionReason.BLOCK_ALIGNMENT_FAILED.value: 0,
    }

    iterator = enumerate(all_sentences)
    if tqdm:
        iterator = tqdm(list(iterator), desc="Aligning script sentences", unit="sentence", file=sys.stderr)

    for idx, sentence in iterator:
        section_changed = (sentence.section_name != last_section_name)
        last_section_name = sentence.section_name

        parent_block = sentence_block_map.get(sentence.sentence_id)
        block_max_idx = parent_block.expanded_end_idx if parent_block and parent_block.expanded_end_idx is not None else total_transcript_words - 1

        adaptive_window_size = compute_adaptive_search_window(
            prev_unmatched=prev_unmatched,
            block_conf_low=prev_unmatched,
            section_changed=section_changed
        )

        min_search_idx = max(0, last_matched_end_idx - monotonic_overlap)
        max_search_idx = min(total_transcript_words - 1, max(block_max_idx, min_search_idx + adaptive_window_size))

        # Pass 1: Candidate Generation inside Expanded Block / Adaptive Window
        candidate_windows = generate_candidate_windows_bounded(
            sentence, transcript_words, word_frequency, inverted_index,
            min_search_idx=min_search_idx, max_search_idx=max_search_idx
        )

        # Fallback to expanded transcript search if empty
        if not candidate_windows and total_transcript_words > 0:
            candidate_windows = generate_candidate_windows_bounded(
                sentence, transcript_words, word_frequency, inverted_index,
                min_search_idx=min_search_idx, max_search_idx=total_transcript_words - 1
            )

        if not candidate_windows:
            rejection = RejectionReason.NO_CANDIDATES
            failure_analytics[rejection.value] += 1
            prev_unmatched = True

            if debug_alignment:
                print(
                    f"\n[ALIGNMENT REJECTED]\n"
                    f"Sentence: \"{sentence.raw_text}\"\n"
                    f"Section: \"{sentence.section_name}\"\n"
                    f"Block ID: {sentence.block_id}\n"
                    f"Search Region: [{min_search_idx},{max_search_idx}]\n"
                    f"Candidate Count: 0\n"
                    f"Best Candidate: N/A\n"
                    f"Best Score: 0.0",
                    file=sys.stderr
                )

            alignment_results[idx] = {
                "sentence": sentence.raw_text,
                "matched": False,
                "rejection_reason": rejection.value
            }
            continue

        best_cand, conf = evaluate_candidate_windows(
            sentence, candidate_windows, transcript_words,
            max_pause_seconds, pause_weight, tie_threshold
        )

        if best_cand and conf >= min_confidence:
            s_idx, e_idx, _, _, _, _, cand_slice = best_cand
            matched_text = " ".join([w.raw_word for w in cand_slice])
            start_fmt = cand_slice[0].start_fmt
            end_fmt = cand_slice[-1].end_fmt

            prev_unmatched = False
            last_matched_end_idx = max(last_matched_end_idx, e_idx)

            alignment_results[idx] = {
                "sentence": sentence.raw_text,
                "start": start_fmt,
                "end": end_fmt,
                "confidence": round(conf, 2),
                "confidence_level": calibrate_confidence_level(conf),
                "matched": True,
                "matched_text": matched_text,
                "start_idx": s_idx,
                "end_idx": e_idx
            }
            continue

        # Sentence failed initial pass -> Emit [ALIGNMENT REJECTED] log
        rejection = RejectionReason.BELOW_THRESHOLD
        s_idx_b, _, _, _, _, _, cand_slice_b = best_cand
        matched_text_b = " ".join([w.raw_word for w in cand_slice_b]) if cand_slice_b else "N/A"

        if debug_alignment:
            print(
                f"\n[ALIGNMENT REJECTED]\n"
                f"Sentence: \"{sentence.raw_text}\"\n"
                f"Section: \"{sentence.section_name}\"\n"
                f"Block ID: {sentence.block_id}\n"
                f"Search Region: [{min_search_idx},{max_search_idx}]\n"
                f"Candidate Count: {len(candidate_windows)}\n"
                f"Best Candidate: \"{matched_text_b}\"\n"
                f"Best Score: {conf:.1f}",
                file=sys.stderr
            )

        # =========================================================================
        # IMPROVEMENT 3: NEIGHBOR-BASED RESCUE PASS
        # Recovers sentences skipped between two successfully matched neighbors:
        # rescue_min = max(0, prev_matched_end_idx - 20)
        # rescue_max = min(transcript_length - 1, next_matched_start_idx + 20)
        # =========================================================================
        prev_bound = max(0, last_matched_end_idx - NEIGHBOR_RESCUE_MARGIN)
        next_bound = total_transcript_words - 1

        # Look ahead to find next matched sentence start index
        for future_sent in all_sentences[idx + 1 : idx + 10]:
            f_block = sentence_block_map.get(future_sent.sentence_id)
            if f_block and f_block.start_idx is not None and f_block.start_idx > last_matched_end_idx:
                next_bound = min(total_transcript_words - 1, f_block.end_idx + NEIGHBOR_RESCUE_MARGIN)
                break

        rescue_min = prev_bound
        rescue_max = next_bound

        rescue_windows = generate_candidate_windows_bounded(
            sentence, transcript_words, word_frequency, inverted_index,
            min_search_idx=rescue_min, max_search_idx=rescue_max
        )

        r_cand, r_conf = evaluate_candidate_windows(
            sentence, rescue_windows, transcript_words,
            max_pause_seconds, pause_weight, tie_threshold
        )

        if r_cand and r_conf >= min_confidence:
            rs_idx, re_idx, _, _, _, _, r_slice = r_cand
            r_matched_text = " ".join([w.raw_word for w in r_slice])
            start_fmt = r_slice[0].start_fmt
            end_fmt = r_slice[-1].end_fmt

            prev_unmatched = False
            last_matched_end_idx = max(last_matched_end_idx, re_idx)

            if debug_alignment:
                print(
                    f"\n[RESCUE PASS SUCCESS]\n"
                    f"Sentence: \"{sentence.raw_text}\"\n"
                    f"Region: [{rs_idx}-{re_idx}]\n"
                    f"Score: {r_conf:.1f}",
                    file=sys.stderr
                )

            alignment_results[idx] = {
                "sentence": sentence.raw_text,
                "start": start_fmt,
                "end": end_fmt,
                "confidence": round(r_conf, 2),
                "confidence_level": calibrate_confidence_level(r_conf),
                "matched": True,
                "matched_text": r_matched_text,
                "start_idx": rs_idx,
                "end_idx": re_idx
            }
        else:
            failure_analytics[rejection.value] += 1
            prev_unmatched = True

            if debug_alignment:
                print(
                    f"\n[RESCUE PASS FAILED]\n"
                    f"Sentence: \"{sentence.raw_text}\"\n"
                    f"Best Rescue Score: {r_conf:.1f} < {min_confidence:.1f}",
                    file=sys.stderr
                )

            alignment_results[idx] = {
                "sentence": sentence.raw_text,
                "matched": False,
                "rejection_reason": rejection.value
            }

    # Alignment Failure Analytics Summary Output
    matched_count = sum(1 for r in alignment_results if r and r.get("matched") is not False)
    unmatched_count = len(alignment_results) - matched_count

    if debug_alignment:
        print("\n" + "=" * 50, file=sys.stderr)
        print("ALIGNMENT FAILURE ANALYTICS SUMMARY", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        print(f"Total Sentences:     {len(alignment_results)}", file=sys.stderr)
        print(f"Matched Sentences:   {matched_count} ({(matched_count/len(alignment_results)*100 if alignment_results else 0):.1f}%)", file=sys.stderr)
        print(f"Unmatched Sentences: {unmatched_count} ({(unmatched_count/len(alignment_results)*100 if alignment_results else 0):.1f}%)", file=sys.stderr)
        print("\nFailure Reason Breakdown:", file=sys.stderr)
        for reason, count in failure_analytics.items():
            print(f"  - {reason:<22}: {count}", file=sys.stderr)
        print("=" * 50, file=sys.stderr)

    return alignment_results


def main():
    parser = argparse.ArgumentParser(description="Bumblebee Stage 2 Alignment Engine with Persistent Logger")
    parser.add_argument("script_file", help="Path to ground-truth script text file")
    parser.add_argument("media_or_json_file", help="Path to video/audio file or transcript JSON file")
    parser.add_argument("--min-confidence", type=float, default=70.0, help="Minimum normal confidence threshold (default: 70.0)")
    parser.add_argument("--debug-alignment", action="store_true", default=DEBUG_ALIGNMENT, help="Enable detailed alignment diagnostics & failure analytics")

    args = parser.parse_args()

    # Initialize persistent Tee logger for run
    with create_run_logger() as logger:
        try:
            if args.media_or_json_file.endswith('.json'):
                with open(args.media_or_json_file, 'r', encoding='utf-8') as f:
                    transcript_tuples = json.load(f)
            else:
                from transcript import generate_transcript
                transcript_tuples = generate_transcript(args.media_or_json_file)

            results = align_script(
                transcript_tuples,
                args.script_file,
                min_confidence=args.min_confidence,
                debug_alignment=args.debug_alignment
            )
            print(json.dumps(results, indent=2))
        except Exception as e:
            print(f"Error during alignment: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
