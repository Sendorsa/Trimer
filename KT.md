# Knowledge Transfer (KT) Document: Bumblebee Video Trimming Pipeline

> [!NOTE]
> **Bumblebee** is an automated two-stage video trimming pipeline. Stage 1 generates word-level speech-to-text transcripts with FFmpeg timestamps. Stage 2 provides **Alignment Diagnostics, Hierarchical Traceability Logging, Adaptive Search Windows, Fallback Regional Search, Structural Block Recovery, and Alignment Failure Analytics**.

---

## 1. System Overview & Architecture

```
+---------------------+       +------------------------+
|  Input Video/Audio  |       |  Ground-Truth Script   |
|   (e.g., video.mp4) |       |   (e.g., script.txt)   |
+----------+----------+       +-----------+------------+
           |                              |
           v                              |
+---------------------+                   |
| STAGE 1:            |                   |
| transcript.py       |                   |
| - generate_transcript()                 |
| - Whisper STT Engine|                   |
| - tqdm Progress Bar |                   |
+----------+----------+                   |
           |                              |
           | (In-Memory Data Flow)        |
           v                              v
   Transcript Tuples           +-----------------------+
 [(word, start, end)] -------->| STAGE 2:              |
                               | align_script.py       |
                               | - Traceability Logs   |
                               | - Adaptive Windows    |
                               | - Regional Fallbacks  |
                               | - Block Recovery      |
                               | - Failure Analytics   |
                               +-----------+-----------+
                                           |
                                           v
                                 Aligned Sentences JSON
                         [{sentence, start, end, confidence}]
```

---

## 2. Repository File Structure

```
├── main.py                # Pipeline entry point connecting Stage 1 & Stage 2 in-memory
├── transcript.py          # Stage 1: In-memory word-level transcription (generate_transcript)
├── align_script.py        # Stage 2: Bumblebee alignment engine with Diagnostics & Adaptive Localization
├── test_align_script.py   # Unit & integration test suite for Stage 2
├── requirements.txt       # Dependencies manifest (openai-whisper, torch, tqdm, rapidfuzz)
└── KT.md                  # Comprehensive Knowledge Transfer documentation
```

---

## 3. Stage 2 Diagnostics & Adaptive Localization Features

### 3.1 Alignment Diagnostics & Rejection Logging
When `DEBUG_ALIGNMENT = True` or `--debug-alignment` is passed, detailed diagnostic logs are emitted to `sys.stderr` for every unmatched sentence:

```text
[ALIGNMENT REJECTED]
  Sentence: "This sentence was never spoken in the video."
  Section: "Types of Gene Mapping"
  Block ID: 1
  Search Range: [2, 22]
  Candidate Count: 21
  Best Candidate Text: "genome sequencing."
  Best Score: 18.95
  Rejection Reason: BELOW_THRESHOLD
```

#### Rejection Reason Categories (`RejectionReason` Enum)
* `NO_CANDIDATES`: Inverted index anchor search returned 0 candidates in search region.
* `BELOW_THRESHOLD`: Candidates found, but best score fell below `min_confidence` (default 70.0).
* `MONOTONIC_REJECTED`: Candidate rejected because start index violated monotonic ordering.
* `OUTSIDE_WINDOW`: Candidates fell outside allowed adaptive search bounds.
* `BLOCK_ALIGNMENT_FAILED`: Parent block failed coarse localization.

---

### 3.2 Hierarchical Traceability Logging
Logs coarse localization bounds for sections, blocks, and individual sentences:

```text
[SECTION ALIGNMENT] Section: "Types of Gene Mapping" | Region: [3-22]
[BLOCK ALIGNMENT]   Block 0: "1. Genetic Mapping (Linkage Mapping)..." | Region: [3-22]
[SENTENCE ALIGNMENT] Sentence: "1. Genetic Mapping (Linkage Mapping)" | Region: [8-12] (Score: 100.0)
```

---

### 3.3 Adaptive Search Windows
Search windows expand dynamically based on alignment difficulty:
* `BASE_SEARCH_WINDOW = 150` transcript words.
* `MAX_SEARCH_WINDOW = 600` transcript words.
* Window expands when:
  - Previous sentence was unmatched (+150)
  - Block confidence was low (+150)
  - Section recently changed (+150)

---

### 3.4 3-Stage Fallback Regional Search
When initial candidate window search yields no candidates or low confidence, the engine triggers a 3-pass regional search:
1. **Pass 1 (Local Window)**: `[min_search_idx, min_search_idx + adaptive_window]`
2. **Pass 2 (Expanded Block Window)**: `[min_search_idx, min_search_idx + MAX_SEARCH_WINDOW]`
3. **Pass 3 (Full Monotonic Range)**: `[0, total_transcript_words - 1]` (bounded by inverted index anchors; never performs unanchored full scans).

---

### 3.5 Structural Block Recovery
Short structural headers (e.g. `Advantages:`, `Limitations:`, `Applications:`, `1.`) are automatically merged with content into single searchable blocks (`Advantages: High Accuracy`, `Limitations: Time Consuming`).

---

### 3.6 Alignment Failure Analytics Summary
After script alignment completes, an analytical report is output:

```text
==================================================
ALIGNMENT FAILURE ANALYTICS SUMMARY
==================================================
Total Sentences:     7
Matched Sentences:   6 (85.7%)
Unmatched Sentences: 1 (14.3%)

Failure Reason Breakdown:
  - NO_CANDIDATES         : 0
  - BELOW_THRESHOLD       : 1
  - MONOTONIC_REJECTED    : 0
  - OUTSIDE_WINDOW        : 0
  - BLOCK_ALIGNMENT_FAILED: 0
==================================================
```

---

## 4. Complexity & Scalability

Let:
* $N$ = Total transcript words (e.g. 10,000+)
* $M$ = Number of script sentences (e.g. 500+)
* $L$ = Average sentence word count ($\approx 10-15$)
* $R$ = Frequency of rare anchor words ($R \ll N$)

The adaptive window and 3-pass regional fallback remain anchored to inverted index word positions.
Overall pipeline complexity remains strictly linear: $\mathbf{\mathcal{O}(N + M \cdot R \cdot K \cdot L)}$.

---

## 5. Usage Commands

```bash
# Run test suite
python3 test_align_script.py

# Run main pipeline with diagnostic logging
python3 main.py input_video.mp4 script.txt --debug-alignment
```
