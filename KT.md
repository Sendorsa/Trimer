# Knowledge Transfer (KT) Document: Bumblebee Video Trimming Pipeline

> [!NOTE]
> **Bumblebee** is an automated two-stage video trimming pipeline. Stage 1 generates word-level speech-to-text transcripts with FFmpeg timestamps. Stage 2 & 3 align ground-truth scripts with noisy transcripts using rare-word anchoring, monotonic ordering, RapidFuzz 4-part scoring, and **Stage 3 Phonetic Rescue Matching** for ASR pronunciation errors.

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
 [(word, start, end)] -------->| STAGE 2 & 3:          |
                               | align_script.py       |
                               | - align_script()      |
                               | - Inverted Indexing   |
                               | - Rare-Word Anchoring |
                               | - Monotonic Bounds    |
                               | - RapidFuzz Scoring   |
                               | - Pause Penalty       |
                               | - Retake Policy       |
                               | - PHONETIC RESCUE     |
                               +-----------+-----------+
                                           |
                                           v
                                 Aligned Sentences JSON
                         [{sentence, start, end, confidence, match_type}]
```

---

## 2. Repository File Structure

```
├── main.py                # Pipeline entry point connecting Stage 1 & Stage 2/3 in-memory
├── transcript.py          # Stage 1: In-memory word-level transcription (generate_transcript)
├── align_script.py        # Stage 2 & 3: Bumblebee alignment engine with Phonetic Rescue
├── test_align_script.py   # Unit & integration test suite for Stage 3 Phonetic Rescue
├── requirements.txt       # Dependencies (openai-whisper, torch, tqdm, rapidfuzz, jellyfish)
└── KT.md                  # Comprehensive Knowledge Transfer documentation
```

---

## 3. Stage 3: Phonetic Rescue Matching Fallback

### 3.1 Goal & Overview
Whisper STT frequently mishears specialized domain terms phonetically (e.g. *centimorgan* $\rightarrow$ *centi morgan*, *Sanger* $\rightarrow$ *Sangal*, *roadmap* $\rightarrow$ *load map*, * crossing over* $\rightarrow$ *cross over*, *nucleotides* $\rightarrow$ *nuclear types*).

Phonetic Rescue acts as a **targeted fallback layer** when normal fuzzy/coverage scoring yields `normal_confidence < min_confidence` (default 70.0).

### 3.2 Top-$N$ Bounded Search & Complexity Guarantee
To preserve linear scaling ($\mathcal{O}(N + M \cdot R \cdot K \cdot L)$) and eliminate false positives:
1. Phonetic search is **never** run across the entire transcript.
2. The engine filters the top $N$ fuzzy candidates already generated during normal scoring (default $N = 5$).
3. Converts clean text to phonetic representations (Metaphone/Soundex) via `jellyfish`.
4. Evaluates:
   $$\text{phonetic\_score} = \text{rapidfuzz.fuzz.ratio}(\text{phonetic\_normalize}(S), \text{phonetic\_normalize}(C))$$

### 3.3 Acceptance Criteria
A rescued phonetic match is accepted if:
$$\text{phonetic\_score} \ge \text{phonetic\_threshold} \quad (\text{default } 85.0) \quad \text{AND} \quad \text{phonetic\_score} > \text{normal\_confidence}$$

---

## 4. Configurable Hyperparameters Table

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_pause_seconds` | `float` | `1.5` | Maximum allowed inter-word gap in seconds before applying pause penalty |
| `pause_weight` | `float` | `5.0` | Penalty multiplier for each second exceeding `max_pause_seconds` |
| `tie_threshold` | `float` | `2.0` | Ranking score difference threshold to trigger retake preference policy |
| `min_confidence` | `float` | `70.0` | Minimum normal score required to accept a standard match |
| `monotonic_overlap` | `int` | `20` | Maximum allowed backward transcript word search overlap for strict chronological monotonicity |
| `phonetic_threshold` | `float` | `85.0` | Minimum score required for Phonetic Rescue fallback acceptance |
| `phonetic_top_candidates` | `int` | `5` | Number of top normal candidates evaluated phonetically |

---

## 5. Data Schema Specification

### 5.1 Normal Match (`"match_type": "normal"`)
```json
{
  "sentence": "Hello everyone.",
  "start": "00:00:28.740",
  "end": "00:00:30.020",
  "confidence": 98.2,
  "confidence_level": "HIGH",
  "matched": true,
  "match_type": "normal",
  "matched_text": "Hello everyone.",
  "start_idx": 25,
  "end_idx": 26
}
```

### 5.2 Phonetic Rescued Match (`"match_type": "phonetic"`)
```json
{
  "sentence": "The unit used is called a centimorgan",
  "start": "00:01:12.400",
  "end": "00:01:15.800",
  "confidence": 91.5,
  "confidence_level": "HIGH",
  "matched": true,
  "match_type": "phonetic",
  "matched_text": "The unit used is called a centi morgan",
  "start_idx": 110,
  "end_idx": 118
}
```

### 5.3 Unmatched Sentence Output Format (< min_confidence & failed rescue)
```json
{
  "sentence": "This sentence was never spoken in the video.",
  "matched": false
}
```

---

## 6. Complexity & Scalability Impact

Let:
* $N$ = Total transcript words (e.g. 10,000+)
* $M$ = Number of script sentences (e.g. 500+)
* $L$ = Average sentence word count ($\approx 10-15$)
* $K_p$ = Number of top candidates evaluated phonetically ($K_p = 5$)

Because phonetic normalization is applied **only to top $K_p=5$ candidates per low-confidence sentence**:
$$\text{Additional Phonetic Overhead} = \mathcal{O}(M_{\text{low\_conf}} \cdot K_p \cdot L) \ll \mathcal{O}(N)$$

The overall pipeline complexity remains **strictly linear** $\mathcal{O}(N + M \cdot R \cdot K \cdot L)$, with zero performance degradation on large transcripts.

---

## 7. Usage Guide & Commands

### 7.1 Running Pipeline via Terminal CLI
```bash
# Standard run
python3 main.py input_video.mp4 script.txt

# Custom phonetic threshold & debug logging
python3 main.py input_video.mp4 script.txt --phonetic-threshold 80.0 --debug
```

### 7.2 Running Test Suite
```bash
python3 test_align_script.py
```
