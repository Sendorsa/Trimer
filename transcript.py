#!/usr/bin/env python3

import os
import sys
import site
import warnings

warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

try:
    import whisper
except ImportError:
    print("Error: 'openai-whisper' package is required. Install it using:\n  pip install openai-whisper", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def format_ffmpeg_timestamp(seconds: float) -> str:
    if seconds is None or seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def generate_transcript(video_path: str, model_size: str = "base") -> list:
    """
    Transcribes video into word-by-word timestamps.

    Returns:
        List[Tuple[str, str, str]]: [("word", "start_fmt", "end_fmt"), ...]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at path: '{video_path}'")

    print(f"[1/2] Loading Whisper model ('{model_size}')...", file=sys.stderr)
    model = whisper.load_model(model_size)

    print(f"[2/2] Transcribing '{video_path}'...", file=sys.stderr)
    result = model.transcribe(video_path, word_timestamps=True, verbose=False)

    word_tuples = []
    segments = result.get("segments", [])
    
    iterator = tqdm(segments, desc="Extracting word timestamps", unit="segment", file=sys.stderr) if tqdm else segments

    for segment in iterator:
        for word_info in segment.get("words", []):
            word = word_info.get("word", "").strip()
            if not word:
                continue
            start_time = format_ffmpeg_timestamp(word_info.get("start", 0.0))
            end_time = format_ffmpeg_timestamp(word_info.get("end", 0.0))
            word_tuples.append((word, start_time, end_time))

    return word_tuples


# Alias for backward compatibility
transcribe_video = generate_transcript


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 transcript.py <video_file_path> [model_size]", file=sys.stderr)
        print("Example: python3 transcript.py input_video.mp4", file=sys.stderr)
        sys.exit(1)

    video_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"

    try:
        results = generate_transcript(video_file, model_size=model_size)
        print(results)
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
