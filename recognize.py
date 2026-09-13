"""Run Part 2 from an existing Part 1 sequence or a recorded video."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from recognition import recognize_file, error_counts


def main():
    parser = argparse.ArgumentParser(description="VisionSpeak Part 2: visual speech recognition")
    parser.add_argument("input", type=Path)
    parser.add_argument("--mode", choices=["auto", "prepared", "mediapipe", "grid"], default="auto")
    parser.add_argument("--decoder", choices=["beam", "greedy"], default="greedy")
    parser.add_argument("--beam-width", type=int, default=50)
    parser.add_argument("--start", type=float, default=0, help="Start of speaking interval in seconds")
    parser.add_argument("--duration", type=float, help="Speaking interval: at most 16 seconds for auto-avsr, 10 for GRID")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--engine", choices=['original', 'lipnet-reference', 'auto-avsr'], default='original')
    parser.add_argument("--reference", help="Actual spoken words, for measuring this clip's error rate")
    parser.add_argument("--output", type=Path, help="Defaults to a new timestamped file under outputs")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("outputs") / f"recognition-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    try:
        if args.output.exists():
            raise ValueError("Output already exists; choose a new --output filename.")
        result = recognize_file(args.input, args.mode, args.weights, args.decoder,
                                args.beam_width, args.start, args.duration, args.engine)
        if args.reference is not None:
            counts = error_counts(args.reference, result["text"])
            result["evaluation"] = {**counts, "reference": args.reference,
                "wer": counts["word_errors"] / counts["reference_words"],
                "cer": counts["character_errors"] / counts["reference_characters"]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        displayed = result['text'].replace('\ufffd', '[unknown]') or '(no text decoded)'
        print(result["message"])
        if result["usable_text"]:
            print(f"Candidate text: {displayed}")
        else:
            print("Raw decoder output is saved in the result file for inspection.")
        print("Confidence: unavailable (not calibrated)")
        if result["model_score"] is not None:
            print(f"Raw model score: {result['model_score']:.4f} (not a correctness probability)")
        if result.get("alternatives"):
            print("Alternative candidates are saved in the result file.")
        if "evaluation" in result:
            print(f"This clip: WER {result['evaluation']['wer']:.1%}; CER {result['evaluation']['cer']:.1%}")
        for warning in result["warnings"]:
            print(f"Note: {warning}")
        print(f"Result saved: {args.output.resolve()}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
