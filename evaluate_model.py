"""Evaluate held-out prepared sequences without training on test labels."""
import argparse
import json
from pathlib import Path

from dataset import load_manifest
from recognition import recognize_file, error_counts


def evaluate(manifest, weights=None, split="test"):
    rows = [r for r in load_manifest(manifest) if r["split"] == split]
    if not rows:
        raise ValueError(f"No examples in the {split} split.")
    results = []
    for row in rows:
        prediction = recognize_file(row["sequence"], weights=weights)
        counts = error_counts(row["text"], prediction["text"])
        results.append({"speaker": row["speaker"], "reference": row["text"], **prediction, **counts})
    word_errors = sum(r["word_errors"] for r in results)
    chars = sum(r["reference_characters"] for r in results)
    return {"split": split, "clips": len(results),
            "speakers": len({r["speaker"] for r in rows}),
            "wer": word_errors / sum(r["reference_words"] for r in results),
            "cer": sum(r["character_errors"] for r in results) / chars,
            "exact_match_rate": sum(r["exact_match"] for r in results) / len(results),
            "confidence_calibrated": False,
            "note": "WER/CER are corpus edit rates and may exceed 1. Character error rate includes spaces. No low-quality results are omitted.",
            "results": results}


def main():
    parser = argparse.ArgumentParser(description="Measure VisionSpeak word and character error rates")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Choose a new output filename.")
        report = evaluate(args.manifest, args.weights, args.split)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Clips: {report['clips']}; WER: {report['wer']:.2%}; CER: {report['cer']:.2%}")
        print(f"Saved: {args.output.resolve()}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
