"""Command-line entry point for VisionSpeak's first project milestone."""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from preprocessing import DEFAULT_MODEL, prepare_video

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def download_model(destination):
    destination = Path(destination)
    if destination.exists():
        print(f"Model already exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        with urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded face landmark model: {destination}")


def main():
    parser = argparse.ArgumentParser(description="VisionSpeak Part 1: prepare lips from a short video.")
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("outputs/prepared"))
    parser.add_argument("--download-model", action="store_true")
    args = parser.parse_args()
    try:
        if args.download_model:
            download_model(args.model)
        if args.video is None:
            if args.download_model:
                return
            parser.error("Provide a video file or --download-model.")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Output folder is not empty. Choose a new --output folder.")
        frames, crops, report = prepare_video(args.video, args.model)
        args.output.mkdir(parents=True, exist_ok=True)
        # Fifteen evenly spaced crops give a compact visual inspection sheet.
        tiles = [crops[i] for i in np.linspace(0, 74, 15).astype(int)]
        sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, 15, 5)])
        if not cv2.imwrite(str(args.output / "lip_preview.png"), sheet):
            raise OSError("Could not write the preview image.")
        np.savez_compressed(args.output / "sequence.npz", frames=frames,
                            source_indices=np.asarray(report["source_indices"]))
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Prepared {report['source_frames']} source frames -> {frames.shape}")
        print(f"Preview, sequence and report saved to: {args.output.resolve()}")
        print("Part 1 prepares visual input; speech recognition is Part 2.")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
