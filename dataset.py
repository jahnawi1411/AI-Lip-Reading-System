"""Validated, speaker-disjoint JSON manifests for prepared lip sequences."""
import json
from pathlib import Path

from recognition import VOCAB, load_sequence


def load_manifest(path):
    path = Path(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty JSON list.")
    speakers, paths = {}, set()
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k].strip()
                for k in ("sequence", "text", "speaker", "split")):
            raise ValueError("Each row needs sequence, text, speaker and split strings.")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Split must be train, validation or test.")
        row["speaker"] = row["speaker"].strip().casefold()
        prior = speakers.setdefault(row["speaker"], row["split"])
        if prior != row["split"]:
            raise ValueError(f"Speaker leakage across splits: {row['speaker']}")
        source = (path.parent / row["sequence"]).resolve()
        if source in paths:
            raise ValueError(f"Duplicate sequence: {source}")
        paths.add(source)
        row["sequence"] = str(source)
        row["text"] = " ".join(row["text"].lower().split())
        unsupported = set(row["text"]) - set(VOCAB)
        if unsupported:
            raise ValueError(f"Unsupported transcript characters: {sorted(unsupported)}. Normalize punctuation explicitly.")
        required_steps = len(row["text"]) + sum(a == b for a, b in zip(row["text"], row["text"][1:]))
        if required_steps > 75:
            raise ValueError("Transcript cannot align to 75 CTC steps (including repeated letters).")
        load_sequence(source)
    return rows


def encode_text(text):
    return [VOCAB.index(char) + 1 for char in text]
