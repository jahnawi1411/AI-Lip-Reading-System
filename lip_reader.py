import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
VOCAB = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = tf.keras.layers.StringLookup(vocabulary=VOCAB, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def configure_tensorflow() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass


def load_video(path: Path) -> tf.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (288, 360):
                raise ValueError("Legacy GRID input requires 360x288 video. Use recognize.py --mode mediapipe for other framing.")
            # Preserve the original checkpoint's BGR-as-RGB grayscale convention.
            frame = tf.image.rgb_to_grayscale(frame)
            frames.append(frame[190:236, 80:220, :])
            if len(frames) > 75:
                raise ValueError("Legacy GRID input requires exactly 75 frames. Use recognize.py for Part 1 sequences.")
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames were read from video: {path}")
    if len(frames) != 75:
        raise ValueError(f"Expected 75 GRID frames, received {len(frames)}.")

    frames = tf.cast(tf.stack(frames), tf.float32)
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(frames)
    return (frames - mean) / tf.maximum(std, tf.keras.backend.epsilon())


def load_alignments(path: Path) -> tf.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Could not find alignment file: {path}")

    tokens = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 3 and parts[2] != "sil":
                tokens.extend([" ", parts[2]])

    chars = tf.strings.unicode_split(tokens, input_encoding="UTF-8")
    return char_to_num(tf.reshape(chars, (-1)))[1:]


def load_data(video_path: Path) -> tuple[tf.Tensor, tf.Tensor]:
    video_path = Path(video_path)
    alignment_path = DATA_DIR / "alignments" / "s1" / f"{video_path.stem}.align"
    return load_video(video_path), load_alignments(alignment_path)


def tensor_to_text(values: Iterable[int]) -> str:
    chars = num_to_char(tf.convert_to_tensor(values)).numpy()
    return b"".join(chars).decode("utf-8").replace("[UNK]", "").strip()


def build_model(feature_projection: str = "flatten") -> tf.keras.Model:
    if feature_projection not in {"flatten", "gap"}:
        raise ValueError("feature_projection must be 'flatten' or 'gap'")

    projection_layer = (
        tf.keras.layers.Flatten()
        if feature_projection == "flatten"
        else tf.keras.layers.GlobalAveragePooling2D()
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(75, 46, 140, 1)),
            tf.keras.layers.Conv3D(128, 3, padding="same"),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling3D((1, 2, 2)),
            tf.keras.layers.Conv3D(256, 3, padding="same"),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling3D((1, 2, 2)),
            tf.keras.layers.Conv3D(75, 3, padding="same"),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling3D((1, 2, 2)),
            tf.keras.layers.TimeDistributed(projection_layer),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    128, kernel_initializer="Orthogonal", return_sequences=True
                )
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    128, kernel_initializer="Orthogonal", return_sequences=True
                )
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                char_to_num.vocabulary_size() + 1,
                kernel_initializer="he_normal",
                activation="softmax",
            ),
        ]
    )
    return model


def _weighted_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    return [layer for layer in model.layers if layer.weights]


def _load_legacy_checkpoint(model: tf.keras.Model, checkpoint_prefix: Path) -> None:
    prefix = str(checkpoint_prefix)

    def variable(name: str) -> np.ndarray:
        return tf.train.load_variable(prefix, f"{name}/.ATTRIBUTES/VARIABLE_VALUE")

    weighted = _weighted_layers(model)
    if len(weighted) != 6:
        raise ValueError("Unexpected model layout for legacy checkpoint loading.")

    for index in range(3):
        weighted[index].set_weights(
            [
                variable(f"layer_with_weights-{index}/kernel"),
                variable(f"layer_with_weights-{index}/bias"),
            ]
        )

    for layer_index, checkpoint_index in [(3, 3), (4, 4)]:
        weighted[layer_index].set_weights(
            [
                variable(f"layer_with_weights-{checkpoint_index}/forward_layer/cell/kernel"),
                variable(
                    f"layer_with_weights-{checkpoint_index}/forward_layer/cell/recurrent_kernel"
                ),
                variable(f"layer_with_weights-{checkpoint_index}/forward_layer/cell/bias"),
                variable(f"layer_with_weights-{checkpoint_index}/backward_layer/cell/kernel"),
                variable(
                    f"layer_with_weights-{checkpoint_index}/backward_layer/cell/recurrent_kernel"
                ),
                variable(f"layer_with_weights-{checkpoint_index}/backward_layer/cell/bias"),
            ]
        )

    weighted[5].set_weights(
        [variable("layer_with_weights-5/kernel"), variable("layer_with_weights-5/bias")]
    )


def _looks_like_tensorflow_checkpoint(path: Path) -> bool:
    return Path(str(path) + ".index").is_file()


def build_model_with_weights(
    weights_path: Path | None = None,
) -> tuple[tf.keras.Model, Path]:
    explicit_weights = Path(weights_path) if weights_path is not None else None
    legacy_checkpoint = MODEL_DIR / "checkpoint"
    h5_checkpoint = MODEL_DIR / "checkpoint.weights.h5"

    if explicit_weights and explicit_weights.suffix == ".h5":
        model = build_model("gap")
        model.load_weights(str(explicit_weights))
        return model, explicit_weights

    if explicit_weights and _looks_like_tensorflow_checkpoint(explicit_weights):
        model = build_model("flatten")
        _load_legacy_checkpoint(model, explicit_weights)
        return model, explicit_weights

    if explicit_weights is not None:
        raise FileNotFoundError(f"Requested weights not found or unsupported: {explicit_weights}")

    if _looks_like_tensorflow_checkpoint(legacy_checkpoint):
        model = build_model("flatten")
        _load_legacy_checkpoint(model, legacy_checkpoint)
        return model, legacy_checkpoint

    if h5_checkpoint.exists():
        model = build_model("gap")
        model.load_weights(str(h5_checkpoint))
        return model, h5_checkpoint

    raise FileNotFoundError(
        "No usable weights found. Expected models/checkpoint.index or "
        "models/checkpoint.weights.h5."
    )


_MODEL_CACHE: dict[str, tuple[tf.keras.Model, Path]] = {}


def get_model(weights_path: Path | None = None) -> tuple[tf.keras.Model, Path]:
    cache_key = str(Path(weights_path).resolve()) if weights_path else "__default__"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = build_model_with_weights(weights_path)
    return _MODEL_CACHE[cache_key]


def predict_text(
    video_path: Path,
    weights_path: Path | None = None,
    include_alignment: bool = True,
) -> tuple[str | None, str, Path]:
    configure_tensorflow()
    model, loaded_from = get_model(weights_path)

    frames = load_video(video_path)
    batch = tf.expand_dims(frames, axis=0)
    yhat = model.predict(batch, verbose=0)
    decoded = tf.keras.backend.ctc_decode(
        yhat, input_length=np.full((batch.shape[0],), yhat.shape[1]), greedy=True
    )[0][0].numpy()[0]

    prediction = tensor_to_text(decoded[decoded >= 0])
    real_text = None
    if include_alignment:
        alignment_path = DATA_DIR / "alignments" / "s1" / f"{Path(video_path).stem}.align"
        if alignment_path.exists():
            real_text = tensor_to_text(load_alignments(alignment_path).numpy())

    return real_text, prediction, loaded_from


def predict(video_path: Path, weights_path: Path | None = None) -> tuple[str | None, str]:
    real_text, prediction, loaded_from = predict_text(video_path, weights_path)
    print(f"Weights: {loaded_from}")
    print(f"Video:   {video_path}")
    if real_text is not None:
        print(f"Real:    {real_text}")
    print(f"Pred:    {prediction}")
    return real_text, prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lip-reading inference.")
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DATA_DIR / "s1" / "bbaf2n.mpg"),
        help="Path to a GRID .mpg video. Defaults to data/s1/bbaf2n.mpg.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional path to a Keras .weights.h5 file or TensorFlow checkpoint prefix.",
    )
    args = parser.parse_args()

    predict(Path(args.video), Path(args.weights) if args.weights else None)


if __name__ == "__main__":
    main()
