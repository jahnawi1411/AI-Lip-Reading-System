"""Fine-tune the recurrent head on Part 1 sequences; preserve source weights."""
import argparse
import json
from pathlib import Path

import numpy as np

from dataset import load_manifest, encode_text
from recognition import load_sequence, weight_fingerprint, BLANK


def ctc_loss_for_text(probabilities, text):
    import tensorflow as tf
    labels = encode_text(text)
    return tf.reduce_mean(tf.nn.ctc_loss(
        labels=tf.constant([labels], tf.int32),
        logits=tf.math.log(tf.clip_by_value(probabilities, 1e-7, 1.0)),
        label_length=[len(labels)], logit_length=[75],
        logits_time_major=False, blank_index=BLANK))


def train(manifest, output, epochs=10, learning_rate=1e-4, weights=None):
    if epochs < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs and learning rate must be positive.")
    rows = load_manifest(manifest)
    training = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    if not training or not validation:
        raise ValueError("Training requires nonempty train and validation splits with different speakers.")
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new output directory to preserve previous training runs.")
    import tensorflow as tf
    from lip_reader import build_model_with_weights, configure_tensorflow
    configure_tensorflow()
    tf.keras.utils.set_random_seed(42)
    model, source = build_model_with_weights(weights)
    if source.suffix != ".h5":
        raise ValueError("Head adaptation currently requires the GAP .weights.h5 checkpoint; pass --weights models/checkpoint.weights.h5.")
    # Head adaptation keeps CPU memory bounded; CNN fine-tuning is a separate experiment.
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv3D):
            layer.trainable = False
    feature_model = tf.keras.Model(model.inputs[0], model.layers[9].output)
    head_input = tf.keras.Input(shape=feature_model.output_shape[1:])
    head_output = head_input
    for layer in model.layers[10:]:
        head_output = layer(head_output)
    head = tf.keras.Model(head_input, head_output)
    optimizer = tf.keras.optimizers.Adam(learning_rate, clipnorm=5.0)
    output.mkdir(parents=True)
    history, best = [], float("inf")
    rng = np.random.default_rng(42)
    for epoch in range(epochs):
        losses = []
        for index in rng.permutation(len(training)):
            row = training[int(index)]
            frames = load_sequence(row["sequence"])[None]
            features = tf.stop_gradient(feature_model(frames, training=False))
            with tf.GradientTape() as tape:
                loss = ctc_loss_for_text(head(features, training=True), row["text"])
            if not bool(tf.math.is_finite(loss)):
                raise ValueError("Nonfinite training loss; check input data and labels.")
            gradients = tape.gradient(loss, head.trainable_variables)
            optimizer.apply_gradients(zip(gradients, head.trainable_variables))
            losses.append(float(loss))
        val_losses = [float(ctc_loss_for_text(model(load_sequence(r["sequence"])[None], training=False), r["text"]))
                      for r in validation]
        val_loss = float(np.mean(val_losses))
        if not np.isfinite(val_loss):
            raise ValueError("Nonfinite validation loss.")
        history.append({"epoch": epoch + 1, "train_ctc_loss": float(np.mean(losses)), "validation_ctc_loss": val_loss})
        if val_loss < best:
            best = val_loss
            model.save_weights(output / "best.weights.h5")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch + 1}/{epochs}: train loss {np.mean(losses):.4f}; validation loss {val_loss:.4f}", flush=True)
    metadata = {"source_weights": str(source.resolve()), "source_sha256": weight_fingerprint(source),
                "preprocessing": "Part 1 uniform-75 grayscale lip crops, sequence standardization",
                "training": "frozen convolution layers; recurrent head adaptation", "seed": 42,
                "learning_rate": learning_rate, "manifest": rows, "best_validation_loss": best,
                "test_used_for_training": False, "confidence_calibrated": False}
    (output / "training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output / "best.weights.h5"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VisionSpeak on labelled Part 1 sequences")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    try:
        print(f"Saved best weights: {train(args.manifest, args.output, args.epochs, args.learning_rate, args.weights)}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
