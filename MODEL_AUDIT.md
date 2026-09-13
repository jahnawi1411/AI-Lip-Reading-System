# Checkpoint compatibility audit

This is the historical audit of the original H5 checkpoint. The new public model
and its measured results are documented in [MODEL_PROGRESS.md](MODEL_PROGRESS.md).

The current H5 weights have SHA-256
`24f5b985a11b0d0b2fb102950a961ae805f3c547dd49b675239208bad0874f46`.
No weights were edited or trained.

The notebook vocabulary matches the application: lowercase alphabet, apostrophe,
question/exclamation marks, digits 1–9 and space; token 0 is the unknown symbol and
the last output class is CTC blank. H5 recurrent input dimensions require the
75-feature global-average-pooling architecture used by the current H5 loader,
rather than the much larger flattened feature tensor of the legacy checkpoint.
All trained layer shapes load successfully. Weight files do not store a complete
provenance record for unweighted layers or preprocessing.

Two notebook quirks were found:

1. `model.add(Conv3D(...), Activation('relu'))` passes an activation object as a
   second argument to `add`, rather than adding a separate activation layer.
   Depending on Keras version, that can fail or leave the activation absent.
2. The notebook subtracts a mean from an unsigned grayscale tensor before casting.
   Unsigned subtraction can wrap around. The current application normalizes after
   casting to floating point.

Both behaviors were examined with the same weights and verified GRID sample.

| Configuration | Greedy output |
| --- | --- |
| Current application | `l` and an unknown token |
| Omit third ReLU | `le e` and an unknown token |
| Notebook unsigned normalization | `la e` and unknown tokens |
| Both notebook variations | `la e o` and an unknown token |

The expected sentence is **set white with p two soon**. None of these variants
recovers it. No variant was selected based on a superficially better-looking result.
The current application remains unchanged. The output comparison is recorded in
`outputs/compatibility-audit.json`.

This audit narrows specific compatibility hypotheses but cannot establish the
training provenance or quality of the original checkpoint. Correct general
lip-reading cannot be promised with this weight file. A known-good compatible
checkpoint or verified original training artifact would be needed to proceed
beyond the tested limits without training; no replacement was made here.
