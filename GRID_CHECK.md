# Existing checkpoint: matched GRID sample check

No training, model replacement or weight changes were performed.

## Test input

- Video: `data/grid_check/id2_vcd_swwp2s.mpg`
- Alignment: `data/grid_check/swwp2s.align`
- Reference, read from alignment: **set white with p two soon**
- Verified: 360 x 288 pixels, 25 FPS, 75 decoded frames (3 seconds).
- Visually inspected frame 35 and the original `[190:236, 80:220]` crop: the crop
  contains the mouth. No MediaPipe crop or temporal resampling was used.
- Original grayscale convention and sequence normalization were preserved.
- Tensor shape: `(75, 46, 140, 1)`; all values finite, mean approximately zero,
  standard deviation approximately one.

Source: the public [LipNet evaluation sample directory](https://github.com/rizkiarm/LipNet/tree/bec592aa1e66378bc7ab33f6110f1c8a431b3b9a/evaluation/samples).
The exact download URLs and file hashes are stored in `data/grid_check/sources.json`.
The corpus clip may contain audio, but inference only decoded image frames.

## Result

| Item | Value |
| --- | --- |
| Reference | set white with p two soon |
| Raw prediction | l followed by an unknown token |
| Token IDs | 12, 0 |
| Word errors | 6 / 6 (100% WER) |
| Character errors | 25 / 25 (100% CER, including spaces) |
| App decoder versus TensorFlow greedy decoder | Exact agreement |
| Calibrated confidence | Unavailable |

Reports: `outputs/grid-check-greedy.json` and
`outputs/grid-decoder-crosscheck.json`.

The model fails on this correctly formatted GRID example too. The sample-video
failure therefore cannot be attributed solely to its different dimensions,
duration, or MediaPipe preprocessing. Our custom greedy decoder agrees with
TensorFlow, so decoder implementation is not the explanation for this result.

One example does not establish dataset-wide accuracy or prove weight corruption.
The remaining investigation concerns the checkpoint's learned quality and whether
the reconstructed architecture, vocabulary and original training preprocessing
exactly match how these weights were produced. Successful weight loading alone
does not establish those facts. No cause beyond these observations is claimed.

SHA-256 before and after inference (identical):
`24f5b985a11b0d0b2fb102950a961ae805f3c547dd49b675239208bad0874f46`.

## Repeat the check

In the activated project environment:

```cmd
python recognize.py data\grid_check\id2_vcd_swwp2s.mpg --mode grid --reference "set white with p two soon"
```

The command generates a new timestamped result file. Downloaded test data remains
in the ignored `data` directory; the original checkpoint is unchanged.
