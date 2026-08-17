# Feature extraction specification

The paper's downstream system uses four pre-extracted streams:

| Stream | Width | Released evidence / implementation |
|---|---:|---|
| DINOv2-face | 384 | `preprocessing/dinov2_face.py`, `preprocessing/fi_visual_dinov2.py` |
| WavLM | 768 | `preprocessing/fi/fi_acoustic.py`, `preprocessing/add_wavlm_to_cache.py` |
| RoBERTa | 1024 | `preprocessing/fi/fi_textual.py` |
| eGeMAPS-LLD | 25 | `preprocessing/fi/fi_acoustic.py` |

## DINOv2-face

The archived final script uses `dinov2_vits14` (384-D), 224×224 face crops, ImageNet normalization, OpenCV/RetinaFace-based face processing, and per-video temporal features.

## WavLM

The archived acoustic script uses 16 kHz audio and WavLM `base+`, yielding 768-D temporal features.

## eGeMAPS

The archived acoustic script uses openSMILE eGeMAPS low-level descriptors (LLD), yielding 25-D temporal features.

## RoBERTa

The formal textual path is `preprocessing/fi/fi_textual.py`, which uses the archived Exordium RoBERTa wrapper and FI transcripts. The final downstream width is 1024.

## Final downstream pad/crop limits

The archived FI loader used by the formal lightweight runs applies the following time limits:

- DINOv2-face: 450
- WavLM: 1500
- RoBERTa: 80
- eGeMAPS-LLD: 1500

The historical preprocessing repository also contained legacy/alternative streams. They are intentionally omitted from the main released preprocessing path so that DeBERTa-v3, wav2vec2, OpenGraphAU, or FabNet are not mistaken for the paper's final four-stream input.

## Processed sample set

Use `artifacts/manifests/` as the authoritative list of samples included in the reported experiments. The manifests are membership/provenance files; evaluation code should join by `sample_id` rather than relying on their row order.
