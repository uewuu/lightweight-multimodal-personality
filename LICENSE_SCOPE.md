# License scope and public data-release boundary

`LICENSE` covers the code and documentation authored for this repository.

It does **not** relicense or redistribute:

- FI/FIv2 videos, audio, transcripts, annotations, labels, or feature caches;
- FI/FIv2-derived binary model checkpoints (`.ckpt`);
- pretrained DINOv2, WavLM, RoBERTa, or other upstream model weights;
- third-party packages such as LinMulT or Exordium.

This public release intentionally omits all four formal FI/FIv2-derived checkpoint binaries. Their expected filenames, endpoint selection rules, neural parameter counts, and SHA-256 fingerprints are retained in `checkpoints/README.md` and `artifacts/provenance/checkpoint_metadata.json` for provenance and verification. Omitting them is a conservative redistribution choice; it is not a legal conclusion that derived checkpoints are prohibited.

Public prediction CSVs contain only sample identifiers and model predictions. The teacher-prediction NPZ was rebuilt without its archived `targets` array. Raw FI/FIv2 media, annotations, labels, transcripts, and pre-extracted feature caches are not included. Users must obtain FI/FIv2 through its authorized access route and comply with the applicable terms.

Local labels or model weights generated from authorized FI/FIv2 access should remain outside this public repository unless the user has independently verified that redistribution is permitted.
