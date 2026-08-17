# Release audit — GitHub Public Release v1.0

Build date: 2026-08-17

## Public-release checkpoint policy

All four FI/FIv2-derived `.ckpt` binaries were removed from this public release. Exact fingerprints are retained for provenance:

- Full seed42 / valid-R² endpoint / 5,476,680 params: `7771bf086cf6e740e89bea7ad83af164e00df3fc777104659ae6539edbe8edc5`
- D-KD seed42 / valid-RACC endpoint / 500,107 params: `7f34cfe4ab8d4961eaea0ccaa2c1239670a5c57ad152025c648d664f6e655346`
- F seed42 / valid-RACC endpoint / 500,107 params: `d41315fd8b3297cc10822d599a3438dd7beab8ba27166cb20f4e0851254bb48a`
- Generic-Joint seed42 / valid-RACC endpoint / 500,875 params: `ecac4d7ede8f9c78e28be3dc2349d81f4a19b849653f7dfcca81e79c77716cce`

Before public removal, the RC2 build QA verified strict loading for the formal F and Generic-Joint checkpoints against the released student model implementation and confirmed the expected parameter counts. The public release retains fingerprints and configurations but not binary weights.

## Metric recomputation from archived predictions (private labels used only during build QA)

- Full: RACC 0.916473661783, R² 0.485586631090, PCC 0.697162199489
- D-KD: RACC 0.916189595352, R² 0.481809686294, PCC 0.695661062565
- D-DCS: RACC 0.916362803791, R² 0.485638011789, PCC 0.695661062565
- F: RACC 0.915532398930, R² 0.472860144008, PCC 0.690358473764
- F-Cal: RACC 0.915837120112, R² 0.478622835290, PCC 0.690358473764
- Generic-Joint: RACC 0.915422481, R² 0.471153477, PCC 0.688459877

## Source-video cluster bootstrap QA (Full → D-DCS)

10,000 iterations, seed 42, 1,999 test samples, 1,454 source-video clusters:

- RACC candidate-minus-baseline: point −0.000110858, 95% CI [−0.001250364, +0.001063923]
- R² candidate-minus-baseline: point +0.000051381, 95% CI [−0.013042151, +0.013946445]
- PCC candidate-minus-baseline: point −0.001501137, 95% CI [−0.010653349, +0.007965397]

All intervals include zero; the release preserves the manuscript's near-Full-retention interpretation rather than superiority/equivalence.

## Direct structural-control QA (Generic-Joint → F)

Formal seed-42 Generic-Joint: 500,875 trainable parameters, `generic_concat_mlp`, rank 8, Prediction KD off, calibration off, validation-RACC endpoint. During private build QA, all 9,995 legacy Generic-Joint GroundTruth values matched the formal F unified target matrix exactly (maximum absolute difference 0.0). The public merged Generic-Joint prediction file retains only `sample_id` and `pred_*` columns.

10,000 paired source-video cluster-bootstrap iterations, seed 42, 1,999 test samples, 1,454 source-video clusters:

- RACC F-minus-Generic: point +0.000109918, 95% CI [−0.000556950, +0.000772199]
- R² F-minus-Generic: point +0.001706667, 95% CI [−0.005942891, +0.009115879]
- PCC F-minus-Generic: point +0.001898597, 95% CI [−0.003083620, +0.006731299]

All intervals include zero. The release preserves the manuscript interpretation: favorable seed-42 point estimates for the multiplicative formulation, but no statistically resolved structural superiority.

## Public code and privacy QA

- `FinalStudent(config.yaml)` instantiates with 500,107 trainable parameters.
- `FinalStudent(configs/generic_joint_seed42.yaml)` instantiates with 500,875 trainable parameters.
- Public prediction CSVs contain no FI target/ground-truth columns.
- The public teacher NPZ contains no `targets` array.
- Raw FI/FIv2 media, transcripts, annotations, labels, and feature caches are absent.
- All `.ckpt` binaries are absent from the public release.
- Reference environment files have local editable-install and Conda-prefix paths removed.
- Machine-specific absolute paths are absent from public files; sanitized provenance YAMLs use placeholders where historical paths existed.
- No `__pycache__` directory is included.

## Known non-blocking limitations

- The exact five FI clips absent from the finalized processed cache are not identified; the included sample set is fully specified by the three manifests.
- D-KD validation predictions are not statically redistributed. They can be regenerated locally with authorized data and a compatible locally held/retrained model.
- The formal seed-42 Generic-Joint direct structural control is frozen. No additional F/Generic multi-seed training is planned for this release; the manuscript treats the direct operator comparison as single-seed structural evidence.
- The broader environment snapshot is a reproducibility reference, not an exact historical package freeze.
- `CITATION.cff` is deferred until the final author list/publication metadata are confirmed.
