from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import pickle
import os
import shutil


# ============================================================
# FI cache preprocessing for FGSM-SDI-master project structure
#
# Expected project structure in your current project:
#   FGSM-SDI-master/
#   ├── datas/
#   │   ├── cache/
#   │   ├── dinov2_face/
#   │   ├── egemaps_lld/
#   │   ├── fabnet/
#   │   ├── opengraphau/
#   │   ├── text/
#   │   └── wav2vec2/
#   └── data/db/FI/gt/annotation_*.pkl   # common original GT path
#
# This script supports:
#   1. reading features from datas/
#   2. adding dinov2_face into cache
#   3. generating fi_train/valid/test*.pkl
#   4. optionally generating split_train/split_valid/split_test
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(script_dir: Path) -> Path:
    """
    Locate the FGSM-SDI-master project root robustly.

    Expected location:
      FGSM-SDI-master/personalitylinmult/preprocess/fi_preprocess.py

    The project root should contain both:
      - personalitylinmult/
      - datas/  or data/
    """
    for candidate in [script_dir, *script_dir.parents]:
        if (candidate / "personalitylinmult").exists() and (
            (candidate / "datas").exists() or (candidate / "data").exists()
        ):
            return candidate

    # Fallback for the current project structure:
    # preprocess -> personalitylinmult -> FGSM-SDI-master
    return script_dir.parent.parent


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

GT_NAME = {
    "test": "annotation_test",
    "valid": "annotation_validation",
    "train": "annotation_training",
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower().strip()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def resolve_project_path(path_str: str) -> Path:
    """Resolve a user path relative to project root unless it is absolute."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def find_existing_dir(candidates):
    for p in candidates:
        p = Path(p)
        if p.exists() and p.is_dir():
            return p
    return Path(candidates[0])


def find_gt_file(gt_root: Path, subset: str) -> Path:
    """Find annotation_*.pkl in several common GT layouts."""
    filename = f"{GT_NAME[subset]}.pkl"
    candidates = [
        gt_root / filename,
        gt_root / "gt" / filename,
        PROJECT_ROOT / "data" / "db" / "FI" / "gt" / filename,
        PROJECT_ROOT / "datas" / "gt" / filename,
        PROJECT_ROOT / "data" / "db_processed" / "fi" / "gt" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find ground-truth file for subset "
        f"'{subset}'. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def find_file_in_subdirs(base_dir, filename, extension=".pkl"):
    """
    Find filename + extension in base_dir and all nested subdirectories.

    Compatible with:
      base_dir/video_id.pkl
      base_dir/test80_01/video_id.pkl
      base_dir/training80_01/video_id.pkl
      base_dir/validation80_25/video_id.pkl
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return None

    direct_path = base_path / f"{filename}{extension}"
    if direct_path.exists():
        return direct_path

    # rglob is safer for your current nested video-id folders.
    matches = list(base_path.rglob(f"{filename}{extension}"))
    if matches:
        return matches[0]

    return None


def load_pickle_feature(path):
    """
    Load pkl feature.

    Compatible formats:
      1. (frame_ids, features)
      2. features
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, tuple) and len(obj) == 2:
        _, feat = obj
    else:
        feat = obj

    return np.asarray(feat, dtype=np.float32)


def load_gt(subset: str, gt_root: Path):
    ocean_name = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]

    gt_file = find_gt_file(gt_root, subset)
    with open(gt_file, "rb") as f:
        gt_dict = pickle.load(f, encoding="latin1")

    video_ids = [elem for elem in gt_dict["openness"].keys()]

    gt = {}
    for video_id in video_ids:
        ocean = []
        for trait in ocean_name:
            ocean.append(gt_dict[trait][video_id])

        # Cut .mp4 from the end of filenames.
        gt[video_id[:-4]] = np.array(ocean, dtype=np.float32)

    return gt


def validate_feature(name, feat, video_id, expected_dim=None):
    if not isinstance(feat, np.ndarray):
        raise ValueError(f"{name} is not numpy array: {video_id}")

    if feat.ndim != 2:
        raise ValueError(f"{name} ndim != 2: {video_id}, shape={feat.shape}")

    if feat.shape[0] == 0:
        raise ValueError(f"{name} empty sequence: {video_id}, shape={feat.shape}")

    if expected_dim is not None and feat.shape[1] != expected_dim:
        raise ValueError(
            f"{name} feature dim mismatch: {video_id}, "
            f"shape={feat.shape}, expected second dim={expected_dim}"
        )

    if not np.isfinite(feat).all():
        raise ValueError(f"{name} contains NaN/Inf: {video_id}, shape={feat.shape}")


def save_split_cache(samples, split_dir: Path, force_rebuild: bool):
    if split_dir.exists() and force_rebuild:
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    for video_id, sample in tqdm(samples.items(), desc=f"write {split_dir.name}"):
        out_file = split_dir / f"{video_id}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_cache(
    subset,
    roberta_dict,
    bert_dict,
    processed_root: Path,
    gt_root: Path,
    include_dinov2=True,
    include_legacy=True,
    force_rebuild=False,
    output_suffix="",
    save_split=False,
):
    gt = load_gt(subset, gt_root)

    cache_name = f"fi_{subset}{output_suffix}.pkl"
    output_path = processed_root / "cache" / cache_name

    split_dir = processed_root / "cache" / f"split_{subset}{output_suffix}"

    if output_path.exists() and not force_rebuild:
        print(f"[SKIP] cache already exists: {output_path}")
        print("       Use --force_rebuild true if you want to rebuild it.")
        return

    samples = {}
    skipped_count = 0
    missing_counter = {}

    for video_id, ocean in tqdm(gt.items(), total=len(gt), desc=f"{subset}"):
        try:
            # Required features for your current 5-modal experiment.
            opengraphau_path = find_file_in_subdirs(processed_root / "opengraphau", video_id, ".pkl")
            egemaps_lld_path = processed_root / "egemaps_lld" / f"{video_id}.npy"
            wav2vec2_path = processed_root / "wav2vec2" / f"{video_id}.npy"

            dinov2_path = None
            if include_dinov2:
                dinov2_path = find_file_in_subdirs(processed_root / "dinov2_face", video_id, ".pkl")

            # Legacy optional features. Keep them for compatibility with old experiments.
            fabnet_path = None
            if include_legacy:
                fabnet_path = find_file_in_subdirs(processed_root / "fabnet", video_id, ".pkl")

            missing_files = []
            if not opengraphau_path:
                missing_files.append(f"opengraphau/{video_id}.pkl")
            if include_dinov2 and not dinov2_path:
                missing_files.append(f"dinov2_face/{video_id}.pkl")
            if not egemaps_lld_path.exists():
                missing_files.append(f"egemaps_lld/{video_id}.npy")
            if not wav2vec2_path.exists():
                missing_files.append(f"wav2vec2/{video_id}.npy")
            if video_id not in roberta_dict:
                missing_files.append(f"roberta_dict[{video_id}]")

            if include_legacy:
                if not fabnet_path:
                    missing_files.append(f"fabnet/{video_id}.pkl")
                if video_id not in bert_dict:
                    missing_files.append(f"bert_dict[{video_id}]")

            if missing_files:
                for item in missing_files:
                    key = item.split("/")[0].split("[")[0]
                    missing_counter[key] = missing_counter.get(key, 0) + 1
                raise FileNotFoundError(f"missing files: {', '.join(missing_files)}")

            # Load required features.
            egemaps_lld = np.asarray(np.load(egemaps_lld_path), dtype=np.float32)
            wav2vec2 = np.asarray(np.load(wav2vec2_path), dtype=np.float32)
            opengraphau = load_pickle_feature(opengraphau_path)
            roberta = np.asarray(roberta_dict[video_id], dtype=np.float32)

            sample_dict = {
                "egemaps_lld": egemaps_lld,
                "opengraphau": opengraphau,
                "wav2vec2": wav2vec2,
                "roberta": roberta,
                "ocean": ocean,
            }

            if include_dinov2:
                sample_dict["dinov2_face"] = load_pickle_feature(dinov2_path)

            if include_legacy:
                sample_dict["fabnet"] = load_pickle_feature(fabnet_path)
                sample_dict["bert"] = np.asarray(bert_dict[video_id], dtype=np.float32)

            # Validate feature shapes. Dims are checked where stable.
            validate_feature("egemaps_lld", sample_dict["egemaps_lld"], video_id)
            validate_feature("opengraphau", sample_dict["opengraphau"], video_id, expected_dim=41)
            validate_feature("wav2vec2", sample_dict["wav2vec2"], video_id, expected_dim=768)
            validate_feature("roberta", sample_dict["roberta"], video_id, expected_dim=1024)

            if include_dinov2:
                validate_feature("dinov2_face", sample_dict["dinov2_face"], video_id, expected_dim=384)
            if include_legacy:
                validate_feature("fabnet", sample_dict["fabnet"], video_id)
                validate_feature("bert", sample_dict["bert"], video_id, expected_dim=768)

        except Exception as e:
            print(f"Invalid sample: {video_id} | {e}")
            skipped_count += 1
            continue

        samples[video_id] = sample_dict

    print(f"\n{subset.upper()} done")
    print(f"  success: {len(samples)}")
    print(f"  skipped: {skipped_count}")
    print(f"  completion: {len(samples) / len(gt) * 100:.2f}%")
    if missing_counter:
        print("  missing summary:")
        for k, v in sorted(missing_counter.items()):
            print(f"    {k}: {v}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved to: {output_path}")

    if save_split:
        save_split_cache(samples, split_dir=split_dir, force_rebuild=force_rebuild)
        print(f"  split cache saved to: {split_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed_dir",
        type=str,
        default="datas",
        help="Feature/cache root. For your current project, use datas. Default: datas.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="data/db/FI",
        help=(
            "Ground-truth root or gt directory. Common value: data/db/FI. "
            "The script will also try data/db/FI/gt and datas/gt automatically."
        ),
    )
    parser.add_argument(
        "--include_dinov2",
        type=str2bool,
        default=True,
        help="Whether to include dinov2_face in cache. Default: true.",
    )
    parser.add_argument(
        "--include_legacy",
        type=str2bool,
        default=True,
        help="Whether to keep fabnet and bert in cache for old experiments. Default: true.",
    )
    parser.add_argument(
        "--force_rebuild",
        type=str2bool,
        default=False,
        help="Whether to rebuild existing cache files. Default: false.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help=(
            "Suffix for output cache name. "
            "For example, --output_suffix _dinov2 generates fi_train_dinov2.pkl. "
            "Default empty string writes fi_train.pkl if force_rebuild=true."
        ),
    )
    parser.add_argument(
        "--save_split",
        type=str2bool,
        default=False,
        help=(
            "Whether to also generate split cache directories. "
            "With --output_suffix _dinov2, output dirs are split_train_dinov2 etc. "
            "Default: false."
        ),
    )

    args = parser.parse_args()

    processed_root = resolve_project_path(args.processed_dir)
    gt_root = resolve_project_path(args.gt_dir)

    print("=" * 70)
    print("FI cache preprocessing")
    print("=" * 70)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Processed root:   {processed_root}")
    print(f"GT root:          {gt_root}")
    print(f"include_dinov2:   {args.include_dinov2}")
    print(f"include_legacy:   {args.include_legacy}")
    print(f"force_rebuild:    {args.force_rebuild}")
    print(f"output_suffix:    '{args.output_suffix}'")
    print(f"save_split:       {args.save_split}")
    print("=" * 70)

    roberta_path = processed_root / "text" / "fi_roberta.pkl"
    bert_path = processed_root / "text" / "fi_bert.pkl"

    if not roberta_path.exists():
        raise FileNotFoundError(f"Cannot find RoBERTa feature file: {roberta_path}")
    if args.include_legacy and not bert_path.exists():
        raise FileNotFoundError(f"Cannot find BERT feature file: {bert_path}")

    with open(roberta_path, "rb") as f:
        roberta_dict = pickle.load(f)

    if args.include_legacy:
        with open(bert_path, "rb") as f:
            bert_dict = pickle.load(f)
    else:
        bert_dict = {}

    for subset in ["train", "valid", "test"]:
        build_cache(
            subset=subset,
            roberta_dict=roberta_dict,
            bert_dict=bert_dict,
            processed_root=processed_root,
            gt_root=gt_root,
            include_dinov2=args.include_dinov2,
            include_legacy=args.include_legacy,
            force_rebuild=args.force_rebuild,
            output_suffix=args.output_suffix,
            save_split=args.save_split,
        )


if __name__ == "__main__":
    main()
