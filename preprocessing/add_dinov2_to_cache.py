from pathlib import Path
from tqdm import tqdm
import argparse
import pickle
import shutil
import numpy as np
import os


SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(script_dir: Path) -> Path:
    """
    自动向上查找项目根目录。
    目标根目录应包含 personalitylinmult 和 datas。
    """
    for candidate in [script_dir, *script_dir.parents]:
        if (candidate / "personalitylinmult").exists() and (candidate / "datas").exists():
            return candidate
    return script_dir.parent.parent


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower().strip()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def find_dinov2_file(dinov2_root: Path, video_id: str) -> Path | None:
    """
    支持：
    datas/dinov2_face/video_id.pkl
    datas/dinov2_face/test80_01/video_id.pkl
    datas/dinov2_face/validation80_25/video_id.pkl
    等任意子目录结构。
    """
    direct = dinov2_root / f"{video_id}.pkl"
    if direct.exists():
        return direct

    matches = list(dinov2_root.rglob(f"{video_id}.pkl"))
    if matches:
        return matches[0]

    return None


def load_dinov2_feature(path: Path) -> np.ndarray:
    """
    兼容两种格式：
    1. (frame_ids, feature)
    2. feature
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, tuple) and len(obj) == 2:
        _, feat = obj
    else:
        feat = obj

    feat = np.asarray(feat, dtype=np.float32)

    if feat.ndim != 2:
        raise ValueError(f"DINOv2 feature ndim != 2, shape={feat.shape}, path={path}")

    if feat.shape[0] == 0:
        raise ValueError(f"DINOv2 feature is empty, shape={feat.shape}, path={path}")

    if feat.shape[1] != 384:
        raise ValueError(f"DINOv2 feature dim should be 384, got shape={feat.shape}, path={path}")

    if not np.isfinite(feat).all():
        raise ValueError(f"DINOv2 feature contains NaN/Inf, shape={feat.shape}, path={path}")

    return feat


def save_split_cache(samples: dict, split_dir: Path, force: bool):
    if split_dir.exists() and force:
        shutil.rmtree(split_dir)

    split_dir.mkdir(parents=True, exist_ok=True)

    for video_id, sample in tqdm(samples.items(), desc=f"write {split_dir.name}"):
        out_file = split_dir / f"{video_id}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)


def add_dinov2_to_cache(
    cache_dir: Path,
    dinov2_dir: Path,
    subset: str,
    output_suffix: str,
    force: bool,
    save_split: bool,
):
    old_cache = cache_dir / f"fi_{subset}.pkl"
    new_cache = cache_dir / f"fi_{subset}{output_suffix}.pkl"
    split_dir = cache_dir / f"split_{subset}{output_suffix}"

    if not old_cache.exists():
        raise FileNotFoundError(f"Cannot find old cache: {old_cache}")

    if new_cache.exists() and not force:
        print(f"[SKIP] New cache already exists: {new_cache}")
        return

    print(f"\n[LOAD] {old_cache}")
    with open(old_cache, "rb") as f:
        samples = pickle.load(f)

    print(f"[INFO] subset={subset}, samples={len(samples)}")

    new_samples = {}
    skipped = 0
    missing = 0
    invalid = 0

    for video_id, sample in tqdm(samples.items(), desc=f"add dinov2 {subset}"):
        try:
            dinov2_path = find_dinov2_file(dinov2_dir, video_id)
            if dinov2_path is None:
                missing += 1
                raise FileNotFoundError(f"missing dinov2 feature: {video_id}")

            dinov2_feat = load_dinov2_feature(dinov2_path)

            # 关键：复制旧 sample，不改旧对象
            new_sample = dict(sample)
            new_sample["dinov2_face"] = dinov2_feat

            # 确保标签还在
            if "ocean" not in new_sample:
                raise KeyError(f"sample has no ocean label: {video_id}")

            new_samples[video_id] = new_sample

        except Exception as e:
            skipped += 1
            invalid += 1
            print(f"[INVALID] {video_id} | {e}")
            continue

    print(f"\n[DONE] subset={subset}")
    print(f"  old samples: {len(samples)}")
    print(f"  new samples: {len(new_samples)}")
    print(f"  skipped:     {skipped}")
    print(f"  missing:     {missing}")
    print(f"  invalid:     {invalid}")
    print(f"  completion:  {len(new_samples) / len(samples) * 100:.2f}%")

    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SAVE] {new_cache}")
    with open(new_cache, "wb") as f:
        pickle.dump(new_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    if save_split:
        print(f"[SAVE SPLIT] {split_dir}")
        save_split_cache(new_samples, split_dir, force=force)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="datas/cache",
        help="Old cache directory. Default: datas/cache",
    )
    parser.add_argument(
        "--dinov2_dir",
        type=str,
        default="datas/dinov2_face",
        help="DINOv2 feature directory. Default: datas/dinov2_face",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_dinov2",
        help="Output suffix. Default: _dinov2",
    )
    parser.add_argument(
        "--force",
        type=str2bool,
        default=True,
        help="Overwrite existing new cache. Default: true",
    )
    parser.add_argument(
        "--save_split",
        type=str2bool,
        default=True,
        help="Also save split cache folders. Default: true",
    )

    args = parser.parse_args()

    cache_dir = resolve_path(args.cache_dir)
    dinov2_dir = resolve_path(args.dinov2_dir)

    print("=" * 70)
    print("Add DINOv2 feature to existing FI cache")
    print("=" * 70)
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Cache dir:     {cache_dir}")
    print(f"DINOv2 dir:    {dinov2_dir}")
    print(f"Output suffix: {args.output_suffix}")
    print(f"Force:         {args.force}")
    print(f"Save split:    {args.save_split}")
    print("=" * 70)

    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}")
    if not dinov2_dir.exists():
        raise FileNotFoundError(f"DINOv2 dir not found: {dinov2_dir}")

    for subset in ["train", "valid", "test"]:
        add_dinov2_to_cache(
            cache_dir=cache_dir,
            dinov2_dir=dinov2_dir,
            subset=subset,
            output_suffix=args.output_suffix,
            force=args.force,
            save_split=args.save_split,
        )


if __name__ == "__main__":
    main()