import argparse
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm


# ============================================================
# Add WavLM features into existing FI cache.
#
# Input examples:
#   datas/cache/fi_train.pkl
#   datas/cache/fi_valid.pkl
#   datas/cache/fi_test.pkl
#
#   or split shards:
#   datas/cache/split_train/*.pkl
#   datas/cache/split_valid/*.pkl
#   datas/cache/split_test/*.pkl
#
# WavLM feature input:
#   datas/wavlm/<video_id>.npy
#
# Output examples:
#   datas/cache/fi_train_wavlm.pkl
#   datas/cache/fi_valid_wavlm.pkl
#   datas/cache/fi_test_wavlm.pkl
#
#   datas/cache/split_train_wavlm/*.pkl
#   datas/cache/split_valid_wavlm/*.pkl
#   datas/cache/split_test_wavlm/*.pkl
#
# Each sample will get:
#   sample["wavlm"] = np.ndarray, shape = (T, 768)
# ============================================================


SUBSETS = ("train", "valid", "test")
FEATURE_KEY = "wavlm"


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def infer_video_id_from_sample(sample: Dict[str, Any], fallback: str | None = None) -> str | None:
    """
    Try common id fields first; otherwise use fallback.
    """
    for key in (
        "video_id",
        "id",
        "name",
        "filename",
        "file",
        "video",
        "video_name",
        "youtube_id",
    ):
        if key in sample:
            value = sample[key]
            if isinstance(value, (str, Path)):
                return Path(str(value)).stem
    return fallback


def normalize_wavlm_feature(x: np.ndarray, path: Path) -> np.ndarray:
    """
    Force WavLM feature to float32 2D array.

    Expected normal shape:
      (T, 768)

    If shape is (1, T, 768), squeeze first dimension.
    """
    x = np.asarray(x)

    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.ndim != 2:
        raise ValueError(f"Invalid WavLM feature shape in {path}: {x.shape}. Expected (T, C).")

    if x.shape[-1] not in (768, 1024):
        print(f"[WARN] Unexpected WavLM dim in {path}: {x.shape}")

    return x.astype(np.float32, copy=False)


class WavLMFinder:
    def __init__(self, wavlm_dir: Path):
        self.wavlm_dir = wavlm_dir
        self._index: Dict[str, Path] | None = None

    def _build_index(self) -> Dict[str, Path]:
        print(f"[INDEX] Building WavLM index from: {self.wavlm_dir}")
        index: Dict[str, Path] = {}
        for p in self.wavlm_dir.rglob("*.npy"):
            index[p.stem] = p
        print(f"[INDEX] Indexed WavLM files: {len(index)}")
        return index

    def find(self, video_id: str) -> Path | None:
        # Fast flat-path check first.
        direct = self.wavlm_dir / f"{video_id}.npy"
        if direct.exists():
            return direct

        # Fallback recursive index.
        if self._index is None:
            self._index = self._build_index()
        return self._index.get(video_id)


def add_wavlm_to_sample(
    sample: Dict[str, Any],
    video_id: str,
    finder: WavLMFinder,
) -> Tuple[Dict[str, Any] | None, str | None]:
    wavlm_path = finder.find(video_id)
    if wavlm_path is None:
        return None, "missing"

    try:
        wavlm = np.load(wavlm_path)
        wavlm = normalize_wavlm_feature(wavlm, wavlm_path)
    except Exception as e:
        return None, f"invalid:{e}"

    new_sample = dict(sample)
    new_sample[FEATURE_KEY] = wavlm
    return new_sample, None


def iter_cache_items(cache_obj: Any) -> Iterable[Tuple[str | None, Dict[str, Any]]]:
    """
    Support common cache formats:
    1. dict[video_id] = sample_dict
    2. list[sample_dict]
    """
    if isinstance(cache_obj, dict):
        for key, value in cache_obj.items():
            if isinstance(value, dict):
                yield str(key), value
            else:
                raise TypeError(f"Unsupported dict cache value type for key={key}: {type(value)}")
    elif isinstance(cache_obj, list):
        for value in cache_obj:
            if isinstance(value, dict):
                yield None, value
            else:
                raise TypeError(f"Unsupported list cache item type: {type(value)}")
    else:
        raise TypeError(f"Unsupported cache object type: {type(cache_obj)}")


def rebuild_cache_like(original: Any, new_items: List[Tuple[str | None, Dict[str, Any]]]) -> Any:
    """
    Preserve original top-level cache type when possible.
    """
    if isinstance(original, dict):
        return {key: sample for key, sample in new_items if key is not None}
    if isinstance(original, list):
        return [sample for _, sample in new_items]
    raise TypeError(f"Unsupported cache object type: {type(original)}")


def save_split_from_items(
    items: List[Tuple[str | None, Dict[str, Any]]],
    split_dir: Path,
    force: bool,
) -> None:
    if split_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output split dir already exists: {split_dir}. "
                "Use --force true to overwrite."
            )
        shutil.rmtree(split_dir)

    split_dir.mkdir(parents=True, exist_ok=True)

    for i, (video_id, sample) in enumerate(tqdm(items, desc=f"save {split_dir.name}")):
        file_stem = video_id if video_id is not None else f"{i:06d}"
        save_pickle(sample, split_dir / f"{file_stem}.pkl")


def process_full_cache_subset(
    subset: str,
    cache_dir: Path,
    finder: WavLMFinder,
    output_suffix: str,
    force: bool,
    save_split: bool,
) -> bool:
    input_path = cache_dir / f"fi_{subset}.pkl"
    output_path = cache_dir / f"fi_{subset}{output_suffix}.pkl"
    split_output_dir = cache_dir / f"split_{subset}{output_suffix}"

    if not input_path.exists():
        return False

    if output_path.exists() and not force:
        raise FileExistsError(f"Output cache already exists: {output_path}. Use --force true to overwrite.")

    print(f"\n[LOAD] {input_path}")
    cache_obj = load_pickle(input_path)

    new_items: List[Tuple[str | None, Dict[str, Any]]] = []
    missing, invalid = [], []

    items = list(iter_cache_items(cache_obj))
    print(f"[INFO] subset={subset}, samples={len(items)}")

    for fallback_id, sample in tqdm(items, desc=f"add wavlm {subset}"):
        video_id = infer_video_id_from_sample(sample, fallback=fallback_id)
        if video_id is None:
            invalid.append("<unknown_id>")
            continue

        new_sample, err = add_wavlm_to_sample(sample, video_id, finder)
        if err is None and new_sample is not None:
            new_items.append((fallback_id or video_id, new_sample))
        elif err == "missing":
            missing.append(video_id)
        else:
            invalid.append(f"{video_id}:{err}")

    new_cache = rebuild_cache_like(cache_obj, new_items)

    print(f"\n[DONE] subset={subset}")
    print(f"  old samples: {len(items)}")
    print(f"  new samples: {len(new_items)}")
    print(f"  missing:     {len(missing)}")
    print(f"  invalid:     {len(invalid)}")
    print(f"  completion:  {len(new_items) / max(len(items), 1) * 100:.2f}%")

    if missing:
        print(f"  missing examples: {missing[:10]}")
    if invalid:
        print(f"  invalid examples: {invalid[:10]}")

    print(f"[SAVE] {output_path}")
    save_pickle(new_cache, output_path)

    if save_split:
        print(f"[SAVE SPLIT] {split_output_dir}")
        save_split_from_items(new_items, split_output_dir, force=force)

    return True


def process_split_cache_subset(
    subset: str,
    cache_dir: Path,
    finder: WavLMFinder,
    output_suffix: str,
    force: bool,
) -> bool:
    input_dir = cache_dir / f"split_{subset}"
    output_dir = cache_dir / f"split_{subset}{output_suffix}"

    if not input_dir.exists():
        return False

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output split dir already exists: {output_dir}. Use --force true to overwrite.")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.pkl"))
    print(f"\n[LOAD SPLIT] {input_dir}")
    print(f"[INFO] subset={subset}, shard files={len(files)}")

    missing, invalid, saved = [], [], 0

    for path in tqdm(files, desc=f"add wavlm split {subset}"):
        sample = load_pickle(path)
        if not isinstance(sample, dict):
            invalid.append(f"{path.name}:not_dict")
            continue

        # For split shards, filename stem is usually the video id.
        video_id = infer_video_id_from_sample(sample, fallback=path.stem)
        if video_id is None:
            invalid.append(f"{path.name}:unknown_id")
            continue

        new_sample, err = add_wavlm_to_sample(sample, video_id, finder)
        if err is None and new_sample is not None:
            save_pickle(new_sample, output_dir / path.name)
            saved += 1
        elif err == "missing":
            missing.append(video_id)
        else:
            invalid.append(f"{video_id}:{err}")

    print(f"\n[DONE SPLIT] subset={subset}")
    print(f"  old shards:  {len(files)}")
    print(f"  new shards:  {saved}")
    print(f"  missing:     {len(missing)}")
    print(f"  invalid:     {len(invalid)}")
    print(f"  completion:  {saved / max(len(files), 1) * 100:.2f}%")

    if missing:
        print(f"  missing examples: {missing[:10]}")
    if invalid:
        print(f"  invalid examples: {invalid[:10]}")

    print(f"[SAVE SPLIT] {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Add WavLM features to existing FI cache.")
    parser.add_argument("--cache_dir", type=str, default="datas/cache", help="Path to FI cache directory.")
    parser.add_argument("--wavlm_dir", type=str, default="datas/wavlm", help="Path to WavLM .npy feature directory.")
    parser.add_argument("--output_suffix", type=str, default="_wavlm", help="Suffix for output cache/split.")
    parser.add_argument("--force", type=str2bool, default=False, help="Overwrite existing output files/dirs.")
    parser.add_argument("--save_split", type=str2bool, default=True, help="When processing full pkl cache, also save split shards.")
    parser.add_argument(
        "--prefer_split",
        type=str2bool,
        default=False,
        help=(
            "If true, process split_train/valid/test directly. "
            "If false, process fi_train/valid/test first when available, then fall back to split."
        ),
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    wavlm_dir = Path(args.wavlm_dir)

    if not cache_dir.exists():
        raise FileNotFoundError(f"cache_dir does not exist: {cache_dir}")
    if not wavlm_dir.exists():
        raise FileNotFoundError(f"wavlm_dir does not exist: {wavlm_dir}")

    print("=" * 70)
    print("Add WavLM feature to existing FI cache")
    print("=" * 70)
    print(f"Cache dir:     {cache_dir.resolve()}")
    print(f"WavLM dir:     {wavlm_dir.resolve()}")
    print(f"Output suffix: {args.output_suffix}")
    print(f"Force:         {args.force}")
    print(f"Save split:    {args.save_split}")
    print(f"Prefer split:  {args.prefer_split}")
    print("=" * 70)

    finder = WavLMFinder(wavlm_dir)

    for subset in SUBSETS:
        processed = False

        if args.prefer_split:
            processed = process_split_cache_subset(
                subset=subset,
                cache_dir=cache_dir,
                finder=finder,
                output_suffix=args.output_suffix,
                force=args.force,
            )
            if not processed:
                processed = process_full_cache_subset(
                    subset=subset,
                    cache_dir=cache_dir,
                    finder=finder,
                    output_suffix=args.output_suffix,
                    force=args.force,
                    save_split=args.save_split,
                )
        else:
            processed = process_full_cache_subset(
                subset=subset,
                cache_dir=cache_dir,
                finder=finder,
                output_suffix=args.output_suffix,
                force=args.force,
                save_split=args.save_split,
            )
            if not processed:
                processed = process_split_cache_subset(
                    subset=subset,
                    cache_dir=cache_dir,
                    finder=finder,
                    output_suffix=args.output_suffix,
                    force=args.force,
                )

        if not processed:
            print(f"[WARN] No cache found for subset={subset}. Tried fi_{subset}.pkl and split_{subset}/")


if __name__ == "__main__":
    main()
