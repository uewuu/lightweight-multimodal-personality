import pickle
import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from exordium.utils.normalize import get_mean_std, standardization
from exordium.utils.padding import pad_or_crop_time_dim


def custom_collate_fn(batch):
    """
    Collate dict samples into a batch while preserving masks and labels.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}

    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [sample[key] for sample in batch]
        first_val = values[0]

        if isinstance(first_val, torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        elif isinstance(first_val, np.ndarray):
            collated[key] = torch.from_numpy(np.stack(values, axis=0))
        else:
            collated[key] = values

    return collated


def get_default_db_root() -> Path:
    return Path(__file__).resolve().parents[2] / "datas"


def resolve_db_root(config: dict | None = None) -> Path:
    if config is not None and config.get("db_root"):
        return Path(config["db_root"])
    return get_default_db_root()


def get_split_dir(db_root: Path, subset: str) -> Path:
    return db_root / "cache" / f"split_{subset}"


class EgemapsDataset(Dataset):
    """
    Used only for computing standardization statistics.
    Supports both split shards and legacy monolithic pkl.
    """

    def __init__(self, subset: str, db_root: str | Path):
        self.subset = subset
        self.db_root = Path(db_root)
        self.samples = self._load_samples()

    def _load_samples(self):
        split_dir = get_split_dir(self.db_root, self.subset)
        if split_dir.exists():
            files = sorted(split_dir.glob("*.pkl"))
            samples = []
            for fp in files:
                with open(fp, "rb") as f:
                    record = pickle.load(f)
                samples.append(record["egemaps_lld"])
            return samples

        monolithic = self.db_root / "cache" / f"fi_{self.subset}.pkl"
        with open(monolithic, "rb") as f:
            records = pickle.load(f)
        return [records[video_id]["egemaps_lld"] for video_id in records]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class AuDataset(Dataset):
    """
    Used only for computing standardization statistics.
    Supports both split shards and legacy monolithic pkl.
    """

    def __init__(self, subset: str, db_root: str | Path):
        self.subset = subset
        self.db_root = Path(db_root)
        self.samples = self._load_samples()

    def _load_samples(self):
        split_dir = get_split_dir(self.db_root, self.subset)
        if split_dir.exists():
            files = sorted(split_dir.glob("*.pkl"))
            samples = []
            for fp in files:
                with open(fp, "rb") as f:
                    record = pickle.load(f)
                samples.append(record["opengraphau"])
            return samples

        monolithic = self.db_root / "cache" / f"fi_{self.subset}.pkl"
        with open(monolithic, "rb") as f:
            records = pickle.load(f)
        return [records[video_id]["opengraphau"] for video_id in records]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TensorDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], 0


def calculate_standardization(db_root: str | Path):
    """
    Compute mean/std for egemaps_lld and opengraphau from train split.
    Supports both split shards and legacy monolithic pkl.
    """
    db_root = Path(db_root)

    egemaps_path = db_root / "standardization" / "egemaps_lld.npz"
    if not egemaps_path.exists():
        ds_egemaps = EgemapsDataset("train", db_root)
        samples_egemaps = np.vstack(ds_egemaps.samples)
        egemaps_path.parent.mkdir(parents=True, exist_ok=True)
        mean, std = get_mean_std(
            DataLoader(TensorDataset(samples_egemaps), batch_size=100, shuffle=False),
            ndim=2,
        )
        np.savez(str(egemaps_path), mean=mean, std=std)

    au_path = db_root / "standardization" / "opengraphau.npz"
    if not au_path.exists():
        ds_au = AuDataset("train", db_root)
        samples_au = np.vstack(ds_au.samples)
        au_path.parent.mkdir(parents=True, exist_ok=True)
        mean, std = get_mean_std(
            DataLoader(TensorDataset(samples_au), batch_size=100, shuffle=False),
            ndim=2,
        )
        np.savez(str(au_path), mean=mean, std=std)


class FiDataset(Dataset):
    """
    Dataset that supports:
    1) split shards:
       cache/split_train/*.pkl
       cache/split_valid/*.pkl
       cache/split_test/*.pkl
    2) legacy monolithic pickle fallback:
       cache/fi_train.pkl, fi_valid.pkl, fi_test.pkl

    This version returns only the modalities requested by config["feature_list"].
    Supported current modalities:
    - egemaps_lld
    - opengraphau
    - dinov2_face
    - wav2vec2
    - wavlm
    - roberta
    - deberta_v3
    """

    def __init__(self, subset: str, config: dict | None = None):
        self.config = config if config is not None else {}
        self.subset = subset
        self.db_root = resolve_db_root(self.config)
        self.target_id = self.config.get("target_id", None)
        self.feature_list = list(
            self.config.get(
                "feature_list",
                ["opengraphau", "wav2vec2", "roberta", "egemaps_lld"],
            )
        )
        self.standardization_params = self._load_standardization_params()
        self.sample_files = self._load_sample_file_index()

    def _load_standardization_params(self):
        d = {}

        egemaps_data = np.load(self.db_root / "standardization" / "egemaps_lld.npz")
        d["egemaps_lld"] = {
            "mean": torch.FloatTensor(egemaps_data["mean"]),
            "std": torch.FloatTensor(egemaps_data["std"]),
        }

        au_data = np.load(self.db_root / "standardization" / "opengraphau.npz")
        d["opengraphau"] = {
            "mean": torch.FloatTensor(au_data["mean"]),
            "std": torch.FloatTensor(au_data["std"]),
        }

        return d

    def _get_subset_max_samples(self):
        if self.subset == "train":
            return self.config.get("train_max_samples", None)
        if self.subset == "valid":
            return self.config.get("valid_max_samples", None)
        if self.subset == "test":
            return self.config.get("test_max_samples", None)
        return None

    def _sample_paths(self, all_paths: list[Path], max_samples):
        if max_samples is None or max_samples >= len(all_paths):
            return all_paths

        sample_seed = int(self.config.get("sample_seed", self.config.get("seed", 42)))
        rng = random.Random(sample_seed)
        sampled = rng.sample(all_paths, int(max_samples))
        return sorted(sampled)

    def _convert_monolithic_to_temp_index(self, sample_path: Path):
        """
        Fallback only. If split_{subset} does not exist, load the monolithic pickle and
        build an in-memory temp list of records.
        """
        print(f"[FiDataset] Fallback to legacy monolithic file: {sample_path}")

        with open(sample_path, "rb") as f:
            records = pickle.load(f)

        all_ids = list(records.keys())
        max_samples = self._get_subset_max_samples()
        if max_samples is not None and max_samples < len(all_ids):
            sample_seed = int(self.config.get("sample_seed", self.config.get("seed", 42)))
            rng = random.Random(sample_seed)
            all_ids = rng.sample(all_ids, int(max_samples))

        temp_items = [records[video_id] for video_id in all_ids]
        del records
        return temp_items

    def _load_sample_file_index(self):
        """
        Preferred mode: use split shards.
        Fallback mode: use old monolithic pickle.
        """
        split_dir = get_split_dir(self.db_root, self.subset)
        if split_dir.exists():
            all_files = sorted(split_dir.glob("*.pkl"))
            if not all_files:
                raise FileNotFoundError(f"[FiDataset] Split dir exists but is empty: {split_dir}")

            max_samples = self._get_subset_max_samples()
            selected_files = self._sample_paths(all_files, max_samples)

            print(f"[FiDataset] Using split shards from: {split_dir}")
            print(f"[FiDataset] Total shard files: {len(all_files)}")
            print(f"[FiDataset] Selected shard files: {len(selected_files)}")
            return selected_files

        monolithic = self.db_root / "cache" / f"fi_{self.subset}.pkl"
        if monolithic.exists():
            return self._convert_monolithic_to_temp_index(monolithic)

        raise FileNotFoundError(
            f"[FiDataset] Neither split dir nor monolithic file found for subset='{self.subset}'. "
            f"Expected one of:\n"
            f"  {split_dir}\n"
            f"  {monolithic}"
        )

    def __len__(self):
        return len(self.sample_files)

    def _get_sample_id(self, idx, sample_dict):
        entry = self.sample_files[idx]

        if isinstance(entry, Path):
            return entry.stem

        if isinstance(sample_dict, dict) and "sample_id" in sample_dict:
            return str(sample_dict["sample_id"])

        return f"{self.subset}_{idx}"

    def _check_required_keys(self, sample_dict: dict):
        # Check only the modalities requested by the active training config.
        # This keeps old four-modal experiments compatible while allowing
        # new five-modal experiments with dinov2_face.
        required_keys = list(self.feature_list) + ["ocean"]

        # egemaps_lld and opengraphau are standardized in __getitem__, so they
        # must exist whenever requested. Other requested modalities are loaded
        # directly from the cache.
        missing = [k for k in required_keys if k not in sample_dict]
        if missing:
            raise KeyError(
                f"[FiDataset] Missing keys in sample: {missing}. "
                f"feature_list={self.feature_list}. "
                f"Available keys: {list(sample_dict.keys())}"
            )

    def _read_sample(self, idx):
        entry = self.sample_files[idx]

        if isinstance(entry, Path):
            with open(entry, "rb") as f:
                sample_dict = pickle.load(f)
            return sample_dict

        return entry

    def _prepare_feature(self, feature_name: str, sample_dict: dict):
        """
        Prepare one requested modality and its temporal mask.

        Temporal lengths follow the original implementation:
        - egemaps_lld: 1500
        - opengraphau: 450
        - wav2vec2: 1500
        - wavlm: 1500
        - roberta: 80
        - deberta_v3: 80
        - dinov2_face: 450
        """
        if feature_name == "egemaps_lld":
            x_lld = standardization(
                torch.FloatTensor(sample_dict["egemaps_lld"]),
                mean=self.standardization_params["egemaps_lld"]["mean"],
                std=self.standardization_params["egemaps_lld"]["std"],
            )
            return pad_or_crop_time_dim(x_lld, 1500)

        if feature_name == "opengraphau":
            x_au = standardization(
                torch.FloatTensor(sample_dict["opengraphau"]),
                mean=self.standardization_params["opengraphau"]["mean"],
                std=self.standardization_params["opengraphau"]["std"],
            )
            return pad_or_crop_time_dim(x_au, 450)

        if feature_name == "wav2vec2":
            return pad_or_crop_time_dim(
                torch.FloatTensor(sample_dict["wav2vec2"]), 1500
            )

        if feature_name == "wavlm":
            return pad_or_crop_time_dim(
                torch.FloatTensor(sample_dict["wavlm"]), 1500
            )

        if feature_name == "roberta":
            return pad_or_crop_time_dim(
                torch.FloatTensor(sample_dict["roberta"]), 80
            )

        if feature_name == "deberta_v3":
            return pad_or_crop_time_dim(
                torch.FloatTensor(sample_dict["deberta_v3"]), 80
            )

        if feature_name == "dinov2_face":
            return pad_or_crop_time_dim(
                torch.FloatTensor(sample_dict["dinov2_face"]), 450
            )

        raise KeyError(
            f"[FiDataset] Unsupported feature_name={feature_name}. "
            "Supported: egemaps_lld, opengraphau, wav2vec2, wavlm, roberta, deberta_v3, dinov2_face."
        )

    def __getitem__(self, idx):
        sample_dict = self._read_sample(idx)
        self._check_required_keys(sample_dict)
        sample_id = self._get_sample_id(idx, sample_dict)

        y = np.asarray(sample_dict["ocean"], dtype=np.float32)
        if self.target_id is not None:
            y = np.expand_dims(y[self.target_id], -1).astype(np.float32)

        output = {
            "sample_id": sample_id,
            "app": y,
        }

        for feature_name in self.feature_list:
            feature_tensor, feature_mask = self._prepare_feature(feature_name, sample_dict)
            output[feature_name] = feature_tensor
            output[f"{feature_name}_mask"] = feature_mask

        return output

class FiDataModule(L.LightningDataModule):
    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = config if config is not None else {}
        self.batch_size = self.config.get("batch_size", 16)

        self.dataset_train = None
        self.dataset_valid = None
        self.dataset_test = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.dataset_train is None:
                print("[FI] Load train data...")
                self.dataset_train = FiDataset("train", self.config)

            if self.dataset_valid is None:
                print("[FI] Load valid data...")
                self.dataset_valid = FiDataset("valid", self.config)

        if stage == "test" or stage is None:
            if self.dataset_test is None:
                print("[FI] Load test data...")
                self.dataset_test = FiDataset("test", self.config)

    def train_dataloader(self):
        num_workers = self.config.get("num_workers", 0)
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    def val_dataloader(self):
        num_workers = self.config.get("num_workers", 0)
        return DataLoader(
            self.dataset_valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    def test_dataloader(self):
        if self.dataset_test is None:
            raise RuntimeError("[FiDataModule] dataset_test is None. Please call setup(stage='test') first.")

        num_workers = self.config.get("num_workers", 0)
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

if __name__ == "__main__":
    db_root = get_default_db_root()
    print("Calculating standardization under:", db_root)
    calculate_standardization(db_root)