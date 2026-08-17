import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

from exordium.video.io import batch_iterator
from exordium.video.detection import Track
from exordium.utils.decorator import load_or_create


class DinoV2FaceWrapper:
    """DINOv2 face feature wrapper.

    This wrapper intentionally follows the same interface as FabNetWrapper:

        ids, features = wrapper.track_to_feature(track, batch_size=30, output_path=...)

    Input faces are expected to be numpy images with shape [H, W, C] in BGR order,
    because detection crops in the existing FI visual pipeline are OpenCV-style BGR images.

    Default model:
        dinov2_vits14, output feature dimension = 384

    Saved format:
        pickle tuple (ids, features)
        ids:      list[int]
        features: np.ndarray [T, 384]
    """

    def __init__(
        self,
        gpu_id: int = 0,
        model_name: str = "dinov2_vits14",
        image_size: int = 224,
        use_torch_hub: bool = True,
        local_repo: str | None = None,
    ):
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available() else "cpu"
        self.model_name = str(model_name)
        self.image_size = int(image_size)

        if use_torch_hub:
            # First run may need internet access to download the DINOv2 repository/weights.
            # After that, torch hub normally reuses the local cache.
            if local_repo is None:
                self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
            else:
                # Optional offline/local use:
                # local_repo should point to a local clone of facebookresearch/dinov2.
                self.model = torch.hub.load(local_repo, self.model_name, source="local")
        else:
            raise ValueError("Only torch.hub loading is implemented in this wrapper.")

        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.image_size, self.image_size), antialias=True),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        logging.info(f"DINOv2 face model {self.model_name} is loaded to {self.device}.")

    def _preprocess_bgr_faces(self, faces: Sequence[np.ndarray]) -> torch.Tensor:
        tensors = []
        for face in faces:
            if face is None:
                continue

            if not isinstance(face, np.ndarray):
                face = np.asarray(face)

            if face.ndim != 3 or face.shape[2] != 3:
                raise ValueError(f"Expected BGR face image [H, W, 3], got {face.shape}")

            # Existing FI crops are OpenCV BGR. DINOv2/ImageNet normalization expects RGB.
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(face_rgb))

        if len(tensors) == 0:
            raise ValueError("No valid face crops were provided to DINOv2.")

        return torch.stack(tensors, dim=0).to(self.device)

    def __call__(self, faces: Sequence[np.ndarray]) -> np.ndarray:
        """DINOv2 inference.

        Args:
            faces: list[np.ndarray], each image has shape [H, W, C] in BGR order.

        Returns:
            np.ndarray with shape [B, D].
            For dinov2_vits14, D = 384.
        """
        samples = self._preprocess_bgr_faces(faces)

        if samples.ndim != 4:
            raise ValueError(
                f"Invalid input shape. Expected [B, C, H, W], got {tuple(samples.shape)}."
            )

        with torch.no_grad():
            features = self.model(samples)

        # torch.hub DINOv2 usually returns a tensor [B, D].
        # Keep a defensive branch in case a dict-like output is used by another model.
        if isinstance(features, dict):
            if "x_norm_clstoken" in features:
                features = features["x_norm_clstoken"]
            elif "cls_token" in features:
                features = features["cls_token"]
            else:
                raise RuntimeError(f"Unsupported DINOv2 output keys: {list(features.keys())}")

        if features.ndim > 2:
            features = features.reshape(features.size(0), -1)

        features = features.detach().float().cpu().numpy()

        if features.ndim != 2 or features.shape[0] != samples.shape[0]:
            raise ValueError(
                f"Invalid DINOv2 feature shape. Expected [B, D], got {features.shape}."
            )

        return features

    @load_or_create("pkl")
    def dir_to_feature(
        self,
        img_paths: list[str],
        batch_size: int = 30,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[list, np.ndarray]:
        """Extract DINOv2 features from image files.

        This method is optional, but mirrors FabNetWrapper.dir_to_feature.
        """
        ids, features = [], []

        for index in tqdm(
            range(0, len(img_paths), batch_size),
            total=np.ceil(len(img_paths) / batch_size).astype(int),
            desc="DINOv2 face extraction",
            disable=not verbose,
        ):
            batch_paths = img_paths[index:index + batch_size]
            ids += [int(Path(p).stem) for p in batch_paths]

            samples = []
            for image_path in batch_paths:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                samples.append(image)

            if len(samples) == 0:
                continue

            feature = self(samples)
            features.append(feature)

        if len(features) == 0:
            raise RuntimeError("No DINOv2 features were extracted from image paths.")

        features = np.concatenate(features, axis=0)
        return ids, features

    @load_or_create("pkl")
    def track_to_feature(
        self,
        track: Track,
        batch_size: int = 30,
        **kwargs,
    ) -> tuple[list, np.ndarray]:
        """Extract DINOv2 features from an exordium Track.

        This method intentionally copies the working logic from FabNetWrapper:
            - iterate over track using batch_iterator
            - skip interpolated detections
            - use detection.bb_crop_wide()
            - return (ids, features)

        Args:
            track: exordium.video.detection.Track
            batch_size: number of face crops per DINOv2 forward pass

        Returns:
            ids: list of frame ids
            features: np.ndarray [T, D]
        """
        ids, features = [], []

        for subset in batch_iterator(track, batch_size):
            valid_detections = [
                detection for detection in subset
                if not detection.is_interpolated
            ]

            if len(valid_detections) == 0:
                continue

            ids += [detection.frame_id for detection in valid_detections]
            samples = [detection.bb_crop_wide() for detection in valid_detections]

            if len(samples) == 0:
                continue

            feature = self(samples)
            features.append(feature)

        if len(features) == 0:
            raise RuntimeError("No DINOv2 features were extracted from the track.")

        features = np.concatenate(features, axis=0)
        return ids, features
