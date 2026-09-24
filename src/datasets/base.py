"""Common dataset interface — plug in requested cattle-pain videos later with zero pipeline changes."""
from typing import Iterator, Tuple
import numpy as np


class BasePainVideoDataset:
    def frames(self, video_id: str) -> Iterator[Tuple[np.ndarray, float]]:
        raise NotImplementedError

    def identity_id(self, video_id: str) -> str:
        raise NotImplementedError

    def weak_label(self, video_id: str) -> int:
        raise NotImplementedError

    def fps(self, video_id: str) -> float:
        raise NotImplementedError
