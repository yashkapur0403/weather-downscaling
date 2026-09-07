"""
utils/model_loader.py
----------------------
Loads the trained residual U-Net (models/best_model.pt) on a background
thread so the FastAPI process can start accepting requests immediately
instead of blocking on a (possibly slow, disk/GPU-bound) checkpoint load.

Usage (see main.py):

    loader = ModelLoader.get_instance()
    loader.load_async("models/best_model.pt")   # returns immediately

    ... later, inside a request handler ...
    loader.wait_until_ready(timeout=30)          # blocks only until loaded
    if loader.model is not None:
        prediction = loader.model(input_tensor)
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Default channel set if the checkpoint doesn't record one itself.
DEFAULT_CHANNELS = ["imd_rain", "dem", "era5_t2m", "era5_t2m_max", "era5_dewp"]


class ModelLoader:
    """Thread-safe singleton that owns the loaded model + its metadata."""

    _instance: Optional["ModelLoader"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.channels: list[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ready = threading.Event()
        self.load_error: Optional[str] = None
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def load_async(self, checkpoint_path: str = "models/best_model.pt") -> threading.Thread:
        """Kick off loading on a daemon thread and return immediately."""
        thread = threading.Thread(
            target=self._load, args=(checkpoint_path,), daemon=True, name="model-loader"
        )
        thread.start()
        return thread

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Block the caller (e.g. a request handler) until loading finishes.

        Returns True if the load completed (success OR failure) within the
        timeout, False if it's still in progress. Check `self.model is None`
        afterwards to distinguish "not ready yet" from "failed to load".
        """
        return self.ready.wait(timeout=timeout)

    # ------------------------------------------------------------------
    def _load(self, checkpoint_path: str) -> None:
        with self._load_lock:
            try:
                from .unet_model import ResidualUNet

                path = Path(checkpoint_path)
                if not path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint not found at {path}. Train the model with the "
                        f"original repo's scripts/train.py + ablation.py, or point "
                        f"model_loader at the right path."
                    )

                ckpt = torch.load(path, map_location=self.device)
                channels = ckpt.get("channels", DEFAULT_CHANNELS)
                state_dict = ckpt.get("model_state_dict", ckpt)

                model = ResidualUNet(in_channels=len(channels))
                model.load_state_dict(state_dict)
                model.eval()
                model.to(self.device)

                self.model = model
                self.channels = channels
                logger.info(
                    "Model loaded from %s on %s with channels=%s", path, self.device, channels
                )
            except Exception as exc:  # noqa: BLE001 - we want to capture *any* load failure
                self.load_error = str(exc)
                self.model = None
                logger.warning("Model failed to load (%s). API will run in degraded mode "
                                "-- predictions will fall back to cached Layer-2 data only.",
                                exc)
            finally:
                self.ready.set()
