from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from omegaconf import OmegaConf


@dataclass
class Config:
    # Noise Schedule
    t: int
    num_inference_steps: int
    schedule_type: str
    schedule_params: Optional[Dict]
    cfg_config: Optional[Dict]
    data_scale: float

    # Training
    max_iters: int
    log_every: int
    batch_size: int
    device: str
    amp: bool

    # Dataset
    dataset_name: str
    num_workers: int

    # Model Architecture
    model_type: str
    model_kwargs: Dict

    # Optimizer Parameters
    learning_rate: float
    weight_decay: float

    # Training Vis
    train_vis_every: int
    inference_vis_every: int

    # Inference Viz
    inference_batch_size: int
    inference_grid_shape: Tuple[int, int]
    inference_vis_dir: Path

    # Loss
    pred_type: str

    # Checkpoint Dumping
    save_checkpoint_every: int
    checkpoint_dir: Path

    # Other
    profile: bool = False
    compile: bool = True


def load_config_from_yaml(path: Path, config_overrides: List[str]) -> Config:
    config = OmegaConf.structured(Config)
    return OmegaConf.merge(config, OmegaConf.load(path), OmegaConf.from_dotlist(config_overrides))
