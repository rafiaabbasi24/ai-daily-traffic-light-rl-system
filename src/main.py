"""src/main.py
Entry point for the Dynamic Traffic Light Control with Multi‑Agent RL project.

This script parses command‑line arguments, loads a configuration file (YAML
format), sets up reproducibility seeds, configures logging, selects the
computation device and launches the training loop via :class:`Trainer`.

Typical usage
-------------
```bash
python -m src.main --config configs/default.yaml --episodes 5000
```
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

# The Trainer class encapsulates the RL training logic.
# It is expected to be defined in ``src/trainer.py``.
try:
    from src.trainer import Trainer
except ImportError as exc:
    raise ImportError(
        "Could not import `Trainer`. Ensure that `src/trainer.py` exists and defines "
        "`Trainer`."
    ) from exc


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with defaults applied.
    """
    parser = argparse.ArgumentParser(
        description="Dynamic Traffic Light Control with Multi‑Agent RL"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML configuration file. Overrides defaults.",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes (overrides config).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config).",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Computation device (overrides config).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory where logs and checkpoints will be saved.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO logging; only warnings and errors are shown.",
    )
    return parser.parse_args()


def load_config(config_path: Path | None) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : Path | None
        Path to the configuration file. If ``None``, an empty config is returned.

    Returns
    -------
    dict
        Configuration dictionary (may be empty).
    """
    if config_path is None:
        return {}
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must contain a YAML mapping at top level.")
    return cfg


def merge_args_into_config(
    args: argparse.Namespace, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Overlay command‑line arguments onto the loaded configuration.

    Non‑``None`` arguments replace corresponding keys in the config. Keys are
    expected to match the top‑level entries of the config file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command‑line arguments.
    config : dict
        Configuration loaded from YAML.

    Returns
    -------
    dict
        Updated configuration dictionary.
    """
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None and key != "quiet":
            # Convert argparse attribute names to config keys (e.g., "log_dir" -> "log_dir")
            config[key] = value
    return config


def set_random_seeds(seed: int) -> None:
    """Set seeds for ``random``, ``numpy`` and ``torch`` for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Ensure deterministic behavior on CUDA (may affect performance)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_logging(log_dir: Path | None, quiet: bool) -> None:
    """Configure the root logger.

    Logs are printed to stdout and, if ``log_dir`` is provided, also written to
    ``training.log`` inside that directory.

    Parameters
    ----------
    log_dir : Path | None
        Directory for log files. Created if it does not exist.
    quiet : bool
        If ``True``, set logging level to ``WARNING``; otherwise ``INFO``.
    """
    level = logging.WARNING if quiet else logging.INFO
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=log_format, handlers=handlers)


def select_device(preferred: str | None) -> torch.device:
    """Select the computation device.

    Parameters
    ----------
    preferred : str | None
        User‑specified device ("cpu" or "cuda"). If ``None``, automatically
        selects CUDA when available.

    Returns
    -------
    torch.device
        The selected device.
    """
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    # Automatic selection
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """Main entry point: parse arguments, prepare environment, and start training."""
    args = parse_args()

    # Load and merge configuration
    config = load_config(args.config)
    config = merge_args_into_config(args, config)

    # Resolve essential configuration values with sensible defaults
    seed: int = config.get("seed", 42)
    episodes: int = config.get("episodes", 1000)
    device_str: str | None = config.get("device")
    log_dir: Path | None = config.get("log_dir")
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)

    # Configure logging early so that subsequent modules can log
    configure_logging(log_dir, args.quiet)
    logger = logging.getLogger(__name__)
    logger.info("Starting Dynamic Traffic Light Control training")
    logger.debug("Merged configuration: %s", config)

    # Set reproducibility seeds
    set_random_seeds(seed)
    logger.info("Random seed set to %d", seed)

    # Choose computation device
    device = select_device(device_str)
    logger.info("Using device: %s", device)

    # Instantiate and run the trainer
    trainer = Trainer(
        config=config,
        device=device,
        log_dir=log_dir,
        total_episodes=episodes,
    )
    trainer.train()

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
