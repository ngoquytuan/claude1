"""
Helper utilities for Keystroke GAN Demo
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist

    Args:
        directory: Directory path to create
    """
    os.makedirs(directory, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to JSON file

    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file

    Args:
        filepath: Input file path

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_numpy(data: np.ndarray, filepath: str) -> None:
    """
    Save numpy array to file

    Args:
        data: Numpy array to save
        filepath: Output file path
    """
    ensure_dir(os.path.dirname(filepath))
    np.save(filepath, data)


def load_numpy(filepath: str) -> np.ndarray:
    """
    Load numpy array from file

    Args:
        filepath: Input file path

    Returns:
        Loaded numpy array
    """
    return np.load(filepath)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function

    Args:
        x: Input array

    Returns:
        Sigmoid output
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def normalize_scores(scores: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    Normalize scores to [min_val, max_val] range

    Args:
        scores: Input scores
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Normalized scores
    """
    scores_min = scores.min()
    scores_max = scores.max()

    if scores_max == scores_min:
        return np.full_like(scores, (min_val + max_val) / 2)

    normalized = (scores - scores_min) / (scores_max - scores_min)
    return normalized * (max_val - min_val) + min_val


def timestamp_to_string(timestamp: float = None) -> str:
    """
    Convert timestamp to readable string

    Args:
        timestamp: Unix timestamp (if None, use current time)

    Returns:
        Formatted datetime string
    """
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistics for data

    Args:
        data: Input array

    Returns:
        Dictionary with statistics
    """
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data))
    }


def print_progress_bar(iteration: int, total: int, prefix: str = '',
                       suffix: str = '', length: int = 50, fill: str = '█') -> None:
    """
    Print progress bar to console

    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Bar length
        fill: Fill character
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)

    if iteration == total:
        print()


def validate_model_path(model_path: str) -> bool:
    """
    Validate if model file exists

    Args:
        model_path: Path to model file

    Returns:
        True if model exists, False otherwise
    """
    return os.path.exists(model_path) and os.path.isfile(model_path)
