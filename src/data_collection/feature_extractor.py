"""
Feature extraction from keystroke data
Extracts timing features: dwell time, flight time, digraph latency, and statistical features
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler
from src.utils.helpers import save_json, load_json, save_numpy, load_numpy
from src.utils.logger import get_logger

logger = get_logger()


class FeatureExtractor:
    """
    Extract timing features from raw keystroke data
    """

    def __init__(self):
        """Initialize feature extractor"""
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False

    def extract_from_sample(self, sample: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a single keystroke sample

        Args:
            sample: Dictionary containing keystroke data with format:
                {
                    "sample_id": int,
                    "timestamp": str,
                    "keypresses": [
                        {
                            "char": str,
                            "position": int,
                            "keydown_time": float,
                            "keyup_time": float
                        },
                        ...
                    ]
                }

        Returns:
            Feature vector as numpy array
        """
        keypresses = sample['keypresses']
        features = []

        # Sort by position to ensure correct order
        keypresses = sorted(keypresses, key=lambda x: x['position'])

        # Extract dwell times (hold time for each key)
        dwell_times = []
        for kp in keypresses:
            dwell = kp['keyup_time'] - kp['keydown_time']
            dwell_times.append(dwell)
            features.append(dwell)

        # Extract flight times (time between release and next press)
        flight_times = []
        for i in range(len(keypresses) - 1):
            flight = keypresses[i + 1]['keydown_time'] - keypresses[i]['keyup_time']
            flight_times.append(flight)
            features.append(flight)

        # Extract digraph latency (time between consecutive key presses)
        digraph_latencies = []
        for i in range(len(keypresses) - 1):
            digraph = keypresses[i + 1]['keydown_time'] - keypresses[i]['keydown_time']
            digraph_latencies.append(digraph)
            features.append(digraph)

        # Extract statistical features
        total_time = keypresses[-1]['keyup_time'] - keypresses[0]['keydown_time']
        avg_dwell = np.mean(dwell_times) if dwell_times else 0.0
        avg_flight = np.mean(flight_times) if flight_times else 0.0
        typing_speed = len(keypresses) / total_time if total_time > 0 else 0.0

        features.extend([total_time, avg_dwell, avg_flight, typing_speed])

        return np.array(features, dtype=np.float32)

    def extract_from_file(self, json_path: str) -> np.ndarray:
        """
        Extract features from all samples in a JSON file

        Args:
            json_path: Path to JSON file with keystroke samples

        Returns:
            Feature matrix with shape (n_samples, feature_dim)
        """
        logger.info(f"Loading keystroke data from {json_path}")
        data = load_json(json_path)

        samples = data['samples']
        logger.info(f"Extracting features from {len(samples)} samples")

        features_list = []
        for i, sample in enumerate(samples):
            try:
                feature_vector = self.extract_from_sample(sample)
                features_list.append(feature_vector)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(samples)} samples")

            except Exception as e:
                logger.warning(f"Failed to extract features from sample {sample.get('sample_id', i)}: {e}")
                continue

        features = np.array(features_list, dtype=np.float32)
        logger.info(f"Extracted features shape: {features.shape}")

        return features

    def normalize_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize features using StandardScaler

        Args:
            features: Feature matrix (n_samples, feature_dim)
            fit: Whether to fit the scaler (True for training data, False for test data)

        Returns:
            Normalized feature matrix
        """
        if fit:
            normalized = self.scaler.fit_transform(features)
            self.is_fitted = True
            logger.info("Fitted scaler on features")
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted yet. Call with fit=True first.")
            normalized = self.scaler.transform(features)

        return normalized

    def save_processed(self, features: np.ndarray, labels: np.ndarray,
                       output_path: str, metadata: Dict[str, Any] = None) -> None:
        """
        Save processed features to disk

        Args:
            features: Feature matrix
            labels: Label array
            output_path: Base output path (without extension)
            metadata: Optional metadata to save
        """
        # Save features
        features_path = f"{output_path}_features.npy"
        save_numpy(features, features_path)
        logger.info(f"Saved features to {features_path}")

        # Save labels
        labels_path = f"{output_path}_labels.npy"
        save_numpy(labels, labels_path)
        logger.info(f"Saved labels to {labels_path}")

        # Save metadata
        if metadata is None:
            metadata = {}

        metadata.update({
            'feature_shape': features.shape,
            'label_shape': labels.shape,
            'n_samples': len(features),
            'feature_dim': features.shape[1] if len(features.shape) > 1 else 0
        })

        metadata_path = f"{output_path}_metadata.json"
        save_json(metadata, metadata_path)
        logger.info(f"Saved metadata to {metadata_path}")

    def generate_feature_names(self, target_text: str) -> List[str]:
        """
        Generate feature names based on target text

        Args:
            target_text: The text being typed

        Returns:
            List of feature names
        """
        feature_names = []

        # Dwell time features
        for i, char in enumerate(target_text):
            feature_names.append(f"dwell_time_{i}_{char}")

        # Flight time features
        for i in range(len(target_text) - 1):
            feature_names.append(f"flight_time_{i}_{target_text[i]}_to_{target_text[i+1]}")

        # Digraph latency features
        for i in range(len(target_text) - 1):
            feature_names.append(f"digraph_latency_{i}_{target_text[i]}{target_text[i+1]}")

        # Statistical features
        feature_names.extend(['total_time', 'avg_dwell', 'avg_flight', 'typing_speed'])

        self.feature_names = feature_names
        return feature_names

    def get_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance based on variance

        Args:
            features: Feature matrix

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(features.shape[1])]

        variances = np.var(features, axis=0)
        total_variance = np.sum(variances)

        importance = {}
        for name, var in zip(self.feature_names, variances):
            importance[name] = float(var / total_variance) if total_variance > 0 else 0.0

        return importance
