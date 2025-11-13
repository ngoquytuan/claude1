"""
Authentication model using One-Class SVM
Learns the keystroke pattern of a legitimate user
"""

import numpy as np
import pickle
from typing import Tuple, Dict, Any
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from config.settings import (
    AUTH_THRESHOLD, SVM_KERNEL, SVM_NU, SVM_GAMMA, MODEL_PATH
)
from src.utils.helpers import ensure_dir, sigmoid, normalize_scores
from src.utils.logger import get_logger

logger = get_logger()


class AuthenticationModel:
    """
    One-Class SVM model for keystroke authentication
    """

    def __init__(self, threshold: float = AUTH_THRESHOLD):
        """
        Initialize authentication model

        Args:
            threshold: Similarity threshold for authentication (0-1)
        """
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.decision_threshold = None
        self.is_trained = False

        logger.info(f"Initialized AuthenticationModel with threshold={threshold}")

    def train(self, X_train: np.ndarray) -> Dict[str, Any]:
        """
        Train the authentication model on legitimate user data

        Args:
            X_train: Training features (n_samples, feature_dim)

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training authentication model on {len(X_train)} samples")

        # Normalize features
        X_normalized = self.scaler.fit_transform(X_train)
        logger.info("Features normalized")

        # Train One-Class SVM
        self.model = OneClassSVM(
            kernel=SVM_KERNEL,
            nu=SVM_NU,
            gamma=SVM_GAMMA
        )
        self.model.fit(X_normalized)
        logger.info("One-Class SVM trained")

        # Calculate decision threshold based on training data
        train_scores = self.model.decision_function(X_normalized)

        # Set threshold at 15th percentile (85% acceptance on training data)
        percentile = (1 - self.threshold) * 100
        self.decision_threshold = np.percentile(train_scores, percentile)

        logger.info(f"Decision threshold set to {self.decision_threshold:.4f}")

        # Calculate training metrics
        predictions = train_scores >= self.decision_threshold
        train_accuracy = np.mean(predictions)

        self.is_trained = True

        metrics = {
            'train_accuracy': float(train_accuracy),
            'decision_threshold': float(self.decision_threshold),
            'train_score_mean': float(np.mean(train_scores)),
            'train_score_std': float(np.std(train_scores)),
            'train_score_min': float(np.min(train_scores)),
            'train_score_max': float(np.max(train_scores))
        }

        logger.info(f"Training complete: accuracy={train_accuracy:.4f}")
        return metrics

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict similarity scores for test samples

        Args:
            X_test: Test features (n_samples, feature_dim)

        Returns:
            Similarity scores (0-1) for each sample
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")

        # Handle single sample
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(1, -1)

        # Normalize features
        X_normalized = self.scaler.transform(X_test)

        # Get decision scores
        decision_scores = self.model.decision_function(X_normalized)

        # Convert to similarity scores (0-1)
        # Higher decision score = more similar to legitimate user
        # Use sigmoid transformation centered at decision_threshold
        similarity_scores = sigmoid((decision_scores - self.decision_threshold) / abs(self.decision_threshold))

        # Ensure scores are in [0, 1]
        similarity_scores = np.clip(similarity_scores, 0, 1)

        return similarity_scores

    def is_authentic(self, X_test: np.ndarray) -> Tuple[bool, float]:
        """
        Check if test sample is from legitimate user

        Args:
            X_test: Test features (can be single sample or batch)

        Returns:
            Tuple of (is_authentic, similarity_score)
        """
        similarity_scores = self.predict(X_test)

        # If single sample
        if len(similarity_scores) == 1:
            score = float(similarity_scores[0])
            is_auth = score >= self.threshold
            return is_auth, score

        # If batch
        return similarity_scores >= self.threshold, similarity_scores

    def evaluate(self, X_legit: np.ndarray, X_impostor: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            X_legit: Legitimate user samples
            X_impostor: Impostor samples

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model: {len(X_legit)} legit, {len(X_impostor)} impostor samples")

        # Predict on legitimate samples
        legit_scores = self.predict(X_legit)
        legit_accepted = legit_scores >= self.threshold

        # Predict on impostor samples
        impostor_scores = self.predict(X_impostor)
        impostor_rejected = impostor_scores < self.threshold

        # Calculate metrics
        true_accept_rate = np.mean(legit_accepted)  # Correctly accept legitimate
        false_reject_rate = 1 - true_accept_rate    # Incorrectly reject legitimate

        true_reject_rate = np.mean(impostor_rejected)  # Correctly reject impostor
        false_accept_rate = 1 - true_reject_rate       # Incorrectly accept impostor

        # Accuracy
        n_correct = np.sum(legit_accepted) + np.sum(impostor_rejected)
        n_total = len(X_legit) + len(X_impostor)
        accuracy = n_correct / n_total

        # Equal Error Rate (EER) - point where FAR = FRR
        # Approximate by finding threshold where they are closest
        all_scores = np.concatenate([legit_scores, impostor_scores])
        all_labels = np.concatenate([np.ones(len(legit_scores)), np.zeros(len(impostor_scores))])

        thresholds = np.linspace(0, 1, 100)
        far_list = []
        frr_list = []

        for thresh in thresholds:
            far = np.mean(impostor_scores >= thresh)
            frr = np.mean(legit_scores < thresh)
            far_list.append(far)
            frr_list.append(frr)

        # Find EER
        far_array = np.array(far_list)
        frr_array = np.array(frr_list)
        eer_index = np.argmin(np.abs(far_array - frr_array))
        eer = (far_array[eer_index] + frr_array[eer_index]) / 2

        metrics = {
            'accuracy': float(accuracy),
            'true_accept_rate': float(true_accept_rate),
            'false_reject_rate': float(false_reject_rate),
            'true_reject_rate': float(true_reject_rate),
            'false_accept_rate': float(false_accept_rate),
            'eer': float(eer),
            'avg_legit_score': float(np.mean(legit_scores)),
            'avg_impostor_score': float(np.mean(impostor_scores)),
            'score_separation': float(np.mean(legit_scores) - np.mean(impostor_scores))
        }

        logger.info(f"Evaluation complete: accuracy={accuracy:.4f}, FAR={false_accept_rate:.4f}, FRR={false_reject_rate:.4f}, EER={eer:.4f}")
        return metrics

    def save_model(self, filepath: str = None) -> str:
        """
        Save model to disk

        Args:
            filepath: Path to save model (default: MODEL_PATH/auth_model.pkl)

        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        if filepath is None:
            ensure_dir(MODEL_PATH)
            filepath = f"{MODEL_PATH}auth_model.pkl"

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'decision_threshold': self.decision_threshold,
            'threshold': self.threshold,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filepath: str = None) -> None:
        """
        Load model from disk

        Args:
            filepath: Path to model file (default: MODEL_PATH/auth_model.pkl)
        """
        if filepath is None:
            filepath = f"{MODEL_PATH}auth_model.pkl"

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.decision_threshold = model_data['decision_threshold']
        self.threshold = model_data['threshold']
        self.is_trained = model_data['is_trained']

        logger.info(f"Model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns:
            Dictionary with model info
        """
        if not self.is_trained:
            return {'is_trained': False}

        return {
            'is_trained': True,
            'threshold': self.threshold,
            'decision_threshold': self.decision_threshold,
            'svm_kernel': SVM_KERNEL,
            'svm_nu': SVM_NU,
            'svm_gamma': SVM_GAMMA
        }
