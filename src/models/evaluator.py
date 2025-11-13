"""
Model evaluation utilities
Evaluate authentication model and GAN attack performance
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.stats import pearsonr

from src.utils.logger import get_logger

logger = get_logger()


class ModelEvaluator:
    """
    Evaluate authentication and GAN models
    """

    def __init__(self):
        """Initialize evaluator"""
        logger.info("Initialized ModelEvaluator")

    def evaluate_authentication(
        self,
        auth_model,
        X_legit: np.ndarray,
        X_impostor: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate authentication model performance

        Args:
            auth_model: Trained authentication model
            X_legit: Legitimate user samples
            X_impostor: Impostor samples

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating authentication model")

        # Use auth_model's built-in evaluate method
        metrics = auth_model.evaluate(X_legit, X_impostor)

        return metrics

    def evaluate_gan_attack(
        self,
        gan_model,
        auth_model,
        n_attempts: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluate GAN attack success rate

        Args:
            gan_model: Trained GAN model
            auth_model: Trained authentication model
            n_attempts: Number of attack attempts to simulate

        Returns:
            Dictionary with attack results
        """
        logger.info(f"Evaluating GAN attack with {n_attempts} attempts")

        # Generate fake samples
        fake_samples = gan_model.generate(n_attempts)

        # Test each fake sample against authentication
        similarity_scores = auth_model.predict(fake_samples)
        is_accepted = similarity_scores >= auth_model.threshold

        # Calculate metrics
        success_rate = np.mean(is_accepted)
        avg_similarity = np.mean(similarity_scores)
        max_similarity = np.max(similarity_scores)
        min_similarity = np.min(similarity_scores)

        # Find best attempts
        best_indices = np.argsort(similarity_scores)[-10:][::-1]
        best_attempts = [
            {
                'attempt_id': int(idx),
                'similarity': float(similarity_scores[idx]),
                'accepted': bool(is_accepted[idx])
            }
            for idx in best_indices
        ]

        results = {
            'n_attempts': n_attempts,
            'success_rate': float(success_rate),
            'n_successful': int(np.sum(is_accepted)),
            'avg_similarity': float(avg_similarity),
            'max_similarity': float(max_similarity),
            'min_similarity': float(min_similarity),
            'std_similarity': float(np.std(similarity_scores)),
            'attempts': [float(s) for s in similarity_scores],
            'best_attempts': best_attempts
        }

        logger.info(f"GAN attack evaluation: success_rate={success_rate:.2%}, avg_similarity={avg_similarity:.4f}")

        return results

    def compare_patterns(
        self,
        real_pattern: np.ndarray,
        fake_pattern: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare two keystroke patterns

        Args:
            real_pattern: Real keystroke feature vector
            fake_pattern: Fake keystroke feature vector

        Returns:
            Dictionary with comparison metrics
        """
        # Handle single patterns
        if len(real_pattern.shape) == 1:
            real_pattern = real_pattern.reshape(1, -1)
        if len(fake_pattern.shape) == 1:
            fake_pattern = fake_pattern.reshape(1, -1)

        # Use first sample if multiple provided
        real = real_pattern[0]
        fake = fake_pattern[0]

        # Calculate MSE
        mse = np.mean((real - fake) ** 2)

        # Calculate correlation
        try:
            correlation, p_value = pearsonr(real, fake)
        except:
            correlation = 0.0
            p_value = 1.0

        # Calculate cosine similarity
        dot_product = np.dot(real, fake)
        norm_real = np.linalg.norm(real)
        norm_fake = np.linalg.norm(fake)

        if norm_real > 0 and norm_fake > 0:
            cosine_similarity = dot_product / (norm_real * norm_fake)
        else:
            cosine_similarity = 0.0

        # Calculate absolute differences
        abs_diff = np.abs(real - fake)
        mean_abs_diff = np.mean(abs_diff)
        max_abs_diff = np.max(abs_diff)

        # Overall similarity score (0-1)
        # Combine multiple metrics
        similarity_score = (
            0.4 * (1 - mse / (mse + 1)) +  # MSE component
            0.3 * (correlation + 1) / 2 +   # Correlation component
            0.3 * (cosine_similarity + 1) / 2  # Cosine component
        )

        results = {
            'mse': float(mse),
            'correlation': float(correlation),
            'correlation_p_value': float(p_value),
            'cosine_similarity': float(cosine_similarity),
            'mean_abs_diff': float(mean_abs_diff),
            'max_abs_diff': float(max_abs_diff),
            'similarity_score': float(similarity_score)
        }

        logger.info(f"Pattern comparison: MSE={mse:.4f}, Correlation={correlation:.4f}, Similarity={similarity_score:.4f}")

        return results

    def analyze_feature_distribution(
        self,
        real_features: np.ndarray,
        fake_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze and compare feature distributions

        Args:
            real_features: Real keystroke features (n_samples, feature_dim)
            fake_features: Fake keystroke features (n_samples, feature_dim)

        Returns:
            Dictionary with distribution analysis
        """
        logger.info("Analyzing feature distributions")

        n_features = real_features.shape[1]

        feature_analysis = []

        for i in range(n_features):
            real_feat = real_features[:, i]
            fake_feat = fake_features[:, i]

            analysis = {
                'feature_idx': i,
                'real_mean': float(np.mean(real_feat)),
                'real_std': float(np.std(real_feat)),
                'real_min': float(np.min(real_feat)),
                'real_max': float(np.max(real_feat)),
                'fake_mean': float(np.mean(fake_feat)),
                'fake_std': float(np.std(fake_feat)),
                'fake_min': float(np.min(fake_feat)),
                'fake_max': float(np.max(fake_feat)),
                'mean_diff': float(abs(np.mean(real_feat) - np.mean(fake_feat))),
                'std_diff': float(abs(np.std(real_feat) - np.std(fake_feat)))
            }

            feature_analysis.append(analysis)

        # Overall distribution similarity
        mean_diffs = [f['mean_diff'] for f in feature_analysis]
        std_diffs = [f['std_diff'] for f in feature_analysis]

        results = {
            'n_features': n_features,
            'feature_analysis': feature_analysis,
            'avg_mean_diff': float(np.mean(mean_diffs)),
            'avg_std_diff': float(np.mean(std_diffs)),
            'max_mean_diff': float(np.max(mean_diffs)),
            'max_std_diff': float(np.max(std_diffs))
        }

        logger.info(f"Distribution analysis complete: avg_mean_diff={results['avg_mean_diff']:.4f}")

        return results

    def generate_attack_report(
        self,
        gan_model,
        auth_model,
        n_attempts: int = 100
    ) -> Dict[str, Any]:
        """
        Generate comprehensive attack report

        Args:
            gan_model: Trained GAN model
            auth_model: Trained authentication model
            n_attempts: Number of attack attempts

        Returns:
            Comprehensive attack report
        """
        logger.info("Generating attack report")

        # Evaluate attack
        attack_results = self.evaluate_gan_attack(gan_model, auth_model, n_attempts)

        # Generate samples for analysis
        fake_samples = gan_model.generate(min(50, n_attempts))

        # Create report
        report = {
            'attack_summary': {
                'total_attempts': attack_results['n_attempts'],
                'successful_attacks': attack_results['n_successful'],
                'success_rate': attack_results['success_rate'],
                'avg_similarity': attack_results['avg_similarity'],
                'max_similarity': attack_results['max_similarity']
            },
            'best_attempts': attack_results['best_attempts'],
            'similarity_distribution': {
                'mean': attack_results['avg_similarity'],
                'std': attack_results['std_similarity'],
                'min': attack_results['min_similarity'],
                'max': attack_results['max_similarity']
            },
            'all_attempts': attack_results['attempts']
        }

        logger.info("Attack report generated")

        return report

    def calculate_eer(
        self,
        legit_scores: np.ndarray,
        impostor_scores: np.ndarray,
        n_thresholds: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate Equal Error Rate (EER)

        Args:
            legit_scores: Similarity scores for legitimate users
            impostor_scores: Similarity scores for impostors
            n_thresholds: Number of thresholds to test

        Returns:
            Tuple of (EER, threshold_at_EER)
        """
        thresholds = np.linspace(0, 1, n_thresholds)

        far_list = []  # False Accept Rate
        frr_list = []  # False Reject Rate

        for thresh in thresholds:
            far = np.mean(impostor_scores >= thresh)
            frr = np.mean(legit_scores < thresh)
            far_list.append(far)
            frr_list.append(frr)

        # Find EER (where FAR ≈ FRR)
        far_array = np.array(far_list)
        frr_array = np.array(frr_list)
        eer_index = np.argmin(np.abs(far_array - frr_array))

        eer = (far_array[eer_index] + frr_array[eer_index]) / 2
        eer_threshold = thresholds[eer_index]

        logger.info(f"EER calculated: {eer:.4f} at threshold {eer_threshold:.4f}")

        return float(eer), float(eer_threshold)
