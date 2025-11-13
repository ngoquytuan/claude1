"""
Demo scenes logic
Implements the 4 main demo scenarios
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple

from src.data_collection.feature_extractor import FeatureExtractor
from src.utils.logger import get_logger

logger = get_logger()


class DemoScenes:
    """
    Implement logic for demo scenes
    """

    def __init__(self, auth_model, gan_model):
        """
        Initialize demo scenes

        Args:
            auth_model: Trained authentication model
            gan_model: Trained GAN model
        """
        self.auth_model = auth_model
        self.gan_model = gan_model
        self.feature_extractor = FeatureExtractor()

        logger.info("Initialized DemoScenes")

    def scene_1_real_user(
        self,
        keystroke_sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Scene 1: Legitimate user login

        Args:
            keystroke_sample: Keystroke sample from real user

        Returns:
            Result dictionary
        """
        logger.info("Running Scene 1: Legitimate User Login")

        # Extract features
        features = self.feature_extractor.extract_from_sample(keystroke_sample)

        # Authenticate
        is_auth, similarity = self.auth_model.is_authentic(features)

        result = {
            'scene': 1,
            'scene_name': 'Legitimate User Login',
            'status': 'success' if is_auth else 'failure',
            'is_authenticated': bool(is_auth),
            'similarity_score': float(similarity),
            'threshold': self.auth_model.threshold,
            'message': self._format_scene1_message(is_auth, similarity),
            'features': features
        }

        logger.info(f"Scene 1 result: {result['status']}, similarity={similarity:.4f}")
        return result

    def scene_2_impostor(
        self,
        keystroke_sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Scene 2: Impostor login attempt

        Args:
            keystroke_sample: Keystroke sample from impostor

        Returns:
            Result dictionary
        """
        logger.info("Running Scene 2: Impostor Login Attempt")

        # Extract features
        features = self.feature_extractor.extract_from_sample(keystroke_sample)

        # Authenticate
        is_auth, similarity = self.auth_model.is_authentic(features)

        result = {
            'scene': 2,
            'scene_name': 'Impostor Login Attempt',
            'status': 'failure' if not is_auth else 'unexpected_success',
            'is_authenticated': bool(is_auth),
            'similarity_score': float(similarity),
            'threshold': self.auth_model.threshold,
            'message': self._format_scene2_message(is_auth, similarity),
            'features': features
        }

        logger.info(f"Scene 2 result: {result['status']}, similarity={similarity:.4f}")
        return result

    def scene_3_gan_attack(
        self,
        n_attempts: int = 10,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Scene 3: GAN attack simulation

        Args:
            n_attempts: Number of attack attempts
            progress_callback: Optional callback function for progress updates

        Returns:
            Attack results dictionary
        """
        logger.info(f"Running Scene 3: GAN Attack with {n_attempts} attempts")

        attack_results = []
        successful_attacks = []

        for i in range(n_attempts):
            # Generate fake sample
            fake_features = self.gan_model.generate(1)[0]

            # Test authentication
            is_auth, similarity = self.auth_model.is_authentic(fake_features)

            attempt_result = {
                'attempt_id': i + 1,
                'similarity_score': float(similarity),
                'is_authenticated': bool(is_auth),
                'status': 'success' if is_auth else 'failure'
            }

            attack_results.append(attempt_result)

            if is_auth:
                successful_attacks.append(attempt_result)
                logger.info(f"Attempt {i+1}: BYPASS SUCCESS! Similarity={similarity:.4f}")

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, n_attempts, similarity, is_auth)

            # Small delay for visual effect
            time.sleep(0.1)

        # Calculate statistics
        success_rate = len(successful_attacks) / n_attempts
        similarities = [r['similarity_score'] for r in attack_results]

        result = {
            'scene': 3,
            'scene_name': 'GAN Attack',
            'n_attempts': n_attempts,
            'n_successful': len(successful_attacks),
            'success_rate': float(success_rate),
            'avg_similarity': float(np.mean(similarities)),
            'max_similarity': float(np.max(similarities)),
            'min_similarity': float(np.min(similarities)),
            'std_similarity': float(np.std(similarities)),
            'threshold': self.auth_model.threshold,
            'attack_results': attack_results,
            'successful_attacks': successful_attacks,
            'best_attempt': max(attack_results, key=lambda x: x['similarity_score']),
            'message': self._format_scene3_message(success_rate, len(successful_attacks), n_attempts),
            'status': 'success' if success_rate > 0 else 'failure'
        }

        logger.info(f"Scene 3 complete: {len(successful_attacks)}/{n_attempts} successful, rate={success_rate:.2%}")
        return result

    def scene_4_visualization(
        self,
        real_pattern: np.ndarray,
        impostor_pattern: np.ndarray,
        gan_pattern: np.ndarray
    ) -> Dict[str, Any]:
        """
        Scene 4: Pattern comparison and visualization

        Args:
            real_pattern: Real user pattern
            impostor_pattern: Impostor pattern
            gan_pattern: GAN-generated pattern

        Returns:
            Comparison results
        """
        logger.info("Running Scene 4: Pattern Comparison")

        # Get similarity scores
        real_score = float(self.auth_model.predict(real_pattern)[0])
        impostor_score = float(self.auth_model.predict(impostor_pattern)[0])
        gan_score = float(self.auth_model.predict(gan_pattern)[0])

        # Compare patterns
        from src.models.evaluator import ModelEvaluator
        evaluator = ModelEvaluator()

        real_vs_gan = evaluator.compare_patterns(real_pattern, gan_pattern)
        real_vs_impostor = evaluator.compare_patterns(real_pattern, impostor_pattern)

        result = {
            'scene': 4,
            'scene_name': 'Pattern Comparison',
            'real_score': real_score,
            'impostor_score': impostor_score,
            'gan_score': gan_score,
            'threshold': self.auth_model.threshold,
            'real_vs_gan_comparison': real_vs_gan,
            'real_vs_impostor_comparison': real_vs_impostor,
            'patterns': {
                'real': real_pattern.tolist(),
                'impostor': impostor_pattern.tolist(),
                'gan': gan_pattern.tolist()
            },
            'message': self._format_scene4_message(real_score, impostor_score, gan_score),
            'status': 'complete'
        }

        logger.info(f"Scene 4 complete: Real={real_score:.2%}, Impostor={impostor_score:.2%}, GAN={gan_score:.2%}")
        return result

    def simulate_impostor_pattern(self, real_pattern: np.ndarray, noise_level: float = 0.3) -> np.ndarray:
        """
        Simulate an impostor pattern by adding noise to real pattern

        Args:
            real_pattern: Real user pattern
            noise_level: Amount of noise to add (0-1)

        Returns:
            Simulated impostor pattern
        """
        noise = np.random.normal(0, noise_level, real_pattern.shape)
        impostor_pattern = real_pattern + noise * np.std(real_pattern)

        # Ensure positive values (timing can't be negative)
        impostor_pattern = np.maximum(impostor_pattern, 0.01)

        logger.info(f"Simulated impostor pattern with noise_level={noise_level}")
        return impostor_pattern

    def _format_scene1_message(self, is_auth: bool, similarity: float) -> str:
        """Format message for Scene 1"""
        if is_auth:
            return f"""
✓ LOGIN SUCCESS!

Password: Correct ✓
Keystroke Pattern: Verified ✓
Similarity Score: {similarity:.1%}

Welcome back, Instructor!
            """
        else:
            return f"""
✗ LOGIN DENIED

Password: Correct ✓
Keystroke Pattern: MISMATCH ✗
Similarity Score: {similarity:.1%}

This doesn't look like you.
            """

    def _format_scene2_message(self, is_auth: bool, similarity: float) -> str:
        """Format message for Scene 2"""
        if not is_auth:
            return f"""
✗ LOGIN DENIED

Password: Correct ✓
Keystroke Pattern: MISMATCH ✗
Similarity Score: {similarity:.1%}

Security Alert: Unauthorized access attempt detected!
The keystroke pattern does not match the legitimate user.
            """
        else:
            return f"""
⚠ UNEXPECTED: LOGIN SUCCESS

Password: Correct ✓
Keystroke Pattern: Verified ✓
Similarity Score: {similarity:.1%}

Warning: This impostor somehow matched the pattern!
            """

    def _format_scene3_message(self, success_rate: float, n_successful: int, n_attempts: int) -> str:
        """Format message for Scene 3"""
        if success_rate > 0:
            return f"""
⚠ GAN ATTACK SUCCESSFUL!

Total Attempts: {n_attempts}
Successful Bypasses: {n_successful}
Success Rate: {success_rate:.1%}

CONCLUSION: The GAN successfully learned to mimic
the legitimate user's keystroke pattern and bypassed
the authentication system!

This demonstrates the vulnerability of behavioral
biometrics to advanced AI attacks.
            """
        else:
            return f"""
✓ SYSTEM SECURE

Total Attempts: {n_attempts}
Successful Bypasses: 0
Success Rate: 0%

All GAN-generated patterns were rejected.
The system successfully defended against the attack.
            """

    def _format_scene4_message(self, real_score: float, impostor_score: float, gan_score: float) -> str:
        """Format message for Scene 4"""
        return f"""
PATTERN ANALYSIS SUMMARY

Legitimate User:  {real_score:.1%}  →  ✓ Authenticated
Impostor Attempt: {impostor_score:.1%}  →  ✗ Denied
GAN Attack:       {gan_score:.1%}  →  {'✓ Bypassed' if gan_score >= self.auth_model.threshold else '✗ Blocked'}

KEY INSIGHTS:
• GAN learned timing patterns from training data
• Generated samples closely mimic real user behavior
• Demonstrates need for multi-factor authentication
• Behavioral biometrics alone are vulnerable to AI attacks

DEFENSE RECOMMENDATIONS:
1. Combine with traditional passwords/2FA
2. Implement anomaly detection for AI-generated patterns
3. Regular model updates with adversarial training
4. Monitor for suspicious authentication patterns
        """
