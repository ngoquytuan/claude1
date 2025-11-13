"""
Visualization utilities for demo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.figure import Figure
from typing import List, Dict, Any
import os

from config.settings import (
    UI_COLOR_SUCCESS, UI_COLOR_FAILURE, UI_COLOR_WARNING, TARGET_TEXT
)
from src.utils.logger import get_logger

logger = get_logger()


class PatternVisualizer:
    """
    Visualize keystroke patterns and comparisons
    """

    def __init__(self):
        """Initialize visualizer"""
        self.colors = {
            'real': '#2ecc71',
            'impostor': '#e74c3c',
            'gan': '#3498db',
            'success': UI_COLOR_SUCCESS,
            'failure': UI_COLOR_FAILURE,
            'warning': UI_COLOR_WARNING
        }

    def plot_timing_diagram(
        self,
        real_pattern: np.ndarray,
        fake_pattern: np.ndarray = None,
        save_path: str = None,
        show: bool = False
    ) -> Figure:
        """
        Plot timing diagram comparing real and fake patterns

        Args:
            real_pattern: Real keystroke feature vector
            fake_pattern: Fake keystroke feature vector (optional)
            save_path: Path to save figure
            show: Whether to show figure

        Returns:
            Matplotlib figure
        """
        # Extract timing features
        n_chars = len(TARGET_TEXT)

        # Dwell times
        real_dwell = real_pattern[:n_chars]
        fake_dwell = fake_pattern[:n_chars] if fake_pattern is not None else None

        # Flight times
        real_flight = real_pattern[n_chars:n_chars*2-1]
        fake_flight = fake_pattern[n_chars:n_chars*2-1] if fake_pattern is not None else None

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot dwell times
        x = np.arange(n_chars)
        axes[0].bar(x - 0.2, real_dwell * 1000, width=0.4, label='Real', color=self.colors['real'], alpha=0.7)
        if fake_dwell is not None:
            axes[0].bar(x + 0.2, fake_dwell * 1000, width=0.4, label='Fake (GAN)', color=self.colors['gan'], alpha=0.7)

        axes[0].set_xlabel('Character Position', fontsize=12)
        axes[0].set_ylabel('Dwell Time (ms)', fontsize=12)
        axes[0].set_title('Dwell Time Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(list(TARGET_TEXT))
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Plot flight times
        x_flight = np.arange(n_chars - 1)
        axes[1].bar(x_flight - 0.2, real_flight * 1000, width=0.4, label='Real', color=self.colors['real'], alpha=0.7)
        if fake_flight is not None:
            axes[1].bar(x_flight + 0.2, fake_flight * 1000, width=0.4, label='Fake (GAN)', color=self.colors['gan'], alpha=0.7)

        axes[1].set_xlabel('Character Transition', fontsize=12)
        axes[1].set_ylabel('Flight Time (ms)', fontsize=12)
        axes[1].set_title('Flight Time Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x_flight)
        transitions = [f"{TARGET_TEXT[i]}→{TARGET_TEXT[i+1]}" for i in range(n_chars-1)]
        axes[1].set_xticklabels(transitions, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Timing diagram saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_similarity_gauge(
        self,
        similarity_score: float,
        threshold: float = 0.85,
        save_path: str = None,
        show: bool = False
    ) -> Figure:
        """
        Plot similarity gauge (semicircle)

        Args:
            similarity_score: Similarity score (0-1)
            threshold: Threshold for acceptance
            save_path: Path to save figure
            show: Whether to show figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        # Draw semicircle background
        theta1, theta2 = 180, 0
        center = (0.5, 0.5)
        radius = 0.4

        # Color zones
        colors = [self.colors['failure'], self.colors['warning'], self.colors['success']]
        ranges = [(0, 0.5), (0.5, threshold), (threshold, 1.0)]

        for (low, high), color in zip(ranges, colors):
            angle_low = 180 - (low * 180)
            angle_high = 180 - (high * 180)

            wedge = Wedge(
                center, radius, angle_high, angle_low,
                facecolor=color, alpha=0.3, edgecolor='black', linewidth=1.5
            )
            ax.add_patch(wedge)

        # Draw needle
        angle = 180 - (similarity_score * 180)
        angle_rad = np.radians(angle)

        needle_x = [center[0], center[0] + radius * 0.9 * np.cos(angle_rad)]
        needle_y = [center[1], center[1] + radius * 0.9 * np.sin(angle_rad)]

        ax.plot(needle_x, needle_y, 'k-', linewidth=3)
        ax.plot(center[0], center[1], 'ko', markersize=10)

        # Add text
        ax.text(0.5, 0.15, f'{similarity_score:.1%}', ha='center', va='center',
                fontsize=32, fontweight='bold')

        status = "AUTHENTICATED" if similarity_score >= threshold else "DENIED"
        status_color = self.colors['success'] if similarity_score >= threshold else self.colors['failure']
        ax.text(0.5, 0.05, status, ha='center', va='center',
                fontsize=16, fontweight='bold', color=status_color)

        # Add labels
        ax.text(0.1, 0.5, '0%', ha='center', va='center', fontsize=12)
        ax.text(0.9, 0.5, '100%', ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.9, 'Similarity Score', ha='center', va='center',
                fontsize=16, fontweight='bold')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Similarity gauge saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_attack_attempts(
        self,
        attempt_scores: List[float],
        threshold: float = 0.85,
        save_path: str = None,
        show: bool = False
    ) -> Figure:
        """
        Plot GAN attack attempts

        Args:
            attempt_scores: List of similarity scores for each attempt
            threshold: Acceptance threshold
            save_path: Path to save figure
            show: Whether to show figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        n_attempts = len(attempt_scores)
        x = np.arange(n_attempts)

        # Color bars based on success/failure
        colors = [self.colors['success'] if s >= threshold else self.colors['failure']
                  for s in attempt_scores]

        bars = ax.bar(x, attempt_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        # Add threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.0%})')

        # Highlight successful attempts
        success_indices = [i for i, s in enumerate(attempt_scores) if s >= threshold]
        if success_indices:
            for idx in success_indices:
                ax.text(idx, attempt_scores[idx] + 0.02, '✓',
                       ha='center', va='bottom', fontsize=16, color='green', fontweight='bold')

        ax.set_xlabel('Attempt Number', fontsize=12)
        ax.set_ylabel('Similarity Score', fontsize=12)
        ax.set_title(f'GAN Attack Attempts ({len(success_indices)}/{n_attempts} successful)',
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Attack attempts plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_feature_comparison(
        self,
        real_pattern: np.ndarray,
        fake_pattern: np.ndarray,
        feature_names: List[str] = None,
        save_path: str = None,
        show: bool = False
    ) -> Figure:
        """
        Plot detailed feature comparison

        Args:
            real_pattern: Real feature vector
            fake_pattern: Fake feature vector
            feature_names: List of feature names
            save_path: Path to save figure
            show: Whether to show figure

        Returns:
            Matplotlib figure
        """
        n_features = len(real_pattern)

        if feature_names is None:
            feature_names = [f'F{i}' for i in range(n_features)]

        # Select subset of features if too many
        max_features = 20
        if n_features > max_features:
            indices = np.linspace(0, n_features-1, max_features, dtype=int)
            real_pattern = real_pattern[indices]
            fake_pattern = fake_pattern[indices]
            feature_names = [feature_names[i] for i in indices]
            n_features = max_features

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(n_features)
        width = 0.35

        ax.bar(x - width/2, real_pattern, width, label='Real', color=self.colors['real'], alpha=0.7)
        ax.bar(x + width/2, fake_pattern, width, label='Fake (GAN)', color=self.colors['gan'], alpha=0.7)

        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Feature Value', fontsize=12)
        ax.set_title('Feature-by-Feature Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature comparison saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def create_comparison_figure(
        self,
        real_pattern: np.ndarray,
        impostor_pattern: np.ndarray,
        gan_pattern: np.ndarray,
        real_score: float,
        impostor_score: float,
        gan_score: float,
        threshold: float = 0.85,
        save_path: str = None,
        show: bool = False
    ) -> Figure:
        """
        Create comprehensive comparison figure for demo

        Args:
            real_pattern: Real user pattern
            impostor_pattern: Impostor pattern
            gan_pattern: GAN-generated pattern
            real_score: Real user similarity score
            impostor_score: Impostor similarity score
            gan_score: GAN similarity score
            threshold: Authentication threshold
            save_path: Path to save figure
            show: Whether to show figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10))

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Extract timing features
        n_chars = len(TARGET_TEXT)

        # Plot 1: Dwell times comparison
        ax1 = fig.add_subplot(gs[0, :2])
        x = np.arange(n_chars)
        width = 0.25

        real_dwell = real_pattern[:n_chars] * 1000
        impostor_dwell = impostor_pattern[:n_chars] * 1000
        gan_dwell = gan_pattern[:n_chars] * 1000

        ax1.bar(x - width, real_dwell, width, label='Legitimate', color=self.colors['real'], alpha=0.7)
        ax1.bar(x, impostor_dwell, width, label='Impostor', color=self.colors['impostor'], alpha=0.7)
        ax1.bar(x + width, gan_dwell, width, label='GAN Attack', color=self.colors['gan'], alpha=0.7)

        ax1.set_title('Dwell Time Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Character')
        ax1.set_ylabel('Time (ms)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(list(TARGET_TEXT))
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Similarity scores
        ax2 = fig.add_subplot(gs[0, 2])
        scores = [real_score, impostor_score, gan_score]
        labels = ['Legitimate', 'Impostor', 'GAN']
        colors = [self.colors['real'], self.colors['impostor'], self.colors['gan']]

        bars = ax2.barh(labels, scores, color=colors, alpha=0.7)
        ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Similarity Score')
        ax2.set_title('Authentication Results', fontweight='bold')
        ax2.legend()

        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax2.text(score + 0.02, i, f'{score:.2%}', va='center')

        # Plot 3: Flight times
        ax3 = fig.add_subplot(gs[1, :])
        x_flight = np.arange(n_chars - 1)

        real_flight = real_pattern[n_chars:n_chars*2-1] * 1000
        impostor_flight = impostor_pattern[n_chars:n_chars*2-1] * 1000
        gan_flight = gan_pattern[n_chars:n_chars*2-1] * 1000

        ax3.plot(x_flight, real_flight, 'o-', label='Legitimate', color=self.colors['real'], linewidth=2)
        ax3.plot(x_flight, impostor_flight, 's-', label='Impostor', color=self.colors['impostor'], linewidth=2)
        ax3.plot(x_flight, gan_flight, '^-', label='GAN Attack', color=self.colors['gan'], linewidth=2)

        ax3.set_title('Flight Time Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Character Transition')
        ax3.set_ylabel('Time (ms)')
        ax3.set_xticks(x_flight)
        transitions = [f"{TARGET_TEXT[i]}→{TARGET_TEXT[i+1]}" for i in range(n_chars-1)]
        ax3.set_xticklabels(transitions, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Summary text
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')

        summary_text = f"""
DEMO SUMMARY:

Scene 1 - Legitimate User:  Score = {real_score:.2%}  →  {'✓ AUTHENTICATED' if real_score >= threshold else '✗ DENIED'}
Scene 2 - Impostor Attempt: Score = {impostor_score:.2%}  →  {'✓ AUTHENTICATED' if impostor_score >= threshold else '✗ DENIED'}
Scene 3 - GAN Attack:       Score = {gan_score:.2%}  →  {'✓ BYPASSED!' if gan_score >= threshold else '✗ BLOCKED'}

Authentication Threshold: {threshold:.0%}

CONCLUSION: GAN successfully learned to mimic keystroke patterns!
        """

        ax4.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=12, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison figure saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig
