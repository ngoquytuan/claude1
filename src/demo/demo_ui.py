"""
Main demo UI for Keystroke GAN Demo
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
from typing import Optional
import numpy as np

from config.settings import (
    UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT, UI_FONT_LARGE, UI_FONT_MEDIUM, UI_FONT_SMALL,
    UI_COLOR_SUCCESS, UI_COLOR_FAILURE, UI_COLOR_INFO, UI_COLOR_WARNING,
    MODEL_PATH, TARGET_TEXT
)
from src.models.authentication import AuthenticationModel
from src.models.gan import KeystrokeGAN
from src.demo.scenes import DemoScenes
from src.demo.visualizer import PatternVisualizer
from src.data_collection.collector import KeystrokeCollector
from src.utils.logger import get_logger
from src.utils.helpers import validate_model_path

logger = get_logger()


class DemoUI:
    """
    Main demo GUI application
    """

    def __init__(self):
        """Initialize demo UI"""
        self.window = tk.Tk()
        self.window.title("Keystroke Authentication + GAN Bypass Demo")
        self.window.geometry(f"{UI_WINDOW_WIDTH}x{UI_WINDOW_HEIGHT}")

        # Models
        self.auth_model: Optional[AuthenticationModel] = None
        self.gan_model: Optional[KeystrokeGAN] = None
        self.scenes: Optional[DemoScenes] = None
        self.visualizer = PatternVisualizer()

        # Demo data
        self.real_pattern = None
        self.impostor_pattern = None
        self.gan_pattern = None

        # UI components
        self.setup_ui()

        logger.info("DemoUI initialized")

    def setup_ui(self):
        """Setup user interface"""
        # Header
        header_frame = tk.Frame(self.window, bg=UI_COLOR_INFO, height=80)
        header_frame.pack(fill=tk.X)

        title_label = tk.Label(
            header_frame,
            text="KEYSTROKE AUTHENTICATION + GAN BYPASS DEMO",
            font=UI_FONT_LARGE,
            bg=UI_COLOR_INFO,
            fg="white"
        )
        title_label.pack(pady=20)

        # Main container
        main_container = tk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls
        left_panel = tk.Frame(main_container, width=300, relief=tk.RIDGE, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.setup_control_panel(left_panel)

        # Right panel - Display
        right_panel = tk.Frame(main_container, relief=tk.RIDGE, borderwidth=2)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.setup_display_panel(right_panel)

        # Status bar
        status_frame = tk.Frame(self.window, relief=tk.SUNKEN, borderwidth=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(
            status_frame,
            text="Ready. Please load models to begin.",
            font=UI_FONT_SMALL,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, padx=5, pady=2)

    def setup_control_panel(self, parent):
        """Setup control panel"""
        # Model loading section
        model_frame = tk.LabelFrame(parent, text="Models", font=UI_FONT_MEDIUM, padx=10, pady=10)
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        self.load_models_btn = tk.Button(
            model_frame,
            text="Load Models",
            font=UI_FONT_MEDIUM,
            command=self.load_models,
            bg=UI_COLOR_INFO,
            fg="white",
            width=20
        )
        self.load_models_btn.pack(pady=5)

        self.models_status_label = tk.Label(
            model_frame,
            text="⚪ Models not loaded",
            font=UI_FONT_SMALL
        )
        self.models_status_label.pack(pady=5)

        # Scene selection
        scene_frame = tk.LabelFrame(parent, text="Select Scene", font=UI_FONT_MEDIUM, padx=10, pady=10)
        scene_frame.pack(fill=tk.X, padx=5, pady=5)

        self.scene_var = tk.IntVar(value=1)

        scenes = [
            (1, "Scene 1: Legitimate User"),
            (2, "Scene 2: Impostor Attempt"),
            (3, "Scene 3: GAN Attack"),
            (4, "Scene 4: Visualization")
        ]

        for value, text in scenes:
            rb = tk.Radiobutton(
                scene_frame,
                text=text,
                variable=self.scene_var,
                value=value,
                font=UI_FONT_SMALL
            )
            rb.pack(anchor=tk.W, pady=2)

        # Action buttons
        action_frame = tk.Frame(parent, padx=10, pady=10)
        action_frame.pack(fill=tk.X, padx=5, pady=5)

        self.run_scene_btn = tk.Button(
            action_frame,
            text="Run Scene",
            font=UI_FONT_MEDIUM,
            command=self.run_selected_scene,
            bg=UI_COLOR_SUCCESS,
            fg="white",
            width=20,
            state=tk.DISABLED
        )
        self.run_scene_btn.pack(pady=5)

        self.collect_data_btn = tk.Button(
            action_frame,
            text="Collect Data",
            font=UI_FONT_MEDIUM,
            command=self.open_data_collector,
            width=20
        )
        self.collect_data_btn.pack(pady=5)

        self.reset_btn = tk.Button(
            action_frame,
            text="Reset",
            font=UI_FONT_MEDIUM,
            command=self.reset,
            width=20
        )
        self.reset_btn.pack(pady=5)

        # Quick stats
        stats_frame = tk.LabelFrame(parent, text="Statistics", font=UI_FONT_MEDIUM, padx=10, pady=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.stats_text = scrolledtext.ScrolledText(
            stats_frame,
            width=30,
            height=10,
            font=UI_FONT_SMALL,
            wrap=tk.WORD
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True)

    def setup_display_panel(self, parent):
        """Setup display panel"""
        # Result display
        result_frame = tk.LabelFrame(parent, text="Demo Output", font=UI_FONT_MEDIUM, padx=10, pady=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.result_text = scrolledtext.ScrolledText(
            result_frame,
            font=("Courier", 11),
            wrap=tk.WORD,
            bg="#f5f5f5"
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Progress bar (hidden by default)
        self.progress_frame = tk.Frame(parent)
        self.progress_label = tk.Label(self.progress_frame, text="Processing...", font=UI_FONT_SMALL)
        self.progress_label.pack()

        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='determinate',
            length=400
        )
        self.progress_bar.pack(pady=5)

    def load_models(self):
        """Load trained models"""
        self.update_status("Loading models...")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Loading models...\n")

        try:
            # Check if models exist
            auth_model_path = os.path.join(MODEL_PATH, 'auth_model.pkl')
            gan_model_path = os.path.join(MODEL_PATH, 'gan_model.h5')

            if not validate_model_path(auth_model_path):
                messagebox.showerror("Error", f"Authentication model not found at {auth_model_path}")
                return

            # Load authentication model
            self.result_text.insert(tk.END, f"Loading authentication model from {auth_model_path}...\n")
            self.auth_model = AuthenticationModel()
            self.auth_model.load_model(auth_model_path)
            self.result_text.insert(tk.END, "✓ Authentication model loaded\n")

            # Load GAN model
            if validate_model_path(gan_model_path.replace('.h5', '_generator.h5')):
                self.result_text.insert(tk.END, f"Loading GAN model from {gan_model_path}...\n")

                # Get feature dimension from auth model
                feature_dim = self.auth_model.scaler.n_features_in_

                self.gan_model = KeystrokeGAN(latent_dim=100, feature_dim=feature_dim)
                self.gan_model.load_models(gan_model_path)
                self.result_text.insert(tk.END, "✓ GAN model loaded\n")
            else:
                self.result_text.insert(tk.END, "⚠ GAN model not found. Scene 3 will be unavailable.\n")

            # Initialize scenes
            self.scenes = DemoScenes(self.auth_model, self.gan_model)

            # Update UI
            self.models_status_label.config(text="✓ Models loaded", fg=UI_COLOR_SUCCESS)
            self.run_scene_btn.config(state=tk.NORMAL)
            self.update_status("Models loaded successfully")

            self.result_text.insert(tk.END, "\n✓ All models loaded successfully!\n")
            self.result_text.insert(tk.END, "You can now run demo scenes.\n")

            # Update stats
            self.update_stats()

        except Exception as e:
            error_msg = f"Failed to load models: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
            self.update_status("Failed to load models")

    def run_selected_scene(self):
        """Run the selected scene"""
        scene_num = self.scene_var.get()

        if self.auth_model is None:
            messagebox.showwarning("Warning", "Please load models first")
            return

        # Run scene in thread to prevent UI freezing
        thread = threading.Thread(target=self._run_scene_thread, args=(scene_num,))
        thread.daemon = True
        thread.start()

    def _run_scene_thread(self, scene_num):
        """Run scene in separate thread"""
        self.run_scene_btn.config(state=tk.DISABLED)
        self.result_text.delete(1.0, tk.END)

        try:
            if scene_num == 1:
                self._run_scene_1()
            elif scene_num == 2:
                self._run_scene_2()
            elif scene_num == 3:
                self._run_scene_3()
            elif scene_num == 4:
                self._run_scene_4()

        except Exception as e:
            error_msg = f"Error running scene: {str(e)}"
            logger.error(error_msg)
            self.result_text.insert(tk.END, f"\n❌ {error_msg}\n")

        finally:
            self.run_scene_btn.config(state=tk.NORMAL)

    def _run_scene_1(self):
        """Run Scene 1: Legitimate user"""
        self.result_text.insert(tk.END, "="*60 + "\n")
        self.result_text.insert(tk.END, "SCENE 1: LEGITIMATE USER LOGIN\n")
        self.result_text.insert(tk.END, "="*60 + "\n\n")

        # For demo, use simulated data or ask for live input
        self.result_text.insert(tk.END, "Simulating legitimate user keystroke pattern...\n\n")

        # Generate a sample pattern (in real demo, this would be live captured)
        if self.real_pattern is None:
            self.real_pattern = self.gan_model.generate(1)[0] if self.gan_model else np.random.rand(33)

        # Create sample
        sample = self._create_mock_sample(self.real_pattern)

        # Run scene
        result = self.scenes.scene_1_real_user(sample)

        # Display result
        self.result_text.insert(tk.END, result['message'])
        self.result_text.insert(tk.END, f"\n{'='*60}\n")

        # Update stats
        self.update_scene_stats(result)

    def _run_scene_2(self):
        """Run Scene 2: Impostor"""
        self.result_text.insert(tk.END, "="*60 + "\n")
        self.result_text.insert(tk.END, "SCENE 2: IMPOSTOR LOGIN ATTEMPT\n")
        self.result_text.insert(tk.END, "="*60 + "\n\n")

        self.result_text.insert(tk.END, "Simulating impostor keystroke pattern...\n\n")

        # Generate impostor pattern
        if self.real_pattern is None:
            self.real_pattern = self.gan_model.generate(1)[0] if self.gan_model else np.random.rand(33)

        self.impostor_pattern = self.scenes.simulate_impostor_pattern(self.real_pattern, noise_level=0.5)

        # Create sample
        sample = self._create_mock_sample(self.impostor_pattern)

        # Run scene
        result = self.scenes.scene_2_impostor(sample)

        # Display result
        self.result_text.insert(tk.END, result['message'])
        self.result_text.insert(tk.END, f"\n{'='*60}\n")

        # Update stats
        self.update_scene_stats(result)

    def _run_scene_3(self):
        """Run Scene 3: GAN attack"""
        if self.gan_model is None:
            messagebox.showwarning("Warning", "GAN model not loaded")
            return

        self.result_text.insert(tk.END, "="*60 + "\n")
        self.result_text.insert(tk.END, "SCENE 3: GAN ATTACK SIMULATION\n")
        self.result_text.insert(tk.END, "="*60 + "\n\n")

        self.result_text.insert(tk.END, "Running GAN attack...\n")
        self.result_text.insert(tk.END, "Generating fake keystroke patterns and testing authentication...\n\n")

        # Show progress
        self.progress_frame.pack(pady=10)
        self.progress_bar['maximum'] = 10

        def progress_callback(current, total, similarity, is_auth):
            self.progress_bar['value'] = current
            status = "✓ BYPASS" if is_auth else "✗ DENIED"
            self.result_text.insert(
                tk.END,
                f"Attempt {current}/{total}: Similarity={similarity:.1%} → {status}\n"
            )
            self.result_text.see(tk.END)
            self.window.update()

        # Run attack
        result = self.scenes.scene_3_gan_attack(n_attempts=10, progress_callback=progress_callback)

        # Hide progress
        self.progress_frame.pack_forget()

        # Display result
        self.result_text.insert(tk.END, "\n" + "="*60 + "\n")
        self.result_text.insert(tk.END, result['message'])
        self.result_text.insert(tk.END, f"\n{'='*60}\n")

        # Store best GAN pattern for visualization
        if result['successful_attacks']:
            best_attempt = result['best_attempt']
            self.gan_pattern = self.gan_model.generate(1)[0]

        # Update stats
        self.update_scene_stats(result)

    def _run_scene_4(self):
        """Run Scene 4: Visualization"""
        self.result_text.insert(tk.END, "="*60 + "\n")
        self.result_text.insert(tk.END, "SCENE 4: PATTERN COMPARISON & VISUALIZATION\n")
        self.result_text.insert(tk.END, "="*60 + "\n\n")

        # Ensure we have patterns
        if self.real_pattern is None:
            self.real_pattern = self.gan_model.generate(1)[0] if self.gan_model else np.random.rand(33)

        if self.impostor_pattern is None:
            self.impostor_pattern = self.scenes.simulate_impostor_pattern(self.real_pattern)

        if self.gan_pattern is None and self.gan_model:
            self.gan_pattern = self.gan_model.generate(1)[0]

        # Run scene
        result = self.scenes.scene_4_visualization(
            self.real_pattern,
            self.impostor_pattern,
            self.gan_pattern
        )

        # Display text result
        self.result_text.insert(tk.END, result['message'])

        # Generate visualizations
        self.result_text.insert(tk.END, "\n\nGenerating visualizations...\n")

        try:
            # Create comparison figure
            fig_path = "comparison_figure.png"
            self.visualizer.create_comparison_figure(
                self.real_pattern,
                self.impostor_pattern,
                self.gan_pattern,
                result['real_score'],
                result['impostor_score'],
                result['gan_score'],
                threshold=self.auth_model.threshold,
                save_path=fig_path
            )

            self.result_text.insert(tk.END, f"✓ Visualization saved to {fig_path}\n")
            self.result_text.insert(tk.END, "\nOpening visualization...\n")

            # Open the image
            import platform
            if platform.system() == 'Darwin':       # macOS
                os.system(f'open {fig_path}')
            elif platform.system() == 'Windows':    # Windows
                os.system(f'start {fig_path}')
            else:                                   # Linux
                os.system(f'xdg-open {fig_path}')

        except Exception as e:
            self.result_text.insert(tk.END, f"⚠ Could not generate visualization: {e}\n")

        self.result_text.insert(tk.END, f"\n{'='*60}\n")

        # Update stats
        self.update_scene_stats(result)

    def _create_mock_sample(self, features: np.ndarray) -> dict:
        """Create mock keystroke sample from features"""
        # This is a simplified version - real implementation would reverse engineer
        # the features into keypresses
        n_chars = len(TARGET_TEXT)

        dwell_times = features[:n_chars]

        keypresses = []
        current_time = 0

        for i, char in enumerate(TARGET_TEXT):
            keydown_time = current_time
            keyup_time = keydown_time + dwell_times[i]

            keypresses.append({
                'char': char,
                'position': i,
                'keydown_time': keydown_time,
                'keyup_time': keyup_time
            })

            # Add flight time
            if i < n_chars - 1:
                flight_time = features[n_chars + i]
                current_time = keyup_time + flight_time
            else:
                current_time = keyup_time

        return {
            'sample_id': 1,
            'timestamp': '2025-11-13T10:00:00',
            'keypresses': keypresses
        }

    def update_stats(self):
        """Update statistics panel"""
        if not self.auth_model:
            return

        stats_text = "MODEL INFORMATION\n"
        stats_text += "="*30 + "\n\n"

        info = self.auth_model.get_model_info()
        stats_text += f"Authentication Model:\n"
        stats_text += f"  Threshold: {info.get('threshold', 0):.2%}\n"
        stats_text += f"  Status: Loaded ✓\n\n"

        if self.gan_model:
            gan_info = self.gan_model.get_model_summary()
            stats_text += f"GAN Model:\n"
            stats_text += f"  Latent Dim: {gan_info['latent_dim']}\n"
            stats_text += f"  Feature Dim: {gan_info['feature_dim']}\n"
            stats_text += f"  Status: Loaded ✓\n\n"

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats_text)

    def update_scene_stats(self, result: dict):
        """Update statistics with scene results"""
        stats_text = f"LAST SCENE RESULT\n"
        stats_text += "="*30 + "\n\n"

        stats_text += f"Scene: {result.get('scene_name', 'N/A')}\n"
        stats_text += f"Status: {result.get('status', 'N/A')}\n\n"

        if 'similarity_score' in result:
            stats_text += f"Similarity: {result['similarity_score']:.2%}\n"
            stats_text += f"Threshold: {result.get('threshold', 0):.2%}\n"
            stats_text += f"Authenticated: {result.get('is_authenticated', False)}\n"

        if 'success_rate' in result:
            stats_text += f"\nAttack Stats:\n"
            stats_text += f"  Success Rate: {result['success_rate']:.2%}\n"
            stats_text += f"  Attempts: {result['n_attempts']}\n"
            stats_text += f"  Successful: {result['n_successful']}\n"

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats_text)

    def open_data_collector(self):
        """Open data collection GUI"""
        try:
            collector = KeystrokeCollector()
            collector.run()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open data collector: {e}")

    def reset(self):
        """Reset demo"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Demo reset.\n")
        self.real_pattern = None
        self.impostor_pattern = None
        self.gan_pattern = None
        self.update_status("Reset complete")

    def update_status(self, message: str):
        """Update status bar"""
        self.status_label.config(text=message)
        logger.info(f"Status: {message}")

    def run(self):
        """Run the GUI application"""
        logger.info("Starting Demo UI")
        self.window.mainloop()


if __name__ == "__main__":
    demo = DemoUI()
    demo.run()
