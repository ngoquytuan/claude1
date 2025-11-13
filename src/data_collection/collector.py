"""
Keystroke data collection with GUI
Captures keystroke timing data including keydown and keyup events
"""

import tkinter as tk
from tkinter import ttk, messagebox
import time
from datetime import datetime
from typing import List, Dict, Any
import os

from config.settings import (
    TARGET_TEXT, NUM_SAMPLES, DATA_RAW_PATH,
    UI_FONT_LARGE, UI_FONT_MEDIUM, UI_COLOR_SUCCESS, UI_COLOR_FAILURE
)
from src.utils.helpers import save_json, ensure_dir
from src.utils.logger import get_logger

logger = get_logger()


class KeystrokeCollector:
    """
    GUI application for collecting keystroke timing data
    """

    def __init__(self, target_text: str = TARGET_TEXT, num_samples: int = NUM_SAMPLES):
        """
        Initialize keystroke collector

        Args:
            target_text: Text that user must type
            num_samples: Number of samples to collect
        """
        self.target_text = target_text
        self.num_samples = num_samples
        self.samples_collected = 0
        self.current_sample = []
        self.samples = []
        self.is_collecting = False
        self.start_time = None

        # Setup GUI
        self.window = tk.Tk()
        self.window.title("Keystroke Data Collection")
        self.window.geometry("800x600")
        self.setup_ui()

        logger.info(f"Initialized KeystrokeCollector: target='{target_text}', num_samples={num_samples}")

    def setup_ui(self):
        """Setup user interface"""
        # Title
        title_label = tk.Label(
            self.window,
            text="KEYSTROKE DATA COLLECTION",
            font=UI_FONT_LARGE,
            fg=UI_COLOR_SUCCESS
        )
        title_label.pack(pady=20)

        # Instructions
        instruction_text = f"Please type the following text exactly:\n\n'{self.target_text}'\n\n" \
                          f"Press Enter after each attempt."
        instruction_label = tk.Label(
            self.window,
            text=instruction_text,
            font=UI_FONT_MEDIUM,
            justify=tk.CENTER
        )
        instruction_label.pack(pady=10)

        # Progress
        self.progress_label = tk.Label(
            self.window,
            text=f"Progress: 0/{self.num_samples}",
            font=UI_FONT_MEDIUM
        )
        self.progress_label.pack(pady=10)

        self.progress_bar = ttk.Progressbar(
            self.window,
            length=400,
            mode='determinate',
            maximum=self.num_samples
        )
        self.progress_bar.pack(pady=10)

        # Input field
        self.input_var = tk.StringVar()
        self.input_var.trace('w', self.on_input_change)

        self.input_entry = tk.Entry(
            self.window,
            textvariable=self.input_var,
            font=UI_FONT_MEDIUM,
            width=30,
            justify=tk.CENTER
        )
        self.input_entry.pack(pady=20)
        self.input_entry.bind('<KeyPress>', self.on_key_press)
        self.input_entry.bind('<KeyRelease>', self.on_key_release)
        self.input_entry.bind('<Return>', self.on_submit)

        # Status
        self.status_label = tk.Label(
            self.window,
            text="Ready to collect data. Start typing!",
            font=UI_FONT_MEDIUM,
            fg="blue"
        )
        self.status_label.pack(pady=10)

        # Buttons
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=20)

        self.start_button = tk.Button(
            button_frame,
            text="Start",
            font=UI_FONT_MEDIUM,
            command=self.start_collection,
            bg=UI_COLOR_SUCCESS,
            fg="white",
            width=15
        )
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.reset_button = tk.Button(
            button_frame,
            text="Reset",
            font=UI_FONT_MEDIUM,
            command=self.reset,
            width=15
        )
        self.reset_button.pack(side=tk.LEFT, padx=10)

        self.save_button = tk.Button(
            button_frame,
            text="Save & Exit",
            font=UI_FONT_MEDIUM,
            command=self.save_and_exit,
            state=tk.DISABLED,
            width=15
        )
        self.save_button.pack(side=tk.LEFT, padx=10)

        # Initially disable input
        self.input_entry.config(state=tk.DISABLED)

    def start_collection(self):
        """Start collecting keystroke data"""
        self.is_collecting = True
        self.input_entry.config(state=tk.NORMAL)
        self.input_entry.focus()
        self.start_button.config(state=tk.DISABLED)
        self.status_label.config(text="Collecting... Type the text and press Enter", fg="green")
        logger.info("Started data collection")

    def on_key_press(self, event):
        """Handle key press event"""
        if not self.is_collecting:
            return

        # Ignore special keys
        if event.keysym in ['Shift_L', 'Shift_R', 'Control_L', 'Control_R', 'Alt_L', 'Alt_R', 'Return']:
            return

        # Record key press time
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time

        # Find if this key is already being tracked (for hold time calculation)
        key_char = event.char if event.char else event.keysym
        position = len(self.current_sample)

        self.current_sample.append({
            'char': key_char,
            'position': position,
            'keydown_time': current_time - self.start_time,
            'keyup_time': None  # Will be filled on key release
        })

    def on_key_release(self, event):
        """Handle key release event"""
        if not self.is_collecting:
            return

        # Ignore special keys
        if event.keysym in ['Shift_L', 'Shift_R', 'Control_L', 'Control_R', 'Alt_L', 'Alt_R', 'Return']:
            return

        # Record key release time
        current_time = time.time()
        key_char = event.char if event.char else event.keysym

        # Find the last keypress for this character without keyup_time
        for kp in reversed(self.current_sample):
            if kp['char'] == key_char and kp['keyup_time'] is None:
                kp['keyup_time'] = current_time - self.start_time
                break

    def on_input_change(self, *args):
        """Handle input text change"""
        current_text = self.input_var.get()
        if len(current_text) > len(self.target_text):
            self.input_var.set(current_text[:len(self.target_text)])

    def on_submit(self, event):
        """Handle Enter key press to submit sample"""
        if not self.is_collecting:
            return

        typed_text = self.input_var.get()

        # Validate input
        if self.validate_input(typed_text):
            self.save_sample()
            self.samples_collected += 1
            self.progress_bar['value'] = self.samples_collected
            self.progress_label.config(text=f"Progress: {self.samples_collected}/{self.num_samples}")

            if self.samples_collected >= self.num_samples:
                self.finish_collection()
            else:
                self.status_label.config(text=f"✓ Sample saved! ({self.samples_collected}/{self.num_samples})", fg="green")
                self.clear_input()
        else:
            self.status_label.config(
                text=f"✗ Incorrect! Expected: '{self.target_text}' Got: '{typed_text}'",
                fg=UI_COLOR_FAILURE
            )
            self.clear_input()

        return 'break'  # Prevent default Enter behavior

    def validate_input(self, typed_text: str) -> bool:
        """
        Validate if typed text matches target

        Args:
            typed_text: Text typed by user

        Returns:
            True if valid, False otherwise
        """
        return typed_text == self.target_text

    def save_sample(self):
        """Save current sample to samples list"""
        # Filter out incomplete keypresses
        valid_keypresses = [kp for kp in self.current_sample if kp['keyup_time'] is not None]

        sample = {
            'sample_id': self.samples_collected + 1,
            'timestamp': datetime.now().isoformat(),
            'keypresses': valid_keypresses
        }

        self.samples.append(sample)
        logger.info(f"Saved sample {sample['sample_id']} with {len(valid_keypresses)} keypresses")

    def clear_input(self):
        """Clear input field and reset current sample"""
        self.input_var.set('')
        self.current_sample = []
        self.start_time = None

    def reset(self):
        """Reset collection"""
        if self.samples_collected > 0:
            if not messagebox.askyesno("Reset", "Are you sure you want to reset? All collected data will be lost."):
                return

        self.samples_collected = 0
        self.samples = []
        self.current_sample = []
        self.start_time = None
        self.is_collecting = False

        self.progress_bar['value'] = 0
        self.progress_label.config(text=f"Progress: 0/{self.num_samples}")
        self.status_label.config(text="Reset complete. Press Start to begin.", fg="blue")
        self.clear_input()

        self.input_entry.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)

        logger.info("Reset collection")

    def finish_collection(self):
        """Finish data collection"""
        self.is_collecting = False
        self.input_entry.config(state=tk.DISABLED)
        self.status_label.config(
            text=f"✓ Collection complete! {self.samples_collected} samples collected.",
            fg=UI_COLOR_SUCCESS
        )
        self.save_button.config(state=tk.NORMAL)
        logger.info(f"Completed collection of {self.samples_collected} samples")

    def export_data(self, user_id: str = "instructor") -> str:
        """
        Export collected data to JSON file

        Args:
            user_id: Identifier for the user

        Returns:
            Path to saved file
        """
        ensure_dir(DATA_RAW_PATH)

        output_data = {
            'metadata': {
                'user_id': user_id,
                'target_text': self.target_text,
                'collection_date': datetime.now().isoformat(),
                'num_samples': len(self.samples)
            },
            'samples': self.samples
        }

        filename = f"keystroke_samples_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(DATA_RAW_PATH, filename)

        save_json(output_data, filepath)
        logger.info(f"Exported data to {filepath}")

        return filepath

    def save_and_exit(self):
        """Save data and exit application"""
        if self.samples_collected == 0:
            messagebox.showwarning("No Data", "No samples collected. Nothing to save.")
            return

        filepath = self.export_data()
        messagebox.showinfo(
            "Success",
            f"Data saved successfully!\n\nFile: {filepath}\nSamples: {self.samples_collected}"
        )
        self.window.quit()

    def run(self):
        """Run the GUI application"""
        logger.info("Starting GUI")
        self.window.mainloop()


if __name__ == "__main__":
    collector = KeystrokeCollector()
    collector.run()
