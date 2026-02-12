"""
Unified logging utilities for Phase 2 experiments.
Tracks all experiments in a single CSV for easy analysis.
"""

import csv
import os
from pathlib import Path
from typing import Optional


def append_to_all_results(
    phase: str,
    class_name: str,
    backbone: str = "wideresnet50",
    k_config: str = "1",
    d2_mode: str = "none",
    d2_starting_points: int = 10,
    d2_exponent: float = 2.0,
    p: float = 0.10,
    tau: float = 0.01,
    postprocess: str = "none",
    pixel_ap: float = 0.0,
    image_auroc: float = 0.0,
    pixel_auroc: float = 0.0,
    csv_path: str = "phase2_all_results.csv"
):
    """
    Append experiment result to unified CSV file.
    
    Phase 2.17: Unified logging for all experiments.
    
    Args:
        phase: Experiment phase (e.g., '2.15b_step1', '2.17_postprocess')
        class_name: MVTec class name
        backbone: Backbone network name
        k_config: Multi-scale k-NN configuration (e.g., '1,3,5')
        d2_mode: D² sampling mode ('none', 'greedy', 'd2')
        d2_starting_points: Number of starting points for D²
        d2_exponent: Exponent for D² probability
        p: Coreset percentage
        tau: τ threshold for filtering
        postprocess: Post-processing method
        pixel_ap: Pixel-level Average Precision
        image_auroc: Image-level AUROC
        pixel_auroc: Pixel-level AUROC
        csv_path: Path to CSV file
    """
    csv_file = Path(csv_path)
    
    # Create header if file doesn't exist
    if not csv_file.exists():
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'phase', 'class', 'backbone', 'k_config', 'd2_mode',
                'd2_starting_points', 'd2_exponent', 'p', 'tau',
                'postprocess', 'pixel_ap', 'image_auroc', 'pixel_auroc'
            ])
    
    # Append result
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            phase, class_name, backbone, k_config, d2_mode,
            d2_starting_points, f"{d2_exponent:.1f}",
            f"{p:.3f}", f"{tau:.4f}", postprocess,
            f"{pixel_ap:.4f}", f"{image_auroc:.4f}", f"{pixel_auroc:.4f}"
        ])


def append_memory_stats(
    phase: str,
    class_name: str,
    backbone: str = "wideresnet50",
    k_config: str = "1",
    d2_mode: str = "none",
    p: float = 0.10,
    tau: float = 0.01,
    bank_size: int = 0,
    min_dist: float = 0.0,
    median_dist: float = 0.0,
    max_dist: float = 0.0,
    random_fallbacks: int = 0,
    rejected_by_tau: int = 0,
    csv_path: str = "phase2_memory_stats.csv"
):
    """
    Append memory bank statistics to CSV file.
    
    Phase 2.17: Memory bank quality analysis.
    
    Args:
        phase: Experiment phase
        class_name: MVTec class name
        backbone: Backbone network name
        k_config: Multi-scale k-NN configuration
        d2_mode: D² sampling mode
        p: Coreset percentage
        tau: τ threshold
        bank_size: Final memory bank size
        min_dist: Minimum pairwise distance
        median_dist: Median pairwise distance
        max_dist: Maximum pairwise distance
        random_fallbacks: Number of random fallbacks
        rejected_by_tau: Number of rejections by τ
        csv_path: Path to CSV file
    """
    csv_file = Path(csv_path)
    
    # Create header if file doesn't exist
    if not csv_file.exists():
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'phase', 'class', 'backbone', 'k_config', 'd2_mode',
                'p', 'tau', 'bank_size', 'min_dist', 'median_dist',
                'max_dist', 'random_fallbacks', 'rejected_by_tau'
            ])
    
    # Append stats
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            phase, class_name, backbone, k_config, d2_mode,
            f"{p:.3f}", f"{tau:.4f}", bank_size,
            f"{min_dist:.4f}", f"{median_dist:.4f}", f"{max_dist:.4f}",
            random_fallbacks, rejected_by_tau
        ])

