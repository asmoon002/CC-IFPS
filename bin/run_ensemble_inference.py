#!/usr/bin/env python3
"""
Phase 2.19: Ensemble Inference
Load multiple trained models and average their anomaly maps
"""

import os
import sys
import glob
import pickle
import logging
import numpy as np
import torch
import click
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import patchcore.datasets.mvtec as mvtec
import patchcore.patchcore
import patchcore.metrics
import patchcore.common

LOGGER = logging.getLogger(__name__)


def load_patchcore_model(model_path, device):
    """Load a single PatchCore model"""
    patchcore_instance = patchcore.patchcore.PatchCore(device)
    patchcore_instance.load_from_path(
        load_path=model_path,
        device=device,
        nn_method=patchcore.common.FaissNN(True, 4)
    )
    return patchcore_instance


@click.command()
@click.option("--gpu", type=int, default=0)
@click.option("--class_name", type=str, required=True)
@click.option("--model_paths", type=str, multiple=True, required=True,
              help="Paths to ensemble models (e.g., Phase2.19_Ensemble_grid_tau_low/models/mvtec_grid)")
@click.option("--data_path", type=str, default="/userHome/userhome4/sehoon/patchcore/data")
@click.option("--output_path", type=str, required=True,
              help="Output path for ensemble results")
def main(gpu, class_name, model_paths, data_path, output_path):
    """
    Ensemble inference: Average anomaly maps from multiple models
    """
    logging.basicConfig(level=logging.INFO)
    LOGGER.info(f"Phase 2.19: Ensemble Inference for {class_name}")
    LOGGER.info(f"Loading {len(model_paths)} models...")
    
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load all models
    models = []
    for i, model_path in enumerate(model_paths):
        LOGGER.info(f"  Loading model {i+1}/{len(model_paths)}: {model_path}")
        try:
            model = load_patchcore_model(model_path, device)
            models.append(model)
        except Exception as e:
            LOGGER.error(f"  Failed to load {model_path}: {e}")
    
    if len(models) == 0:
        LOGGER.error("No models loaded! Exiting.")
        return
    
    LOGGER.info(f"Successfully loaded {len(models)} models")
    
    # Load test dataset
    LOGGER.info(f"Loading test dataset for {class_name}...")
    dataset = mvtec.MVTecDataset(
        source=data_path,
        classname=class_name,
        resize=256,
        imagesize=224,
        split=mvtec.DatasetSplit.TEST,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Run ensemble inference
    LOGGER.info("Running ensemble inference...")
    all_scores = []
    all_segmentations = []
    labels_gt = []
    masks_gt = []
    
    for image_data in dataloader:
        labels_gt.append(image_data["is_anomaly"].item())
        masks_gt.append(image_data["mask"].numpy()[0])
        image = image_data["image"]
        
        # Get predictions from all models
        model_scores = []
        model_segmentations = []
        
        for model in models:
            scores, segmentations = model._predict(image)
            model_scores.append(scores[0])
            model_segmentations.append(segmentations[0])
        
        # Average predictions
        ensemble_score = np.mean(model_scores)
        ensemble_segmentation = np.mean(model_segmentations, axis=0)
        
        all_scores.append(ensemble_score)
        all_segmentations.append(ensemble_segmentation)
    
    # Compute metrics
    LOGGER.info("Computing evaluation metrics...")
    
    # Image-level AUROC
    image_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
        all_scores, labels_gt
    )["auroc"]
    
    # Pixel-level metrics (all images)
    pixel_metrics = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        all_segmentations, masks_gt
    )
    
    # Pixel-level metrics (anomaly images only)
    sel_idxs = [i for i in range(len(masks_gt)) if np.sum(masks_gt[i]) > 0]
    anomaly_pixel_metrics = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        [all_segmentations[i] for i in sel_idxs],
        [masks_gt[i] for i in sel_idxs],
    )
    
    # Print results
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Ensemble Results for {class_name}")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"Instance AUROC: {image_auroc:.4f}")
    LOGGER.info(f"Full Pixel AUROC: {pixel_metrics['auroc']:.4f}")
    LOGGER.info(f"Full Pixel AP: {pixel_metrics['ap']:.4f}")
    LOGGER.info(f"Anomaly Pixel AUROC: {anomaly_pixel_metrics['auroc']:.4f}")
    LOGGER.info(f"Anomaly Pixel AP: {anomaly_pixel_metrics['ap']:.4f}")
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    
    import csv
    csv_path = os.path.join(output_path, "ensemble_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Row Names', 'Instance AUROC', 'Full Pixel AUROC', 'Full Pixel AP'])
        writer.writerow([
            f'mvtec_{class_name}',
            f'{image_auroc:.4f}',
            f'{pixel_metrics["auroc"]:.4f}',
            f'{pixel_metrics["ap"]:.4f}'
        ])
    
    LOGGER.info(f"\n✅ Results saved to: {csv_path}")


if __name__ == "__main__":
    main()

