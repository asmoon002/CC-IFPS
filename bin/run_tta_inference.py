#!/usr/bin/env python3
"""
Phase 2.20: Test-Time Augmentation (TTA)
Apply multiple augmentations and average predictions
"""

import os
import sys
import logging
import numpy as np
import torch
import click
import torchvision.transforms.functional as TF

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import patchcore.datasets.mvtec as mvtec
import patchcore.patchcore
import patchcore.metrics
import patchcore.common

LOGGER = logging.getLogger(__name__)


def apply_tta_augmentations(image):
    """
    Apply TTA augmentations.
    
    Returns:
        List of (augmented_image, reverse_transform_fn)
    """
    augmentations = []
    
    # 1. Original
    augmentations.append((image, lambda x: x))
    
    # 2. Horizontal flip
    augmentations.append((
        TF.hflip(image),
        lambda x: TF.hflip(x)
    ))
    
    # 3. Vertical flip
    augmentations.append((
        TF.vflip(image),
        lambda x: TF.vflip(x)
    ))
    
    # 4. Rotate 90
    augmentations.append((
        TF.rotate(image, 90),
        lambda x: TF.rotate(x, -90)
    ))
    
    # 5. Rotate 180
    augmentations.append((
        TF.rotate(image, 180),
        lambda x: TF.rotate(x, 180)
    ))
    
    # 6. Rotate 270
    augmentations.append((
        TF.rotate(image, 270),
        lambda x: TF.rotate(x, -270)
    ))
    
    return augmentations


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
@click.option("--model_path", type=str, required=True,
              help="Path to trained model directory")
@click.option("--dataset_path", type=str, default="/userHome/userhome4/sehoon/patchcore/data")
@click.option("--output_dir", type=str, required=True,
              help="Output directory for TTA results")
@click.option("--num_augmentations", type=int, default=6,
              help="Number of TTA augmentations (1=original, 6=all)")
@click.option("--multiscale_knn", type=str, default="1,3,5,9",
              help="Multi-scale k values for ensemble")
@click.option("--result_score_sigma", type=int, default=4,
              help="Gaussian smoothing sigma")
@click.option("--use_density_weighted_scoring", is_flag=True,
              help="Use density-weighted scoring")
def main(gpu, class_name, model_path, dataset_path, output_dir, num_augmentations,
         multiscale_knn, result_score_sigma, use_density_weighted_scoring):
    """
    TTA inference: Average predictions from multiple augmentations
    """
    logging.basicConfig(level=logging.INFO)
    LOGGER.info(f"Phase 2.24: TTA Inference for {class_name}")
    LOGGER.info(f"Using {num_augmentations} augmentations")
    LOGGER.info(f"Multi-scale k: {multiscale_knn}")
    LOGGER.info(f"Smoothing sigma: {result_score_sigma}")
    LOGGER.info(f"Density weighting: {use_density_weighted_scoring}")
    
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    # Find model file
    model_file = None
    if os.path.isdir(model_path):
        models_dir = os.path.join(model_path, "models")
        if os.path.isdir(models_dir):
            for fname in os.listdir(models_dir):
                if fname.startswith("mvtec_") and not fname.endswith(".txt"):
                    model_file = os.path.join(models_dir, fname)
                    break
    
    if model_file is None:
        LOGGER.error(f"Could not find model in {model_path}")
        sys.exit(1)
    
    # Load model
    LOGGER.info(f"Loading model: {model_file}")
    model = load_patchcore_model(model_file, device)
    
    # Load test dataset
    LOGGER.info(f"Loading test dataset for {class_name}...")
    dataset = mvtec.MVTecDataset(
        source=dataset_path,
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
    
    # Run TTA inference
    LOGGER.info("Running TTA inference...")
    all_scores = []
    all_segmentations = []
    labels_gt = []
    masks_gt = []
    
    for image_data in dataloader:
        labels_gt.append(image_data["is_anomaly"].item())
        masks_gt.append(image_data["mask"].numpy()[0])
        image = image_data["image"]
        
        # Get augmentations
        augmentations = apply_tta_augmentations(image)[:num_augmentations]
        
        # Get predictions from all augmentations
        aug_scores = []
        aug_segmentations = []
        
        for aug_image, reverse_fn in augmentations:
            scores, segmentations = model._predict(aug_image)
            
            # Reverse transform the segmentation
            seg_tensor = torch.from_numpy(segmentations[0]).unsqueeze(0).unsqueeze(0)
            reversed_seg = reverse_fn(seg_tensor).squeeze().numpy()
            
            aug_scores.append(scores[0])
            aug_segmentations.append(reversed_seg)
        
        # Average predictions
        tta_score = np.mean(aug_scores)
        tta_segmentation = np.mean(aug_segmentations, axis=0)
        
        all_scores.append(tta_score)
        all_segmentations.append(tta_segmentation)
    
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
    LOGGER.info(f"TTA Results for {class_name} ({num_augmentations} augmentations)")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"Instance AUROC: {image_auroc:.4f}")
    LOGGER.info(f"Full Pixel AUROC: {pixel_metrics['auroc']:.4f}")
    LOGGER.info(f"Full Pixel AP: {pixel_metrics['ap']:.4f}")
    LOGGER.info(f"Anomaly Pixel AUROC: {anomaly_pixel_metrics['auroc']:.4f}")
    LOGGER.info(f"Anomaly Pixel AP: {anomaly_pixel_metrics['ap']:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    import csv
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'instance_auroc', 'full_pixel_auroc', 'full_pixel_ap', 
                        'anomaly_pixel_auroc', 'anomaly_pixel_ap'])
        writer.writerow([
            class_name,
            f'{image_auroc:.4f}',
            f'{pixel_metrics["auroc"]:.4f}',
            f'{pixel_metrics["ap"]:.4f}',
            f'{anomaly_pixel_metrics["auroc"]:.4f}',
            f'{anomaly_pixel_metrics["ap"]:.4f}'
        ])
    
    LOGGER.info(f"\n✅ Results saved to: {csv_path}")


if __name__ == "__main__":
    main()

