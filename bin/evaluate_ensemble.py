#!/usr/bin/env python3
"""
Ensemble Evaluator for PatchCore
Loads multiple models (different τ values) and computes ensemble predictions.
"""

import contextlib
import logging
import os
import sys
import click
import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {
    "mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"],
}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="Ensemble_Evaluation")
@click.option("--log_project", type=str, default="project")
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["testing"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["testing"].name

        with device_context:
            torch.cuda.empty_cache()

            # Load ensemble models
            ensemble_models = methods["get_ensemble_models"](device)
            
            LOGGER.info(f"Loaded {len(ensemble_models)} models for ensemble")
            
            # Evaluate ensemble
            aggregator = {"scores": [], "segmentations": [], "labels_gt": [], "masks_gt": []}
            
            for image, mask, label in dataloaders["testing"]:
                if isinstance(image, dict):
                    image = image["image"]
                
                image = image.to(torch.float).to(device)
                
                # Get predictions from all models
                ensemble_scores = []
                ensemble_segmentations = []
                
                for model_idx, patchcore_model in enumerate(ensemble_models):
                    with torch.no_grad():
                        scores, segmentations, features = patchcore_model.predict(image)
                    
                    ensemble_scores.append(scores)
                    ensemble_segmentations.append(segmentations)
                
                # Average predictions
                avg_scores = np.mean(ensemble_scores, axis=0)
                avg_segmentations = np.mean(ensemble_segmentations, axis=0)
                
                aggregator["scores"].append(avg_scores)
                aggregator["segmentations"].append(avg_segmentations)
                aggregator["labels_gt"].extend(label.numpy().tolist())
                aggregator["masks_gt"].extend(mask.numpy().tolist())
            
            # Compute metrics
            scores = np.array(aggregator["scores"])
            segmentations = np.array(aggregator["segmentations"])
            labels_gt = np.array(aggregator["labels_gt"])
            masks_gt = np.array(aggregator["masks_gt"])
            
            # Flatten for pixel-level metrics
            scores = scores.flatten()
            segmentations = segmentations.flatten()
            masks_gt = masks_gt.flatten()
            
            # Image-level AUROC
            image_scores = np.max(segmentations.reshape(len(aggregator["labels_gt"]), -1), axis=1)
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                image_scores, labels_gt
            )["auroc"]
            
            # Pixel-level metrics
            pixel_metrics = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            
            # Full metrics
            full_pixel_auroc = pixel_metrics["auroc"]
            full_pixel_ap = pixel_metrics.get("ap", 0.0)
            full_pixel_precision = pixel_metrics.get("precision", 0.0)
            full_pixel_recall = pixel_metrics.get("recall", 0.0)
            full_pixel_f1 = pixel_metrics.get("f1_score", 0.0)
            
            # Anomaly-only metrics
            anomaly_mask = masks_gt > 0.5
            if anomaly_mask.sum() > 0:
                anomaly_pixel_metrics = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                    segmentations[anomaly_mask], masks_gt[anomaly_mask]
                )
                anomaly_pixel_auroc = anomaly_pixel_metrics["auroc"]
                anomaly_pixel_ap = anomaly_pixel_metrics.get("ap", 0.0)
                anomaly_pixel_precision = anomaly_pixel_metrics.get("precision", 0.0)
                anomaly_pixel_recall = anomaly_pixel_metrics.get("recall", 0.0)
                anomaly_pixel_f1 = anomaly_pixel_metrics.get("f1_score", 0.0)
            else:
                anomaly_pixel_auroc = 0.0
                anomaly_pixel_ap = 0.0
                anomaly_pixel_precision = 0.0
                anomaly_pixel_recall = 0.0
                anomaly_pixel_f1 = 0.0
            
            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "full_pixel_ap": full_pixel_ap,
                    "full_pixel_precision": full_pixel_precision,
                    "full_pixel_recall": full_pixel_recall,
                    "full_pixel_f1": full_pixel_f1,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                    "anomaly_pixel_ap": anomaly_pixel_ap,
                    "anomaly_pixel_precision": anomaly_pixel_precision,
                    "anomaly_pixel_recall": anomaly_pixel_recall,
                    "anomaly_pixel_f1": anomaly_pixel_f1,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{}: {:.4f}".format(key, item))

    # Store results
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]

    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("ensemble_models")
@click.option("--model_paths", "-m", type=str, multiple=True, required=True)
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=4)
def ensemble_models(model_paths, faiss_on_gpu, faiss_num_workers):
    def get_ensemble_models(device):
        """Load multiple PatchCore models for ensemble."""
        loaded_models = []
        
        for model_path in model_paths:
            LOGGER.info(f"Loading model from: {model_path}")
            
            # Load PatchCore model
            loaded_patchcore = patchcore.patchcore.PatchCore(device)
            loaded_patchcore.load_from_path(
                load_path=model_path,
                device=device,
                nn_method=patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers),
            )
            
            loaded_models.append(loaded_patchcore)
        
        return loaded_models
    
    return ("get_ensemble_models", get_ensemble_models)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=1, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split="test",
                seed=seed,
                augment=augment,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name + "_" + subdataset

            dataloader_dict = {"testing": test_dataloader}
            dataloaders.append(dataloader_dict)

        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()

