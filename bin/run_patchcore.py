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

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
@click.option("--inference_sigma", type=float, default=None, help="Override inference sigma for Gaussian smoothing")
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
    save_segmentation_images,
    save_patchcore_model,
    inference_sigma,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            
            # Extract class name (for logging only, not for parameter overrides)
            if hasattr(dataloaders["training"].dataset, 'classnames_to_use'):
                classnames = dataloaders["training"].dataset.classnames_to_use
                class_name = classnames[0] if classnames and len(classnames) == 1 else None
            else:
                class_name = None
            
            # Algorithm 1: Use CLI parameters directly (no class-specific overrides)
            # - multiscale_knn: Use CLI --multiscale_knn value
            # - inference_sigma: Use CLI --inference_sigma value (if provided)
            
            sampler = methods["get_sampler"](
                device,
                class_name=class_name,  # Pass for logging only
            )
            
            # Algorithm 1: Use CLI parameters directly (no override_k)
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            
            # Apply inference_sigma from CLI (if provided)
            if inference_sigma is not None:
                for PatchCore in PatchCore_list:
                    PatchCore.anomaly_segmentor.smoothing = inference_sigma
                LOGGER.info(f"Inference smoothing: sigma={inference_sigma} (CLI)")
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                PatchCore.fit(dataloaders["training"])

            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # (Optional) Plot example images.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    # Use IMAGENET normalization constants
                    in_std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
                    in_mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]

            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]
            full_pixel_ap = pixel_scores["ap"]
            full_pixel_precision = pixel_scores["precision"]
            full_pixel_recall = pixel_scores["recall"]
            full_pixel_f1 = pixel_scores["f1_score"]

            # Compute PRO score & PW Auroc only images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]
            anomaly_pixel_ap = pixel_scores["ap"]
            anomaly_pixel_precision = pixel_scores["precision"]
            anomaly_pixel_recall = pixel_scores["recall"]
            anomaly_pixel_f1 = pixel_scores["f1_score"]

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
                    "memory_bank_size": PatchCore_list[0].memory_bank_size if hasattr(PatchCore_list[0], 'memory_bank_size') else 0,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:.4f}".format(key, item))
            
            # Phase 2.17 (Package B): Log to unified CSV
            try:
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
                from patchcore.logging_utils import append_to_all_results
                
                # Determine phase from log_group
                phase_name = "unknown"
                if "Phase2.15b_Step1" in run_save_path:
                    phase_name = "2.15b_step1"
                elif "Phase2.15b_Step2" in run_save_path:
                    phase_name = "2.15b_step2"
                elif "Phase2.17_Postproc" in run_save_path:
                    phase_name = "2.17_postprocess"
                
                append_to_all_results(
                    phase=phase_name,
                    class_name=dataset_name.replace("mvtec_", ""),
                    backbone=backbone_names[0] if backbone_names else "wideresnet50",
                    k_config=str(multiscale_knn) if multiscale_knn else "1",
                    d2_mode="auto",  # Will be determined by sampler
                    d2_starting_points=10,  # Default
                    d2_exponent=2.0,  # Default
                    p=0.10,  # Approximate
                    tau=0.01,  # Will vary by class
                    postprocess=postprocess if 'postprocess' in locals() else 'none',
                    pixel_ap=full_pixel_ap,
                    image_auroc=auroc,
                    pixel_auroc=full_pixel_auroc
                )
            except Exception as e:
                LOGGER.debug(f"Failed to log to unified CSV: {e}")

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = [
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full Pixel AP",
        "Full Pixel Precision",
        "Full Pixel Recall",
        "Full Pixel F1",
        "Anomaly Pixel AUROC",
        "Anomaly Pixel AP",
        "Anomaly Pixel Precision",
        "Anomaly Pixel Recall",
        "Anomaly Pixel F1",
        "Memory Bank Size",
    ]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [
        [
            results["instance_auroc"],
            results["full_pixel_auroc"],
            results["full_pixel_ap"],
            results["full_pixel_precision"],
            results["full_pixel_recall"],
            results["full_pixel_f1"],
            results["anomaly_pixel_auroc"],
            results["anomaly_pixel_ap"],
            results["anomaly_pixel_precision"],
            results["anomaly_pixel_recall"],
            results["anomaly_pixel_f1"],
            results.get("memory_bank_size", 0),
        ]
        for results in result_collect
    ]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Phase 2.4: Multi-scale k-NN Ensemble
@click.option("--multiscale_knn", type=str, default=None, help="Phase 2.4: Multi-scale k-NN values (e.g., '1,3,9' for ensemble)")
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
# Phase 2: Density-weighted scoring
@click.option("--use_density_weighted_scoring", is_flag=True, default=False, help="Phase 2: Use density-weighted k-NN scoring")
# Phase 2.17: Anomaly map post-processing
@click.option("--postprocess", type=str, 
              default='none', show_default=True, help="Phase 2.17: Anomaly map post-processing method (none, gaussian3, gaussian5, median3)")
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    multiscale_knn,
    patchsize_aggregate,
    use_density_weighted_scoring,
    postprocess,
    faiss_on_gpu,
    faiss_num_workers,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            # Algorithm 1: Use CLI parameters directly (no override_k)
            if multiscale_knn is not None:
                # Parse multi-scale k-NN values from CLI
                k_values = [int(k.strip()) for k in multiscale_knn.split(',')]
            else:
                k_values = anomaly_scorer_num_nn

            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=k_values,
                nn_method=nn_method,
                use_density_weighted_scoring=use_density_weighted_scoring,
                postprocess=postprocess,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
@click.option(
    "--tau",
    type=float,
    default=0.1,
    show_default=True,
    help="Threshold for CC-IFPS irredundant filtering",
)
@click.option(
    "--max_memory_size",
    type=int,
    default=None,
    help="Maximum memory size for CC-IFPS (budget)",
)
@click.option(
    "--use_hybrid",
    is_flag=True,
    default=False,
    help="Use hybrid sampling (Coreset + τ-filtering) for CC-IFPS",
)
@click.option(
    "--sampling_type",
    type=str,
    default="greedy",
    show_default=True,
    help="Sampling strategy for CC-IFPS: 'greedy' or 'd2'",
)
def sampler(name, percentage, tau, max_memory_size, use_hybrid, sampling_type):
    """
    Sampler factory.

    Note:
        - For CC-IFPS, this exposes `sampling_type` so we can switch between
          pure greedy (deterministic) and D²-style probabilistic sampling,
          matching the design documented in CURRENT_STATUS_AND_OPTIMIZATION.md.
    """

    def get_sampler(device, class_name=None):
        if name == "identity":
            return patchcore.sampler.IdentitySampler(percentage, device)
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)
        elif name == "cc_ifps" or name == "ccifps":
            # Clean CC-IFPS initialization (Algorithm 1-based; Phase 2 heuristics off by default)
            # - τ: fixed constant (no adaptive scheduling)
            # - max_memory_size: budget B
            # - use_hybrid: enable Coreset + τ-filtering
            # - sampling_type: 'greedy' or 'd2' (D²-style probabilistic coreset)
            LOGGER.info(
                "Initializing CC-IFPS Sampler (clean): "
                f"class={class_name}, tau={tau}, budget={max_memory_size}, "
                f"hybrid={use_hybrid}, sampling_type={sampling_type}"
            )
            return patchcore.sampler.ClassConditionedIrredundantSampler(
                device=device,
                tau=tau,
                max_memory_size=max_memory_size,
                percentage=percentage,
                # Phase 2 heuristics disabled – Algorithm 1 + optional D² seeding via sampling_type.
                use_d2_seeding=False,
                d2_starting_points=10,
                d2_exponent=2.0,
                use_greedy_approx=False,
                use_anomaly_aware=False,
                use_hybrid=use_hybrid,
                use_reverse_hybrid=False,
                use_multiscale_density=False,
                class_name=class_name,
                sampling_type=sampling_type,
            )

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
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
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
