"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


def apply_postprocess(anomaly_map, method='none'):
    """
    Apply post-processing to anomaly map.
    
    Phase 2.17: Post-processing for Pixel AP improvement.
    
    Args:
        anomaly_map: numpy array [H, W]
        method: 'none', 'gaussian3', 'gaussian5', 'median3',
                or dynamic gaussian variants like 'gaussian4', 'gaussian8'
    
    Returns:
        Processed anomaly map [H, W]
    """
    if method == 'none':
        return anomaly_map
    
    # Ensure float32 for processing
    anomaly_map = anomaly_map.astype(np.float32)
    
    method_lower = method.lower()

    # --- Gaussian Smoothing ---
    # Support:
    #   - 'gaussian3', 'gaussian5' (backward-compatible, kernel-size based)
    #   - 'gaussian{sigma}' (e.g., gaussian4 -> sigma=4)
    if method_lower.startswith('gaussian'):
        # Backward-compatible fixed kernels
        if method_lower == 'gaussian3':
            return cv2.GaussianBlur(anomaly_map, (3, 3), 0)
        if method_lower == 'gaussian5':
            return cv2.GaussianBlur(anomaly_map, (5, 5), 0)

        # Dynamic sigma: parse trailing number after 'gaussian'
        sigma = 3.0  # sensible default
        suffix = method_lower.replace('gaussian', '')
        if suffix:
            try:
                sigma = float(suffix)
            except ValueError:
                # Fallback: keep default sigma
                pass

        # Use OpenCV's automatic kernel size selection with given sigma
        # (ksize=(0,0) lets OpenCV derive appropriate kernel from sigma)
        return cv2.GaussianBlur(anomaly_map, (0, 0), sigmaX=sigma)

    # --- Median Filtering ---
    if method_lower == 'median3':
        # Median filter requires uint8, so normalize
        map_min, map_max = anomaly_map.min(), anomaly_map.max()
        if map_max - map_min < 1e-8:
            return anomaly_map
        normalized = ((anomaly_map - map_min) / (map_max - map_min) * 255).astype(np.uint8)
        filtered = cv2.medianBlur(normalized, 3)
        # Convert back to original scale
        return filtered.astype(np.float32) / 255.0 * (map_max - map_min) + map_min
    
    # Fallback: unknown method, return original map
    return anomaly_map


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device
        self.postprocess = 'none'  # Phase 2.17: Default no postprocessing

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=None,
        nn_method=patchcore.common.FaissNN(False, 4),
        use_density_weighted_scoring=False,
        postprocess='none',
        **kwargs,
    ):
        # Create default IdentitySampler if not provided
        if featuresampler is None:
            featuresampler = patchcore.sampler.IdentitySampler(percentage=0.1, device=device)
        # Phase 2.17: Store postprocess method
        self.postprocess = postprocess
        if postprocess != 'none':
            LOGGER.info(f"Phase 2.17: Anomaly map postprocess = {postprocess}")
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, 
            nn_method=nn_method,
            use_density_weighted_scoring=use_density_weighted_scoring
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        
        # Log Memory Efficiency (Added)
        original_count = len(features)
        LOGGER.info(f"Full Feature Count (Before Sampling): {original_count}")

        features, selection_weights = self.featuresampler.run(features)
        
        # Store memory bank size
        self.memory_bank_size = len(features)
        
        # Log Reduction Ratio (Added)
        if original_count > 0:
            reduction_ratio = (1 - self.memory_bank_size / original_count) * 100
            LOGGER.info(f"Memory Efficiency: {original_count} -> {self.memory_bank_size} (Reduced by {reduction_ratio:.2f}%)")
        
        LOGGER.info(f"Memory Bank Size: {self.memory_bank_size} patches")
        
        # Phase 2: Compute and store patch densities if density-weighted scoring is enabled
        if (hasattr(self.featuresampler, 'use_multiscale_density') and 
            self.featuresampler.use_multiscale_density and
            self.anomaly_scorer.use_density_weighted_scoring):
            import torch.nn.functional as F
            features_tensor = torch.from_numpy(features).to(self.device)
            features_norm = F.normalize(features_tensor, p=2, dim=1)
            
            # Compute multi-scale density
            density_scores = self.featuresampler._compute_multiscale_density(features_norm)
            self.anomaly_scorer.nn_method.patch_densities = density_scores.cpu().numpy()

        self.anomaly_scorer.fit(detection_features=[features], detection_weights=[selection_weights])

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            # Phase 2.17: Apply postprocessing to patch_scores before segmentation
            if hasattr(self, 'postprocess') and self.postprocess != 'none':
                processed_scores = []
                for i in range(batchsize):
                    # Handle both torch.Tensor and numpy.ndarray
                    if isinstance(patch_scores, torch.Tensor):
                        score_map = patch_scores[i].cpu().numpy()
                    else:
                        score_map = patch_scores[i]
                    
                    processed_map = apply_postprocess(score_map, self.postprocess)
                    processed_scores.append(torch.from_numpy(processed_map).to(self.device))
                patch_scores = torch.stack(processed_scores)

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
