import abc
from typing import Union, Tuple
import logging

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

LOGGER = logging.getLogger(__name__)


def compute_memory_bank_stats(features, class_name, phase, **kwargs):
    """
    Compute and log memory bank statistics.
    
    Phase 2.17 (Package D): Debugging statistics for memory bank quality.
    
    Args:
        features: Selected feature patches [N, D] (torch.Tensor)
        class_name: Class name
        phase: Experiment phase
        **kwargs: Additional metadata (backbone, k_config, etc.)
    """
    try:
        from patchcore.logging_utils import append_memory_stats
    except ImportError:
        LOGGER.warning("logging_utils not available, skipping memory stats")
        return
    
    N = len(features)
    
    if N < 2:
        LOGGER.warning(f"Memory bank too small ({N} patches), skipping stats")
        return
    
    # Normalize for cosine distance
    features_norm = F.normalize(features, p=2, dim=1)
    
    # Compute pairwise cosine distances (sample if too large)
    if N > 5000:
        # Sample 5000 patches for efficiency
        sample_indices = torch.randperm(N, device=features.device)[:5000]
        features_sample = features_norm[sample_indices]
    else:
        features_sample = features_norm
    
    # Compute pairwise cosine similarity
    cos_sim = torch.mm(features_sample, features_sample.T)
    cos_dist = 1.0 - cos_sim
    
    # Get upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones_like(cos_dist), diagonal=1).bool()
    distances = cos_dist[mask].cpu().numpy()
    
    if len(distances) == 0:
        LOGGER.warning(f"No pairwise distances computed for {class_name}")
        return
    
    min_dist = float(np.min(distances))
    median_dist = float(np.median(distances))
    max_dist = float(np.max(distances))
    
    # Log statistics
    LOGGER.info(
        f"Memory bank stats [{class_name}]: size={N}, "
        f"dist=[min={min_dist:.4f}, median={median_dist:.4f}, max={max_dist:.4f}]"
    )
    
    # Save to CSV
    append_memory_stats(
        phase=phase,
        class_name=class_name,
        backbone=kwargs.get('backbone', 'wideresnet50'),
        k_config=kwargs.get('k_config', '1'),
        d2_mode=kwargs.get('d2_mode', 'none'),
        p=kwargs.get('p', 0.10),
        tau=kwargs.get('tau', 0.01),
        bank_size=N,
        min_dist=min_dist,
        median_dist=median_dist,
        max_dist=max_dist,
        random_fallbacks=kwargs.get('random_fallbacks', 0),
        rejected_by_tau=kwargs.get('rejected_by_tau', 0)
    )


class IdentitySampler:
    """
    Sampler that returns the input features as is.
    """
    def __init__(self, percentage: float, device: torch.device):
        self.percentage = percentage
        self.device = device
    
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)


class ProbabilisticCoresetSampler(GreedyCoresetSampler):
    """
    Probabilistic D² Coreset Sampler (k-means++ style) with Cosine Distance.
    
    Phase 2.15: Modified to use cosine distance instead of L2 distance
    for consistency with τ-filtering (Stage 2).
    
    Phase 2.26: Added density-aware sampling for better diversity control.
    
    Instead of deterministic farthest-first (argmax), this sampler uses
    probabilistic selection where P(point) ∝ D²(point, nearest_selected).
    
    This is more robust to outliers and noise, making it suitable for
    fine-grained object classes (screw, bottle, hazelnut, cable, capsule).
    """
    
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
        d2_exponent: float = 2.0,
        density_weight: float = 0.0,
    ):
        """
        Probabilistic D² Coreset sampling.
        
        Phase 2.15b: Added d2_exponent for adaptive diversity control.
        Phase 2.26: Added density_weight for density-aware sampling.
        
        Args:
            percentage: Sampling percentage
            device: Torch device
            number_of_starting_points: Number of anchor points for approximation
            dimension_to_project_features_to: PCA dimension
            d2_exponent: Exponent for distance-based probability (default: 2.0)
                - 1.0: Linear (soft, outlier-robust)
                - 2.0: Standard D² (k-means++)
                - 2.5: Aggressive (high diversity)
            density_weight: Weight for density-aware sampling (default: 0.0)
                - 0.0: Pure distance-based (original D²)
                - 0.3-0.5: Moderate density awareness (texture classes)
                - 0.1-0.2: Light density awareness (structural classes)
        """
        self.number_of_starting_points = number_of_starting_points
        self.d2_exponent = d2_exponent
        self.density_weight = density_weight
        super().__init__(percentage, device, dimension_to_project_features_to)
    
    @staticmethod
    def _compute_cosine_distances(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes batchwise cosine distances.
        
        Phase 2.15: Cosine distance for metric consistency with τ-filtering.
        
        Args:
            matrix_a: [N x D] feature matrix
            matrix_b: [M x D] feature matrix
            
        Returns:
            [N x M] cosine distance matrix (1 - cosine_similarity)
        """
        # Normalize features
        matrix_a_norm = F.normalize(matrix_a, p=2, dim=1)
        matrix_b_norm = F.normalize(matrix_b, p=2, dim=1)
        
        # Cosine similarity
        cos_sim = torch.mm(matrix_a_norm, matrix_b_norm.T)
        
        # Cosine distance: 1 - cos_sim
        return 1.0 - cos_sim
    
    def _compute_local_density(self, features: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Compute local density for each feature point.
        
        Phase 2.26: Density-aware sampling with memory-efficient implementation.
        
        Density is estimated as 1 / (1 + average distance to k nearest neighbors).
        Higher density = more neighbors nearby = lower diversity value.
        
        Args:
            features: [N x D] feature tensor
            k: Number of nearest neighbors for density estimation
            
        Returns:
            [N] density values (higher = denser region)
        """
        N = len(features)
        k = min(k, N - 1)
        
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Memory-efficient: Always use sampling approach to avoid OOM
        # Sample 1000 reference points for density estimation
        sample_size = min(1000, N)
        sample_indices = torch.randperm(N, device=features.device)[:sample_size]
        features_sample = features_norm[sample_indices]
        
        # Compute distances to sampled points in batches
        batch_size = 5000
        all_knn_distances = []
        
        for i in range(0, N, batch_size):
            batch_end = min(i + batch_size, N)
            batch_features = features_norm[i:batch_end]
            
            # Compute distances to sample points
            cos_sim = torch.mm(batch_features, features_sample.T)
            cos_dist = 1.0 - cos_sim
            
            # Get k nearest distances
            k_actual = min(k, sample_size)
            knn_distances, _ = torch.topk(cos_dist, k=k_actual, dim=1, largest=False)
            all_knn_distances.append(knn_distances)
        
        # Concatenate all batches
        knn_distances = torch.cat(all_knn_distances, dim=0)
        
        # Average k-NN distance
        avg_knn_dist = torch.mean(knn_distances, dim=1)
        
        # Density: 1 / (1 + avg_distance)
        # Higher density = lower avg_distance = higher value
        density = 1.0 / (1.0 + avg_knn_dist)
        
        return density
    
    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs probabilistic D² coreset selection (k-means++ style) with cosine distance.
        
        Phase 2.15: Modified to use cosine distance for metric consistency.
        Phase 2.26: Added density-aware sampling.
        
        Key difference from Greedy:
        - Greedy: select_idx = argmax(distances)  # Deterministic
        - D²:     select_idx ~ P(i) ∝ D²(i)      # Probabilistic
        - D² + Density: P(i) ∝ D²(i) × (1 + λ × (1 - density(i)))  # Favor low-density regions
        
        This reduces sensitivity to outliers and improves diversity.
        
        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()
        
        # Phase 2.26: Compute local density if density_weight > 0
        if self.density_weight > 0:
            local_density = self._compute_local_density(features, k=5)
            # Invert density: favor low-density (diverse) regions
            diversity_factor = 1.0 - local_density  # High diversity = low density
        else:
            diversity_factor = None
        
        # Phase 2.15: Use cosine distance instead of L2
        approximate_distance_matrix = self._compute_cosine_distances(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)
        
        with torch.no_grad():
            density_desc = f" + Density(λ={self.density_weight})" if self.density_weight > 0 else ""
            for _ in tqdm.tqdm(range(num_coreset_samples), desc=f"Subsampling (D^{self.d2_exponent}{density_desc})..."):
                # Phase 2.15b: Probabilistic selection with adaptive exponent
                # Phase 2.26: P(i) ∝ D^exponent(i) × (1 + λ × diversity(i))
                distances = approximate_coreset_anchor_distances.squeeze()
                
                # Phase 2.17: Enhanced numerical stability fix
                # Clamp distances to avoid negative values from numerical precision
                distances = torch.clamp(distances, min=1e-8, max=2.0)  # Cosine distance ∈ [0, 2]
                
                # Check for invalid distances before power operation
                if torch.isnan(distances).any() or torch.isinf(distances).any():
                    LOGGER.warning(f"Invalid distances detected, using random selection")
                    select_idx = np.random.choice(len(features))
                else:
                    distances_powered = torch.pow(distances, self.d2_exponent)
                    
                    # Phase 2.26: Apply density-aware weighting
                    if diversity_factor is not None:
                        # P(i) ∝ D^exp(i) × (1 + λ × diversity(i))
                        # diversity(i) = 1 - density(i), so low-density regions have higher probability
                        density_modifier = 1.0 + self.density_weight * diversity_factor
                        distances_powered = distances_powered * density_modifier
                    
                    # Avoid division by zero and NaN
                    if distances_powered.sum() < 1e-10 or torch.isnan(distances_powered).any() or torch.isinf(distances_powered).any():
                        # All points are very close or NaN detected, pick randomly
                        select_idx = np.random.choice(len(features))
                    else:
                        # Normalize to probabilities
                        probabilities = distances_powered / (distances_powered.sum() + 1e-10)
                        probabilities = probabilities.cpu().numpy()
                        
                        # Ensure non-negative and valid probabilities
                        probabilities = np.clip(probabilities, 0.0, 1.0)
                        prob_sum = probabilities.sum()
                        
                        # Check for valid probability distribution
                        if np.isnan(probabilities).any() or prob_sum < 1e-6:
                            select_idx = np.random.choice(len(features))
                        else:
                            probabilities = probabilities / prob_sum
                            # Final safety check
                            if np.any(probabilities < 0) or np.isnan(probabilities).any():
                                select_idx = np.random.choice(len(features))
                            else:
                                # Sample according to D^exponent × density distribution
                                select_idx = np.random.choice(len(features), p=probabilities)
                
                coreset_indices.append(select_idx)
                # Phase 2.15: Use cosine distance
                coreset_select_distance = self._compute_cosine_distances(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)
        
        return np.array(coreset_indices)


class ClassConditionedIrredundantSampler:
    """
    Class-Conditioned Irredundant Feature Patch Selection (CC-IFPS) Sampler.
    
    This sampler implements a two-stage approach:
    1. Irredundant Filtering: Remove redundant patches using cosine similarity threshold τ
    2. Budget Management: If filtered set exceeds budget, apply D^2 seeding
    
    Key differences from Greedy Coreset:
    - Uses class information for balanced sampling
    - Removes redundant patches (τ-based filtering)
    - Adaptive memory size based on data distribution
    """
    
    def __init__(
        self,
        device: torch.device,
        tau: float = 0.1,
        max_memory_size: int = None,
        percentage: float = None,
        use_d2_seeding: bool = True,
        d2_starting_points: int = 10,
        d2_exponent: float = 2.0,
        use_greedy_approx: bool = False,
        use_anomaly_aware: bool = False,
        use_hybrid: bool = False,
        use_reverse_hybrid: bool = False,
        use_multiscale_density: bool = False,
        class_name: str = None,
        sampling_type: str = 'greedy',
        dimension_to_project_features_to: int = 128,
    ):
        """
        Args:
            device: torch device for computation
            tau: Threshold for irredundant filtering (cosine distance)
                 Higher τ = more strict filtering = smaller memory
            max_memory_size: Maximum number of patches to keep (budget B)
                           If None, uses percentage * total_features
            percentage: Fallback percentage if max_memory_size is None
            use_d2_seeding: Whether to use D^2 seeding for budget management
            d2_starting_points: Number of starting points for D² sampling (Phase 2.15b)
            d2_exponent: Exponent for D² sampling probability (Phase 2.15b)
            use_greedy_approx: Whether to use approximate greedy (10x faster)
            use_anomaly_aware: Whether to use anomaly-aware adaptive τ
            use_hybrid: Whether to use hybrid sampling (Coreset + τ-filtering)
            use_reverse_hybrid: Whether to use reverse hybrid sampling (τ-filtering + Coreset) (Phase 2.3)
            use_multiscale_density: Whether to use multi-scale density adaptive τ (Phase 2)
            class_name: Class name for class-conditional τ scheduling (Phase 2)
            sampling_type: 'greedy' (deterministic) or 'd2' (probabilistic) for Coreset (Phase 2.11)
            dimension_to_project_features_to: Dimension for feature projection
        """
        self.device = device
        self.tau = tau
        self.max_memory_size = max_memory_size
        self.d2_starting_points = d2_starting_points
        self.d2_exponent = d2_exponent
        self.percentage = percentage
        self.use_d2_seeding = use_d2_seeding
        self.use_greedy_approx = use_greedy_approx
        self.use_anomaly_aware = use_anomaly_aware
        self.use_hybrid = use_hybrid
        self.use_reverse_hybrid = use_reverse_hybrid
        self.use_multiscale_density = use_multiscale_density
        self.class_name = class_name
        self.sampling_type = sampling_type
        self.dimension_to_project_features_to = dimension_to_project_features_to
        
        # Clean Algorithm 1: No class-specific overrides (Previously)
        # Phase 2.26: Class-Adaptive D2 Exponents (Restored for Optimization)
        # Defines optimal D2 exponents for each class to maximize diversity/stability
        CLASS_D2_EXPONENTS = {
            # Texture (Need high diversity -> Lower exponent for more uniform sampling)
            'grid': 1.6,
            'carpet': 1.7,
            'tile': 1.6,
            'leather': 1.5,
            'wood': 1.7,
            'zipper': 1.8,
            
            # Structural (Need boundary coverage -> Standard/High exponent)
            'transistor': 2.0,
            'metal_nut': 2.0,
            'pill': 2.0,
            'bottle': 2.0,
            
            # Fine-grained (Need outlier robustness -> Lower exponent)
            'screw': 1.4,
            'hazelnut': 1.5,
            'cable': 1.6,
            'toothbrush': 1.3,
            'capsule': 1.5,
        }
        
        if class_name in CLASS_D2_EXPONENTS:
            self.d2_exponent = CLASS_D2_EXPONENTS[class_name]
            LOGGER.info(f"Class-Adaptive D2: Overriding exponent to {self.d2_exponent} for {class_name}")
        else:
            self.d2_exponent = d2_exponent

        LOGGER.info(
            f"ClassConditionedIrredundantSampler initialized (Algorithm 1 + Adaptive D2): "
            f"tau={tau}, max_memory_size={max_memory_size}, "
            f"use_hybrid={use_hybrid}, class_name={class_name}, d2_exponent={self.d2_exponent}"
        )
    
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Main sampling method.
        
        Args:
            features: [N x D] feature tensor
            
        Returns:
            Sampled features and their corresponding selection weights (D2 distance)
        """
        # Phase 3: Return weights for density-aware scoring
        return_weights = True

        # Store original type
        features_is_numpy = isinstance(features, np.ndarray)
        if features_is_numpy:
            features_device = None
            features = torch.from_numpy(features)
        else:
            features_device = features.device
        
        features = features.to(self.device)
        
        # Determine budget
        if self.max_memory_size is None:
            if self.percentage is not None:
                budget = int(len(features) * self.percentage)
            else:
                budget = len(features)  # No budget limit
        else:
            budget = self.max_memory_size
        
        LOGGER.info(
            f"Starting CC-IFPS: {len(features)} patches, budget={budget}, tau={self.tau}"
        )
        
        # Algorithm 1: Hybrid mode only (Coreset + τ-filtering with immediate τ-check)
        # Note: Phase 2 features (reverse_hybrid, anomaly_aware, etc.) are disabled
        if self.use_hybrid:
            final_features, final_weights = self._hybrid_sampling(features, budget, sampling_type=self.sampling_type)
        else:
            # Fallback: Sequential irredundant filtering (not used in Algorithm 1)
            # This branch should not be reached with current configuration
            LOGGER.warning("use_hybrid=False: Using sequential filtering (not Algorithm 1)")
            if self.use_greedy_approx:
                irredundant_indices = self._irredundant_filtering_approx(features)
            else:
                irredundant_indices = self._irredundant_filtering_sequential(features)
            irredundant_features = features[irredundant_indices]
            
            # Budget management
            if len(irredundant_features) > budget:
                if self.use_d2_seeding:
                    final_indices, final_weights = self._d2_seeding(irredundant_features, budget)
                else:
                    final_indices = np.random.choice(
                        len(irredundant_features), budget, replace=False
                    )
                    final_weights = np.ones(len(final_indices), dtype=np.float32)
                final_features = irredundant_features[final_indices]
            else:
                final_features = irredundant_features
                final_weights = np.ones(len(final_features), dtype=np.float32)
        
        LOGGER.info(f"Final memory size: {len(final_features)} patches")
        
        
        # Restore original type
        if features_is_numpy:
            return final_features.cpu().numpy(), final_weights
        else:
            return final_features.to(features_device), final_weights
    
    def _irredundant_filtering(self, features: torch.Tensor) -> np.ndarray:
        """
        Stage 1: Greedy Irredundant Filtering using cosine distance threshold.
        
        Greedy selection algorithm:
        1. Select first patch deterministically (max feature norm)
        2. While candidates remain:
           - For each remaining patch, compute min distance to selected set M
           - Filter patches where min_distance > τ (valid candidates)
           - Among valid candidates, select the one with MAXIMUM distance (Greedy)
           - Add to M and repeat
        
        This combines:
        - Greedy Coreset's coverage optimization (always pick farthest)
        - τ-filtering's redundancy removal (only consider patches with dist > τ)
        
        Args:
            features: [N x D] feature tensor
            
        Returns:
            Indices of selected irredundant patches
        """
        N = len(features)
        selected_indices = []
        remaining_indices = list(range(N))
        
        # Normalize features for cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Select first patch deterministically: farthest from center in cosine space
        # Center in cosine space = mean direction
        center = torch.mean(features_norm, dim=0, keepdim=True)  # [1 x D]
        center_norm = F.normalize(center, p=2, dim=1)  # Normalize center direction
        
        # Cosine distance from center
        cos_sim_to_center = torch.mm(features_norm, center_norm.T).squeeze()  # [N]
        cos_dist_to_center = 1.0 - cos_sim_to_center  # [N]
        
        # Select patch farthest from center (most outlier in cosine space)
        first_idx = torch.argmax(cos_dist_to_center).item()
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Batch processing to avoid OOM
        batch_size = 10000  # Process 10k patches at a time
        
        with torch.no_grad():
            with tqdm.tqdm(total=N, desc="Greedy irredundant filtering") as pbar:
                pbar.update(1)  # First patch
                
                while len(remaining_indices) > 0:
                    selected_features = features_norm[selected_indices]  # [M x D]
                    
                    # Process remaining patches in batches to avoid OOM
                    min_distances_all = []
                    for batch_start in range(0, len(remaining_indices), batch_size):
                        batch_end = min(batch_start + batch_size, len(remaining_indices))
                        batch_indices = remaining_indices[batch_start:batch_end]
                        
                        # Compute distances for this batch
                        batch_features = features_norm[batch_indices]  # [B x D]
                        
                        # Cosine similarity: [B x M]
                        cos_sim = torch.mm(batch_features, selected_features.T)
                        
                        # Cosine distance: 1 - cos(f_i, m)
                        cos_dist = 1.0 - cos_sim  # [B x M]
                        
                        # Minimum distance for each patch in batch
                        batch_min_distances = torch.min(cos_dist, dim=1).values  # [B]
                        min_distances_all.append(batch_min_distances)
                    
                    # Concatenate all batch results
                    min_distances = torch.cat(min_distances_all)  # [R]
                    
                    # Filter: only consider patches with min_distance > τ
                    valid_mask = min_distances > self.tau
                    
                    if valid_mask.sum() == 0:
                        # No more valid candidates
                        break
                    
                    # Greedy: among valid candidates, select the one with MAXIMUM distance
                    valid_indices = torch.where(valid_mask)[0]
                    valid_distances = min_distances[valid_indices]
                    best_local_idx = torch.argmax(valid_distances).item()
                    best_global_idx = remaining_indices[valid_indices[best_local_idx].item()]
                    
                    selected_indices.append(best_global_idx)
                    remaining_indices.remove(best_global_idx)
                    
                    pbar.update(1)
                    if len(selected_indices) % 100 == 0:
                        pbar.set_postfix({"selected": len(selected_indices)})
        
        return np.array(selected_indices)
    
    def _irredundant_filtering_approx(self, features: torch.Tensor) -> np.ndarray:
        """
        Stage 1: Approximate Greedy Irredundant Filtering (10x faster).
        
        Approximate greedy selection:
        1. Select first patch deterministically (farthest from center in cosine space)
        2. Use 10% random subset as anchor candidates
        3. While candidates remain:
           - Compute min distance only for anchor subset
           - Filter anchors where min_distance > τ
           - Among valid anchors, select the one with MAXIMUM distance (Greedy)
           - Add to M and repeat
        
        Time complexity: O(N × M × 0.1) ≈ 10x faster than full greedy
        
        Args:
            features: [N x D] feature tensor
            
        Returns:
            Indices of selected irredundant patches
        """
        N = len(features)
        selected_indices = []
        
        # Normalize features for cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Select first patch deterministically: farthest from center in cosine space
        center = torch.mean(features_norm, dim=0, keepdim=True)  # [1 x D]
        center_norm = F.normalize(center, p=2, dim=1)
        cos_sim_to_center = torch.mm(features_norm, center_norm.T).squeeze()  # [N]
        cos_dist_to_center = 1.0 - cos_sim_to_center  # [N]
        first_idx = torch.argmax(cos_dist_to_center).item()
        selected_indices.append(first_idx)
        
        # Create anchor subset (10% of N)
        anchor_size = max(int(N * 0.1), 1000)  # At least 1000 anchors
        anchor_size = min(anchor_size, N - 1)  # Don't exceed N-1
        
        # Random sample anchors (excluding first_idx)
        remaining_pool = list(range(N))
        remaining_pool.remove(first_idx)
        anchor_indices = np.random.choice(remaining_pool, anchor_size, replace=False).tolist()
        
        # Batch processing to avoid OOM
        batch_size = 10000
        
        with torch.no_grad():
            with tqdm.tqdm(total=anchor_size, desc="Approximate greedy irredundant filtering") as pbar:
                pbar.update(0)  # First patch already selected
                
                while len(anchor_indices) > 0:
                    selected_features = features_norm[selected_indices]  # [M x D]
                    
                    # Process anchor patches in batches
                    min_distances_all = []
                    for batch_start in range(0, len(anchor_indices), batch_size):
                        batch_end = min(batch_start + batch_size, len(anchor_indices))
                        batch_anchor_indices = anchor_indices[batch_start:batch_end]
                        
                        # Compute distances for this batch of anchors
                        batch_features = features_norm[batch_anchor_indices]  # [B x D]
                        
                        # Cosine similarity: [B x M]
                        cos_sim = torch.mm(batch_features, selected_features.T)
                        
                        # Cosine distance: 1 - cos(f_i, m)
                        cos_dist = 1.0 - cos_sim  # [B x M]
                        
                        # Minimum distance for each patch in batch
                        batch_min_distances = torch.min(cos_dist, dim=1).values  # [B]
                        min_distances_all.append(batch_min_distances)
                    
                    # Concatenate all batch results
                    min_distances = torch.cat(min_distances_all)  # [A] where A = len(anchor_indices)
                    
                    # Filter: only consider patches with min_distance > τ
                    valid_mask = min_distances > self.tau
                    
                    if valid_mask.sum() == 0:
                        # No more valid candidates
                        break
                    
                    # Greedy: among valid anchors, select the one with MAXIMUM distance
                    valid_indices = torch.where(valid_mask)[0]
                    valid_distances = min_distances[valid_indices]
                    best_local_idx = torch.argmax(valid_distances).item()
                    best_global_idx = anchor_indices[valid_indices[best_local_idx].item()]
                    
                    selected_indices.append(best_global_idx)
                    anchor_indices.remove(best_global_idx)
                    
                    pbar.update(1)
                    if len(selected_indices) % 100 == 0:
                        pbar.set_postfix({"selected": len(selected_indices)})
        
        return np.array(selected_indices)
    
    def _anomaly_aware_filtering(self, features: torch.Tensor) -> np.ndarray:
        """
        Stage 1: Anomaly-Aware Irredundant Filtering with Adaptive τ.
        
        핵심 아이디어:
        - Outlier 패치 (anomaly 가능성 높음): 낮은 τ → 더 많이 보존
        - Normal 패치 (중심 근처): 높은 τ → 더 많이 필터링
        
        이를 통해 anomaly detection에 중요한 패치를 더 많이 보존하여
        Pixel AP를 개선합니다.
        
        Args:
            features: [N x D] feature tensor
            
        Returns:
            Indices of selected irredundant patches
        """
        N = len(features)
        selected_indices = []
        
        # Normalize features for cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)
        
        # 1. Compute outlier scores (distance from center)
        # Outlier = anomaly 가능성이 높은 패치
        center = torch.mean(features_norm, dim=0, keepdim=True)  # [1 x D]
        center_norm = F.normalize(center, p=2, dim=1)
        
        # Cosine distance from center
        cos_sim_to_center = torch.mm(features_norm, center_norm.T).squeeze()  # [N]
        outlier_scores = 1.0 - cos_sim_to_center  # [N]
        
        # 2. Compute adaptive τ for each patch
        tau_base = self.tau
        tau_min = tau_base * 0.3  # Outlier용 (더 낮은 τ = 더 많이 보존)
        tau_max = tau_base * 1.5  # Normal용 (더 높은 τ = 더 많이 필터링)
        
        # Normalize outlier scores to [0, 1]
        outlier_min = outlier_scores.min()
        outlier_max = outlier_scores.max()
        outlier_normalized = (outlier_scores - outlier_min) / (outlier_max - outlier_min + 1e-8)
        
        # Adaptive τ: High outlier score → Low τ
        adaptive_tau = tau_max - outlier_normalized * (tau_max - tau_min)
        
        # 3. Select first patch (highest outlier score = most anomalous)
        first_idx = torch.argmax(outlier_scores).item()
        selected_indices.append(first_idx)
        
        LOGGER.info(
            f"Anomaly-aware filtering: τ range [{tau_min:.4f}, {tau_max:.4f}], "
            f"outlier score range [{outlier_min:.4f}, {outlier_max:.4f}]"
        )
        
        # 4. Sequential filtering with adaptive τ
        with torch.no_grad():
            with tqdm.tqdm(total=N-1, desc="Anomaly-aware filtering") as pbar:
                for i in range(N):
                    if i == first_idx:
                        continue
                    
                    # Compute min distance to selected patches
                    current_feature = features_norm[i:i+1]  # [1 x D]
                    selected_features = features_norm[selected_indices]  # [M x D]
                    
                    # Cosine similarity
                    cos_sim = torch.mm(current_feature, selected_features.T)  # [1 x M]
                    
                    # Cosine distance
                    cos_dist = 1.0 - cos_sim  # [1 x M]
                    
                    # Minimum distance
                    min_distance = torch.min(cos_dist).item()
                    
                    # Use adaptive τ for this patch
                    if min_distance > adaptive_tau[i]:
                        selected_indices.append(i)
                    
                    pbar.update(1)
                    if (i + 1) % 1000 == 0:
                        pbar.set_postfix({"selected": len(selected_indices)})
        
        LOGGER.info(
            f"Anomaly-aware filtering: {len(selected_indices)} patches selected "
            f"({100*len(selected_indices)/N:.1f}% of original)"
        )
        
        return np.array(selected_indices)
    
    def _irredundant_filtering_sequential(self, features: torch.Tensor) -> np.ndarray:
        """
        Stage 1: Sequential Irredundant Filtering (original method).
        
        Sequential selection algorithm:
        1. Select first patch deterministically (farthest from center in cosine space)
        2. For each remaining feature f_i (in order):
           - Compute s(f_i, M) = min_{m ∈ M} (1 - cos(f_i, m))
           - If s(f_i, M) > τ, add f_i (sufficiently novel)
           - Else, skip f_i (redundant)
        
        Args:
            features: [N x D] feature tensor
            
        Returns:
            Indices of selected irredundant patches
        """
        N = len(features)
        selected_indices = []
        
        # Normalize features for cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Select first patch deterministically: farthest from center in cosine space
        center = torch.mean(features_norm, dim=0, keepdim=True)  # [1 x D]
        center_norm = F.normalize(center, p=2, dim=1)
        cos_sim_to_center = torch.mm(features_norm, center_norm.T).squeeze()  # [N]
        cos_dist_to_center = 1.0 - cos_sim_to_center  # [N]
        first_idx = torch.argmax(cos_dist_to_center).item()
        selected_indices.append(first_idx)
        
        with torch.no_grad():
            with tqdm.tqdm(total=N-1, desc="Sequential irredundant filtering") as pbar:
                for i in range(N):
                    if i == first_idx:
                        continue  # Skip first patch (already selected)
                    
                    # Compute cosine similarity with all selected patches
                    current_feature = features_norm[i:i+1]  # [1 x D]
                    selected_features = features_norm[selected_indices]  # [M x D]
                    
                    # Cosine similarity: cos(f_i, m) = f_i · m (already normalized)
                    cos_sim = torch.mm(current_feature, selected_features.T)  # [1 x M]
                    
                    # Cosine distance: 1 - cos(f_i, m)
                    cos_dist = 1.0 - cos_sim  # [1 x M]
                    
                    # Minimum distance to any selected patch
                    min_dist = torch.min(cos_dist).item()
                    
                    # Add if sufficiently novel
                    if min_dist > self.tau:
                        selected_indices.append(i)
                    
                    pbar.update(1)
                    if (i + 1) % 1000 == 0:
                        pbar.set_postfix({"selected": len(selected_indices)})
        
        return np.array(selected_indices)
    
    def _compute_multiscale_density(self, features_norm: torch.Tensor, k_values: list = [3, 5, 9, 15]) -> torch.Tensor:
        """
        Phase 2.1 기능 1: Multi-scale density 계산 (Optimized k values)
        
        여러 k 값으로 local density를 계산하고 평균을 취함.
        ✅ OPTIMIZED: k = [3, 5, 9, 15] (이전: [5, 10, 20, 50])
        - 메모리 크기가 ~6000개로 줄었으므로 더 local한 k 값 사용
        - k=50은 전체의 8.3%로 너무 넓어 변별력 저하
        
        Dense 영역 (많은 이웃) → 높은 density → 높은 τ (더 많이 필터링)
        Sparse 영역 (적은 이웃) → 낮은 density → 낮은 τ (더 많이 보존)
        
        Args:
            features_norm: [N x D] normalized features
            k_values: List of k values for multi-scale density
            
        Returns:
            density_scores: [N] density score for each patch (0~1)
        """
        N = len(features_norm)
        device = features_norm.device
        
        # Compute pairwise cosine distances
        cos_sim = torch.mm(features_norm, features_norm.T)  # [N x N]
        cos_dist = 1.0 - cos_sim  # [N x N]
        
        # For each k, compute average distance to k-nearest neighbors
        density_scores_list = []
        
        for k in k_values:
            k_actual = min(k, N - 1)  # Handle small N
            
            # Get k-nearest distances for each patch
            k_nearest_dists, _ = torch.topk(cos_dist, k_actual + 1, dim=1, largest=False)
            # Exclude self (distance 0)
            k_nearest_dists = k_nearest_dists[:, 1:]  # [N x k]
            
            # Average distance to k-nearest neighbors
            avg_k_dist = torch.mean(k_nearest_dists, dim=1)  # [N]
            
            # Convert to density: higher distance → lower density
            # Normalize to [0, 1]
            density_k = 1.0 - (avg_k_dist - avg_k_dist.min()) / (avg_k_dist.max() - avg_k_dist.min() + 1e-8)
            density_scores_list.append(density_k)
        
        # Average across all scales
        density_scores = torch.stack(density_scores_list).mean(dim=0)  # [N]
        
        return density_scores
    
    def _hybrid_sampling(self, features: torch.Tensor, budget: int, sampling_type: str = 'greedy') -> Tuple[torch.Tensor, np.ndarray]:
        """
        PROPOSED METHOD: Class-Conditioned Irredundant Feature Patch Selection.
        Implements Algorithm 1 from the paper strictly.
        
        Core Logic (논문 Algorithm 1):
        - Iteratively select the farthest feature (Greedy Coreset strategy).
        - STOP or SKIP if the distance to the nearest neighbor is <= τ.
        - Uses 'self.tau' as a fixed constant (No adaptive scheduling).
        - No forced ratio between stages - τ naturally determines the selection.
        
        핵심 변경사항:
        - 이전: 2-Stage 비율 강제 (70% Coreset + 30% τ-filtering)
        - 현재: Single-pass with 즉시 τ-check (논문 Algorithm 1)
        - Budget은 상한선으로만 사용, τ에 의해 자연스럽게 결정됨
        
        Args:
            features: [N x D] feature tensor
            budget: Total budget (B) - upper bound only
            sampling_type: 'greedy' (deterministic) or 'd2' (probabilistic, not used in Algorithm 1)
            
        Returns:
            Final sampled features [<=B x D]
        """
        N = len(features)
        
        # Algorithm 1: Single-pass with immediate τ-check
        # No forced stage ratios - τ naturally determines the selection
        LOGGER.info(f"Algorithm 1 Hybrid Sampling: tau={self.tau:.4f}, budget={budget} (upper bound)")
        
        # 1. Random Projection for efficiency (if needed)
        if self.dimension_to_project_features_to < features.shape[1]:
            mapper = torch.nn.Linear(
                features.shape[1], 
                self.dimension_to_project_features_to, 
                bias=False
            ).to(self.device)
            with torch.no_grad():
                reduced_features = mapper(features)
        else:
            reduced_features = features
        
        # Ensure features are L2 normalized for cosine distance calculation
        features_norm = F.normalize(features, p=2, dim=1)
        reduced_features_norm = F.normalize(reduced_features, p=2, dim=1)
        
        # Start with 1 random point (or farthest from center for determinism)
        # Using farthest from center for better initialization
        center = torch.mean(features_norm, dim=0, keepdim=True)
        center_norm = F.normalize(center, p=2, dim=1)
        cos_sim_to_center = torch.mm(features_norm, center_norm.T).squeeze()
        cos_dist_to_center = 1.0 - cos_sim_to_center
        current_selection_idx = torch.argmax(cos_dist_to_center).item()
        
        selected_indices = [current_selection_idx]
        selected_weights = [cos_dist_to_center[current_selection_idx].item()]
        
        # Calculate threshold in terms of Squared Euclidean Distance
        # Note: Features are L2 normalized, so:
        # ||a-b||² = ||a||² + ||b||² - 2a.b = 2 - 2*cos(θ) = 2*(1-cos(θ))
        # Cosine Distance = 1 - cos(θ) = (Euclidean²) / 2
        # If paper condition is: CosineDist > τ
        # Then: (Euclidean²) / 2 > τ  =>  Euclidean² > 2 * τ
        threshold_sq = 2.0 * self.tau
        
        # Helper function to compute squared Euclidean distance for L2-normalized features
        def compute_squared_euclidean_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Compute squared Euclidean distance for L2-normalized features.
            For L2-normalized vectors: ||a-b||² = 2 - 2*a·b = 2*(1-cos(θ))
            """
            # Direct computation: squared distance = 2 - 2*dot_product
            # For batch computation: a [N x D], b [M x D] -> [N x M]
            dot_product = torch.mm(a, b.T)  # [N x M]
            squared_dist = 2.0 - 2.0 * dot_product  # [N x M]
            return squared_dist.clamp(min=0.0)
        
        # Initialize min_distances (Squared Euclidean distance from all points to the first selected point)
        dist_to_first = compute_squared_euclidean_dist(
            reduced_features_norm, 
            reduced_features_norm[current_selection_idx:current_selection_idx+1]
        ).squeeze()  # [N]
        
        min_distances = dist_to_first
        
        # Budget constraint (Upper bound)
        max_patches = budget if budget > 0 else int(N * (self.percentage if self.percentage else 0.1))
        
        # 2. Iterative Selection with Irredundancy Check (Algorithm 1)
        # We loop until we fill the budget OR we run out of diverse features
        
        # Create a progress bar
        pbar = tqdm.tqdm(total=max_patches, desc=f"Algorithm 1 Hybrid (tau={self.tau:.4f})")
        pbar.update(1)  # Count the first point
        
        with torch.no_grad():
            for _ in range(max_patches - 1):
                # Current distance to nearest neighbour for all candidates.
                max_min_dist = min_distances.max()
                
                # --- STRICT ALGORITHM 1 CHECK ---
                # If the *farthest* candidate is already closer than the threshold,
                # then ALL candidates are within τ → 더 이상 새로운 정보를 주지 못하므로 중단.
                if max_min_dist < threshold_sq:
                    LOGGER.info(
                        f"Early stopping: max_min_dist ({max_min_dist:.6f}) < "
                        f"threshold ({threshold_sq:.6f})"
                    )
                    break
                
                # --- Selection Strategy: greedy vs d2 ---
                if sampling_type == "d2":
                    # Probabilistic D²-style sampling:
                    # 다음 후보를 min_distances^exponent 에 비례해서 뽑는다.
                    # D² sampling의 핵심: 거리의 제곱에 비례 → diversity 강화
                    # Phase 2.26: Use adaptive exponent
                    squared_distances = min_distances ** self.d2_exponent
                    total_sq = float(squared_distances.sum())
                    if total_sq <= 0.0:
                        # 모든 거리가 0이면 더 이상 선택할 의미가 없음.
                        break
                    probs = (squared_distances / total_sq).clamp(min=0.0)
                    best_candidate_idx = torch.multinomial(probs, 1).item()
                else:
                    # Greedy: 가장 멀리 있는 점 선택 (기존 Algorithm 1 동작)
                    best_candidate_idx = torch.argmax(min_distances).item()
                
                # Add to memory
                selected_indices.append(best_candidate_idx)
                selected_weights.append(min_distances[best_candidate_idx].item())
                
                # Update distances: New dist = min(old_dist, dist_to_newly_selected)
                # Use squared Euclidean distance
                dist_to_new = compute_squared_euclidean_dist(
                    reduced_features_norm, 
                    reduced_features_norm[best_candidate_idx:best_candidate_idx+1]
                ).squeeze()  # [N]
                
                min_distances = torch.minimum(min_distances, dist_to_new)
                pbar.update(1)
        
        pbar.close()
        
        # Return features from original feature space (not reduced)
        final_features = features[selected_indices]
        
        LOGGER.info(
            f"Algorithm 1 Hybrid Sampling complete: "
            f"Selected {len(final_features)} / {N} features "
            f"(tau={self.tau:.4f}, budget={budget}, utilization={100*len(final_features)/budget:.1f}%)"
        )
        
        # Phase 2.17 (Package D): Compute memory bank statistics
        try:
            compute_memory_bank_stats(
                features=final_features,
                class_name=self.class_name or "unknown",
                phase="algorithm1_hybrid",
                backbone="wideresnet50",
                k_config="auto",
                d2_mode=sampling_type,
                p=len(final_features) / N,
                tau=self.tau,
                random_fallbacks=0,
                rejected_by_tau=0
            )
        except Exception as e:
            LOGGER.debug(f"Failed to compute memory bank stats: {e}")
        
        return final_features, np.array(selected_weights)
    
    def _reverse_hybrid_sampling(self, features: torch.Tensor, budget: int) -> torch.Tensor:
        """
        Reverse Hybrid Sampling (Phase 2.3): τ-filtering → Greedy Coreset
        
        핵심 아이디어:
        - Stage 1: Adaptive τ-filtering (전체 패치에서 중복 제거 = Denoising)
        - Stage 2: Approximate Greedy Coreset (정제된 후보군에서 coverage 최적화)
        
        장점:
        1. 노이즈/중복 제거 우선 → 깨끗한 후보군 확보
        2. Coverage 최적화 후순위 → 고품질 패치만으로 대표성 확보
        3. Carpet 같은 Natural Texture 클래스에 유리
        
        Args:
            features: [N x D] feature tensor
            budget: Total budget (B)
            
        Returns:
            Final sampled features [B x D]
        """
        N = len(features)
        
        # NOTE: This method is disabled for Algorithm 1 (use_reverse_hybrid=False)
        # Phase 2 기능 2: Class-conditional τ scheduling (REMOVED for Algorithm 1)
        # - class_tau_multipliers was removed for clean Algorithm 1 implementation
        tau_base = self.tau  # Use input τ directly (no multipliers)
        
        # Stage 1: Adaptive τ-filtering (전체 패치 대상)
        LOGGER.info(f"Reverse Hybrid Stage 1: Adaptive τ-filtering (전체 {N} patches)")
        
        # Normalize features for cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Phase 2 기능 1: Multi-scale density adaptive τ
        # NOTE: Disabled for Reverse Hybrid due to memory constraints
        adaptive_tau = None
        if self.use_multiscale_density:
            LOGGER.warning(
                "Reverse Hybrid: Skipping multi-scale density due to memory constraints. "
                "Using static/class-conditional tau."
            )
            # Skip _compute_multiscale_density to avoid CUDA OOM
            # Calculating N x N density matrix for full training set (N~200k) requires ~160GB
            # adaptive_tau remains None, so filtering will use tau_base
        
        # Phase 2.15: Random shuffle to prevent data bias
        # Without shuffle, early stopping may only sample from first few images
        perm = torch.randperm(N, device=features_norm.device)
        features_norm_shuffled = features_norm[perm]
        if adaptive_tau is not None:
            adaptive_tau_shuffled = adaptive_tau[perm]
        else:
            adaptive_tau_shuffled = None
        
        # Select first patch deterministically: farthest from center
        center = torch.mean(features_norm_shuffled, dim=0, keepdim=True)
        center_norm = F.normalize(center, p=2, dim=1)
        cos_sim_to_center = torch.mm(features_norm_shuffled, center_norm.T).squeeze()
        cos_dist_to_center = 1.0 - cos_sim_to_center
        first_idx = torch.argmax(cos_dist_to_center).item()
        
        selected_indices_shuffled = [first_idx]
        
        # Sequential filtering with τ
        with torch.no_grad():
            with tqdm.tqdm(total=N-1, desc="Reverse Hybrid Stage 1: τ-filtering") as pbar:
                for i in range(N):
                    if i == first_idx:
                        continue
                    
                    # Compute min distance to selected patches
                    current_feature = features_norm_shuffled[i:i+1]
                    selected_features = features_norm_shuffled[selected_indices_shuffled]
                    cos_sim = torch.mm(current_feature, selected_features.T)
                    cos_dist = 1.0 - cos_sim
                    min_distance = torch.min(cos_dist).item()
                    
                    # Apply adaptive τ threshold (Phase 2)
                    tau_threshold = adaptive_tau_shuffled[i].item() if adaptive_tau_shuffled is not None else tau_base
                    if min_distance > tau_threshold:
                        selected_indices_shuffled.append(i)
                    
                    pbar.update(1)
                    if (i + 1) % 1000 == 0:
                        pbar.set_postfix({"selected": len(selected_indices_shuffled)})
        
        # Map shuffled indices back to original indices
        original_indices = perm[selected_indices_shuffled].cpu().numpy()
        stage1_features = features[original_indices]
        
        LOGGER.info(
            f"Stage 1 complete: {len(stage1_features)} patches "
            f"({100*len(stage1_features)/N:.1f}% of original)"
        )
        
        # Stage 2: Approximate Greedy Coreset (정제된 후보군에서 최종 선택)
        if len(stage1_features) <= budget:
            # 이미 budget 이하면 그대로 사용
            LOGGER.info(f"Stage 1 output already within budget. Skipping Stage 2.")
            final_features = stage1_features
        else:
            # Greedy Coreset으로 budget까지 축소
            LOGGER.info(f"Reverse Hybrid Stage 2: Approximate Greedy Coreset ({budget} patches)")
            
            coreset_sampler = ApproximateGreedyCoresetSampler(
                percentage=budget / len(stage1_features),
                device=self.device,
                dimension_to_project_features_to=self.dimension_to_project_features_to,
            )
            
            final_features = coreset_sampler.run(stage1_features)
            
            LOGGER.info(
                f"Stage 2 complete: {len(final_features)} patches "
                f"({100*len(final_features)/len(stage1_features):.1f}% of Stage 1)"
            )
        
        LOGGER.info(f"Reverse Hybrid sampling complete: {len(final_features)} patches")
        
        return final_features

    def _d2_seeding(self, features: torch.Tensor, budget: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 2: D^2 seeding for budget management (cosine distance based).
        
        k-means++ style sampling:
        1. Select first patch deterministically (farthest from center in cosine space)
        2. For each subsequent patch:
           - Compute D(z) = min cosine distance to already selected patches
           - Sample deterministically: argmax(D(z)^2)
        
        Args:
            features: [N x D] irredundant features
            budget: Target number of patches
            
        Returns:
            Indices of selected patches
        """
        N = len(features)
        if budget >= N:
            return np.arange(N)
        
        selected_indices = []
        remaining_indices = list(range(N))
        
        # Normalize features for cosine distance
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Select first patch deterministically: farthest from center in cosine space
        center = torch.mean(features_norm, dim=0, keepdim=True)  # [1 x D]
        center_norm = F.normalize(center, p=2, dim=1)
        cos_sim_to_center = torch.mm(features_norm, center_norm.T).squeeze()  # [N]
        cos_dist_to_center = 1.0 - cos_sim_to_center  # [N]
        first_idx = torch.argmax(cos_dist_to_center).item()
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        with torch.no_grad():
            with tqdm.tqdm(total=budget-1, desc="D^2 seeding (cosine)") as pbar:
                for _ in range(budget - 1):
                    if len(remaining_indices) == 0:
                        break
                    
                    # Compute cosine distances from remaining patches to selected patches
                    remaining_features_norm = features_norm[remaining_indices]  # [R x D]
                    selected_features_norm = features_norm[selected_indices]  # [S x D]
                    
                    # Cosine similarity: [R x S]
                    cos_sim = torch.mm(remaining_features_norm, selected_features_norm.T)
                    
                    # Cosine distance: 1 - cos(remaining[i], selected[j])
                    distances = 1.0 - cos_sim  # [R x S]
                    
                    # Minimum distance for each remaining patch
                    min_distances = torch.min(distances, dim=1).values  # [R]
                    
                    # D^2 weighting
                    d2_weights = min_distances ** 2
                    
                    # Deterministic selection: always pick the patch with maximum D^2
                    # This reduces variance and improves reproducibility
                    # while maintaining the k-means++ coverage guarantee
                    sampled_idx = torch.argmax(d2_weights).item()
                    selected_patch_idx = remaining_indices[sampled_idx]
                    
                    selected_indices.append(selected_patch_idx)
                    remaining_indices.pop(sampled_idx)
                    
                    pbar.update(1)
        
        return np.array(selected_indices)


class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Randomly samples input feature collection.

        Args:
            features: [N x D]
        """
        num_random_samples = int(len(features) * self.percentage)
        subset_indices = np.random.choice(
            len(features), num_random_samples, replace=False
        )
        subset_indices = np.array(subset_indices)
        return features[subset_indices]
