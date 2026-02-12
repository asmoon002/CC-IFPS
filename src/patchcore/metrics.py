"""Anomaly metrics."""
import numpy as np
from sklearn import metrics


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, AP, Precision, Recall, F1-score) 
    for anomaly segmentations and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    
    Returns:
        Dictionary containing:
        - auroc: Area Under ROC Curve
        - ap: Average Precision (Pixel AP)
        - precision: Precision at optimal F1 threshold
        - recall: Recall at optimal F1 threshold
        - f1_score: F1-score at optimal threshold
        - fpr, tpr: False/True Positive Rates for ROC curve
        - optimal_threshold: Threshold that maximizes F1-score
        - optimal_fpr, optimal_fnr: FPR and FNR at optimal threshold
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    
    # Compute Average Precision (AP)
    ap = metrics.average_precision_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    
    # Compute metrics at optimal threshold
    optimal_precision = precision[np.argmax(F1_scores)]
    optimal_recall = recall[np.argmax(F1_scores)]
    optimal_f1 = F1_scores[np.argmax(F1_scores)]
    
    # Also compute using sklearn's classification_report style metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    binary_predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    binary_ground_truth = flat_ground_truth_masks.astype(int)
    
    # Handle case where there are no positive predictions
    if np.sum(binary_predictions) == 0:
        sklearn_precision = 0.0
        sklearn_recall = 0.0
        sklearn_f1 = 0.0
    else:
        sklearn_precision = precision_score(binary_ground_truth, binary_predictions, zero_division=0)
        sklearn_recall = recall_score(binary_ground_truth, binary_predictions, zero_division=0)
        sklearn_f1 = f1_score(binary_ground_truth, binary_predictions, zero_division=0)
    
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "ap": ap,  # Average Precision (Pixel AP)
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
        "precision": sklearn_precision,  # Precision at optimal threshold
        "recall": sklearn_recall,  # Recall at optimal threshold
        "f1_score": sklearn_f1,  # F1-score at optimal threshold
        "optimal_precision": optimal_precision,  # From PR curve
        "optimal_recall": optimal_recall,  # From PR curve
        "optimal_f1": optimal_f1,  # From PR curve
    }
