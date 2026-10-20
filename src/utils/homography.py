import numpy as np
import torch
from kornia.geometry.homography import find_homography_dlt

from ..third_party.gluefactory.geometry.homography import homography_corner_error, sym_homography_error
from ..third_party.gluefactory.robust_estimators import load_estimator
from ..third_party.gluefactory.utils.tools import AUCMetric


def compute_homography_metrics(batch, config):
    """Return HPatches matching and homography metrics for one image pair."""
    ransac_thr = config.DATASET.RANSAC_THR
    if isinstance(ransac_thr, float):
        ransac_thr = [ransac_thr]
    
    pred_data = prepare_homography_data(batch, topk=config.DATASET.TOPK)
    
    matching_metrics = compute_matching_accuracy(pred_data)
    dlt_metrics = compute_dlt_homography(pred_data)
    ransac_metrics = compute_ransac_homography(pred_data, ransac_thr)
    
    metrics = {
        **matching_metrics,
        **dlt_metrics,
        **ransac_metrics
    }
    
    return {'metrics': metrics}


def prepare_homography_data(batch, topk=1000):
    """Convert matches and homography to resized-image coordinates.

    Keep the top matches by coarse confidence when topk is positive.
    """

    device = batch['scale0'].device

    S0 = torch.eye(3, dtype=torch.float32, device=device)
    S1 = torch.eye(3, dtype=torch.float32, device=device)

    S0[0, 0], S0[1, 1] = batch['scale0'][0]
    S1[0, 0], S1[1, 1] = batch['scale1'][0]

    pts0 = batch['mkpts0_f'] / batch['scale0'][0]
    pts1 = batch['mkpts1_f'] / batch['scale1'][0]
    score = batch['mconf_c']

    if topk is not None and topk > 0:
        k = min(topk, score.shape[0])
        idx = torch.topk(score, k=k, largest=True, sorted=True).indices

        pts0 = pts0[idx]
        pts1 = pts1[idx]
        score = score[idx]

    return {
        'H_0to1': torch.linalg.inv(S1) @ batch['H_0to1'][0] @ S0,
        'pts0': pts0,
        'pts1': pts1,
        'score': score,
        'size': batch['prepad_size0'][0].flip(dims=[0])
    }


def compute_matching_accuracy(pred_data):
    """Return match count and precision based on symmetric homography error."""
    H_gt = pred_data["H_0to1"]
    assert H_gt.ndim == 2
    
    err = sym_homography_error(pred_data['pts0'], pred_data['pts1'], H_gt)
    
    return {
        'num_matches': pred_data['pts0'].shape[0],
        'prec@3px': (err < 3).float().mean().nan_to_num().item(),
        'prec@5px': (err < 5).float().mean().nan_to_num().item(),
        'prec@10px': (err < 10).float().mean().nan_to_num().item(),
    }


def compute_dlt_homography(pred_data):
    """Return the corner error in pixels for weighted DLT, or inf on failure."""
    H_gt = pred_data["H_0to1"]
    assert H_gt.ndim == 2
    
    # Homography estimation requires at least four correspondences.
    if pred_data['pts0'].shape[0] < 4:
        return {'H_error_dlt': float('inf')}
    
    try:
        h_dlt = find_homography_dlt(
            pred_data['pts0'][None],
            pred_data['pts1'][None],
            pred_data['score'][None]
        )[0]
        error_dlt = homography_corner_error(h_dlt, H_gt, pred_data['size'])
        return {'H_error_dlt': error_dlt.item()}
    except Exception as e:
        print(f"DLT homography estimation failed: {e}")
        return {'H_error_dlt': float('inf')}


def compute_ransac_homography(pred_data, ransac_thresholds):
    """Return RANSAC metrics for the threshold with the lowest corner error."""
    best_metrics = None
    best_error = float('inf')
    
    for th in ransac_thresholds:
        metrics = estimate_homography_ransac(
            pred_data,
            {"estimator": "opencv", "ransac_th": th, "options": {"confidence": 0.99999}}
        )
        
        if metrics['H_error_ransac'] < best_error:
            best_error = metrics['H_error_ransac']
            best_metrics = metrics
            best_metrics['ransac_threshold'] = th
    
    return best_metrics if best_metrics else {
        'H_error_ransac': float('inf'),
        'ransac_inl': 0,
        'ransac_inl_ratio': 0.0,
        'ransac_threshold': ransac_thresholds[0] if ransac_thresholds else 1.0
    }


def estimate_homography_ransac(pred_data, conf):
    """Run RANSAC on CPU and return corner error and inlier statistics."""
    H_gt = pred_data["H_0to1"]
    estimator = load_estimator("homography", conf["estimator"])(conf)
    
    data = {
        "m_kpts0": pred_data['pts0'].detach().cpu(),
        "m_kpts1": pred_data['pts1'].detach().cpu()
    }
    
    est = estimator(data)
    
    if est["success"]:
        M = torch.tensor(
            est["M_0to1"],
            dtype=torch.float32,
            device=pred_data['size'].device
        ).clone().detach()
        error = homography_corner_error(M, H_gt, pred_data['size']).item()
    else:
        error = float('inf')
    
    results = {'H_error_ransac': error}
    
    if "inliers" in est:
        inl = est["inliers"]
        results['ransac_inl'] = inl.float().sum().item()
        results['ransac_inl_ratio'] = inl.float().mean().item()
    else:
        results['ransac_inl'] = 0
        results['ransac_inl_ratio'] = 0.0
    
    return results


def aggregate_homography_metrics(metrics_list, config):
    """Aggregate per-pair metrics into summary statistics and error AUCs."""
    if not metrics_list:
        return {}
    
    aggregated = {}
    metric_keys = set()
    for m in metrics_list:
        metric_keys.update(m.keys())
    
    for key in metric_keys:
        values = [m[key] for m in metrics_list if key in m]
        if 'error' in key:
            auc_thresholds = [3, 5, 10]
            auc_scores = compute_auc_scores(values, auc_thresholds)
            for th, score in zip(auc_thresholds, auc_scores):
                aggregated[f'{key}@{th}px'] = score

        values = [value for value in values if np.isfinite(value)]
        
        if values:
            if key in ['num_matches', 'ransac_inl']:
                aggregated[f'total_{key}'] = sum(values)
                aggregated[f'mean_{key}'] = np.mean(values)
                aggregated[f'median_{key}'] = np.median(values)
            elif 'prec@' in key or 'ratio' in key:
                aggregated[f'mean_{key}'] = np.mean(values)
                aggregated[f'median_{key}'] = np.median(values)
            elif 'error' in key:
                aggregated[f'mean_{key}'] = np.mean(values)
                aggregated[f'median_{key}'] = np.median(values)
                aggregated[f'std_{key}'] = np.std(values)
            elif key == 'ransac_threshold':
                from collections import Counter
                counter = Counter(values)
                aggregated['most_common_ransac_threshold'] = counter.most_common(1)[0][0]
            else:
                aggregated[f'mean_{key}'] = np.mean(values)
    
    if 'total_num_matches' in aggregated:
        aggregated['pairs_evaluated'] = len(metrics_list)
        aggregated['matches_per_pair'] = aggregated['total_num_matches'] / len(metrics_list)
    
    return aggregated


def compute_auc_scores(errors, thresholds):
    """Return GlueFactory error AUCs, retaining failed pairs in the denominator."""
    if not errors:
        return [0.0] * len(thresholds)

    metric = AUCMetric(thresholds, errors)
    return metric.compute()
