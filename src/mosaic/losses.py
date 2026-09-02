"""Observed-mask losses for MOSAIC models."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    observed = mask.to(dtype=prediction.dtype)
    denominator = observed.sum().clamp_min(1.0)
    return ((prediction - target).square() * observed).sum() / denominator


def masked_rmse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(masked_mse(prediction, target, mask))


def masked_delta_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pairs: Sequence[Tuple[int, List[int]]],
) -> torch.Tensor:
    if not pairs:
        return prediction.sum() * 0.0
    squared_error = prediction.sum() * 0.0
    observed_count = prediction.new_tensor(0.0)
    for treatment, controls in pairs:
        control_index = torch.as_tensor(controls, device=prediction.device, dtype=torch.long)
        control_mask = mask.index_select(0, control_index)
        control_count = control_mask.sum(dim=0)
        valid = mask[treatment] & (control_count > 0)
        denominator = control_count.clamp_min(1).to(dtype=prediction.dtype)
        pred_control = (prediction.index_select(0, control_index) * control_mask).sum(dim=0) / denominator
        true_control = (target.index_select(0, control_index) * control_mask).sum(dim=0) / denominator
        error = (prediction[treatment] - pred_control) - (target[treatment] - true_control)
        squared_error = squared_error + (error.square() * valid).sum()
        observed_count = observed_count + valid.sum()
    return squared_error / observed_count.clamp_min(1.0)


def direction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 1.0,
) -> torch.Tensor:
    active = mask & (target.abs() > threshold)
    if not bool(active.any()):
        return prediction.sum() * 0.0
    wrong = torch.relu(-(prediction * target))
    return wrong[active].mean()


def mosaic_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pairs: Sequence[Tuple[int, List[int]]] = (),
    lambda_delta: float = 0.05,
    measurement_penalty: torch.Tensor = None,
    measurement_l2: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    absolute = masked_mse(prediction, target, mask)
    delta = masked_delta_mse(prediction, target, mask, pairs)
    regularization = prediction.sum() * 0.0 if measurement_penalty is None else measurement_penalty
    total = absolute + float(lambda_delta) * delta + float(measurement_l2) * regularization
    return total, {"absolute_mse": absolute, "delta_mse": delta, "measurement_penalty": regularization}

