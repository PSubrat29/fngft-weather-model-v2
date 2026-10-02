from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def prediction_loss(pred: Tensor, target: Tensor) -> Tensor:
    return torch.mean((pred - target) ** 2)


def mae_loss(pred: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.abs(pred - target))


def gradient_loss(pred: Tensor, target: Tensor) -> Tensor:
    px = pred[..., :, 1:] - pred[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    py = pred[..., 1:, :] - pred[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return torch.mean((px - tx) ** 2) + torch.mean((py - ty) ** 2)


def spectral_energy_loss(pred: Tensor, target: Tensor) -> Tensor:
    pred_hat = torch.fft.rfft2(pred, norm="ortho")
    target_hat = torch.fft.rfft2(target, norm="ortho")
    ep = (pred_hat.abs() ** 2).mean(dim=(0, 1))
    et = (target_hat.abs() ** 2).mean(dim=(0, 1))
    return torch.mean((torch.log1p(ep) - torch.log1p(et)) ** 2)


def order_regularization(info: Dict[str, Tensor]) -> Tensor:
    alpha = info["alpha"]
    beta = info["beta"]
    kappa = info["kappa"]
    smooth_alpha = torch.mean(torch.abs(alpha[..., :, 1:] - alpha[..., :, :-1])) + torch.mean(torch.abs(alpha[..., 1:, :] - alpha[..., :-1, :]))
    smooth_beta = torch.mean(torch.abs(beta[..., :, 1:] - beta[..., :, :-1])) + torch.mean(torch.abs(beta[..., 1:, :] - beta[..., :-1, :]))
    positivity = torch.relu(-kappa).mean()
    return 0.1 * (smooth_alpha + smooth_beta) + positivity


def composite_real_data_loss(pred: Tensor, target: Tensor, info: Dict[str, Tensor]) -> Dict[str, Tensor]:
    pred_l = prediction_loss(pred, target)
    grad_l = gradient_loss(pred, target)
    spec_l = spectral_energy_loss(pred, target)
    reg_l = order_regularization(info)
    total = pred_l + 0.10 * grad_l + 0.02 * spec_l + 0.01 * reg_l
    return {"total": total, "pred": pred_l, "grad": grad_l, "spectral": spec_l, "order_reg": reg_l}


def fractional_operator_consistency_loss(predicted_closure: Tensor, inferred_sgs: Tensor) -> Tensor:
    """Scientific loss used when paired high-resolution SGS targets are available."""
    return torch.mean((predicted_closure - inferred_sgs) ** 2)
