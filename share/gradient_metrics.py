"""Diagnostics for interactions between PINN loss-component gradients."""

from dataclasses import dataclass
from itertools import combinations

import torch


@dataclass
class GradientStats:
    """Per-term gradient magnitude statistics over all trainable parameters."""

    max_abs: dict
    mean_abs: dict

    def reduced(self, accelerator):
        """Return statistics that are identical on every distributed process.

        ``torch.autograd.grad`` does not trigger DDP gradient synchronization,
        so per-process statistics are combined explicitly. Parameter counts are
        identical on every process, which makes the mean of the per-process
        means the global mean; the global maximum is gathered element-wise.
        """
        return GradientStats(
            max_abs={
                name: accelerator.gather(value.reshape(1)).max().item()
                for name, value in self.max_abs.items()
            },
            mean_abs={
                name: accelerator.reduce(value, reduction="mean").item()
                for name, value in self.mean_abs.items()
            },
        )


def _component_gradients(loss_dict, parameters):
    """Return per-term detached gradients and raw gradient norms."""
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    names = [key.removesuffix("_loss") for key in loss_dict]
    gradients = {}
    norms = {}

    for name in names:
        component_gradients = torch.autograd.grad(
            loss_dict[f"{name}_loss"],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        gradients[name] = tuple(
            None if gradient is None else gradient.detach()
            for gradient in component_gradients
        )
        squared_norm = loss_dict[f"{name}_loss"].new_zeros(())
        for gradient in gradients[name]:
            if gradient is not None:
                squared_norm = squared_norm + gradient.square().sum()
        norms[name] = torch.sqrt(squared_norm.clamp_min(0.0))

    return names, gradients, norms


def _gradient_metrics(loss_dict, names, gradients, norms, weights, epsilon):
    metrics = {}
    for name, norm in norms.items():
        metrics[f"{name}_gradient_norm"] = norm
        metrics[f"{name}_weighted_gradient_norm"] = norm * abs(weights[name])

    for first, second in combinations(names, 2):
        dot_product = loss_dict[f"{first}_loss"].new_zeros(())
        for first_gradient, second_gradient in zip(
            gradients[first], gradients[second]
        ):
            if first_gradient is not None and second_gradient is not None:
                dot_product = dot_product + first_gradient.mul(second_gradient).sum()
        denominator = (norms[first] * norms[second]).clamp_min(epsilon)
        metrics[f"gradient_cosine_{first}_{second}"] = dot_product / denominator
    return metrics


def _flatten_gradients(gradients, weights):
    """Concatenate each term's weighted parameter gradients into one flat tensor."""
    flattened = {}
    for name, values in gradients.items():
        available = [gradient.reshape(-1) for gradient in values if gradient is not None]
        if available:
            flattened[name] = torch.cat(available) * abs(weights[name])
    return flattened


def _gradient_statistics(names, gradients):
    max_abs = {}
    mean_abs = {}
    for name in names:
        available = [gradient for gradient in gradients[name] if gradient is not None]
        if not available:
            raise RuntimeError(f"No gradients were found for loss term {name!r}.")
        element_count = sum(gradient.numel() for gradient in available)
        max_abs[name] = torch.stack(
            [gradient.abs().max() for gradient in available]
        ).max()
        mean_abs[name] = (
            sum(gradient.abs().sum() for gradient in available) / element_count
        )
    return GradientStats(max_abs=max_abs, mean_abs=mean_abs)


def component_gradient_metrics(
    loss_dict, parameters, weights, epsilon=1e-12, return_histograms=False
):
    """Return raw/weighted gradient norms and pairwise cosine similarities.

    Metrics are computed before the optimizer backward pass. The graph is
    retained so the caller can subsequently backpropagate the balanced loss.
    With ``return_histograms=True``, also return each term's flattened
    weighted parameter gradients for distribution logging.
    """
    names, gradients, norms = _component_gradients(loss_dict, parameters)
    metrics = _gradient_metrics(loss_dict, names, gradients, norms, weights, epsilon)
    if return_histograms:
        return metrics, _flatten_gradients(gradients, weights)
    return metrics


def component_gradient_statistics(loss_dict, parameters):
    """Return per-term max and mean absolute gradient element statistics."""
    names, gradients, _ = _component_gradients(loss_dict, parameters)
    return _gradient_statistics(names, gradients)


def component_gradient_summary(
    loss_dict, parameters, weights, epsilon=1e-12, return_histograms=False
):
    """Return diagnostics and gradient statistics from a single autograd pass."""
    names, gradients, norms = _component_gradients(loss_dict, parameters)
    metrics = _gradient_metrics(loss_dict, names, gradients, norms, weights, epsilon)
    statistics = _gradient_statistics(names, gradients)
    if return_histograms:
        return metrics, statistics, _flatten_gradients(gradients, weights)
    return metrics, statistics
