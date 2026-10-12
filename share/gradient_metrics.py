"""Diagnostics for interactions between PINN loss-component gradients."""

from itertools import combinations

import torch


def component_gradient_metrics(loss_dict, parameters, weights, epsilon=1e-12):
    """Return raw/weighted gradient norms and pairwise cosine similarities.

    Metrics are computed before the optimizer backward pass. The graph is
    retained so the caller can subsequently backpropagate the balanced loss.
    """
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    names = [key.removesuffix("_loss") for key in loss_dict]
    gradients = {}
    squared_norms = {}

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
        squared_norms[name] = squared_norm

    norms = {
        name: torch.sqrt(squared_norm.clamp_min(0.0))
        for name, squared_norm in squared_norms.items()
    }
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
