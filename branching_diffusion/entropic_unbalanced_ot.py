"""Entropic unbalanced optimal transport and Sinkhorn divergence."""

from __future__ import annotations

from typing import Optional, Union

import torch


def compute_entropic_uot_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    w_x: Optional[torch.Tensor] = None,
    w_y: Optional[torch.Tensor] = None,
    eps: float = 0.01,
    lam: float = 1.0,
    max_iters: int = 100,
    p: float = 2.0,
    delta: float = 1e-10,
    return_components: bool = False,
    balanced: bool = False,
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the entropic unbalanced optimal-transport objective.

    Purpose:
        Solves the finite-support UOT objective with generalized Sinkhorn
        scaling in the direct domain.

    Args:
        x: Source support with shape `(n, d)` and floating dtype.
        y: Target support with shape `(m, d)` and floating dtype.
        w_x: Optional source event mass with shape `(n,)`.
        w_y: Optional target event mass with shape `(m,)`.
        eps: Entropic blur parameter.
        lam: Mass variation penalty.
        max_iters: Number of direct-domain scaling iterations.
        p: Ground-cost power used by `torch.cdist(..., p) ** p`.
        delta: Small stabilizer added to denominators and logarithms.
        return_components: If true, return the component dictionary.
        balanced: If true, use the balanced update equations.

    Returns:
        Either a scalar tensor or a dictionary with `total`, `transport`,
        `entropy`, `kl_p`, and `kl_q`.
    """

    n, _ = x.shape
    m, _ = y.shape
    device = x.device
    a = torch.ones(n, device=device) / n if w_x is None else w_x
    b = torch.ones(m, device=device) / m if w_y is None else w_y
    cost = torch.cdist(x, y, p=p) ** p
    cost = torch.clamp(cost, max=1e3)
    kernel = torch.exp(-cost / eps)
    kernel = torch.clamp(kernel, min=1e-30, max=1e30)
    u = torch.ones(n, device=device)
    v = torch.ones(m, device=device)

    if balanced:
        for _ in range(max_iters):
            kv = kernel @ v
            u = a / (kv + delta)
            ktu = kernel.t() @ u
            v = b / (ktu + delta)
            u = torch.clamp(u, min=1e-30, max=1e30)
            v = torch.clamp(v, min=1e-30, max=1e30)
    else:
        alpha = lam / (lam + eps)
        for _ in range(max_iters):
            kv = kernel @ v
            u = torch.pow(a / (kv + delta), alpha)
            ktu = kernel.t() @ u
            v = torch.pow(b / (ktu + delta), alpha)
            u = torch.clamp(u, min=1e-30, max=1e30)
            v = torch.clamp(v, min=1e-30, max=1e30)

    plan = u.unsqueeze(1) * kernel * v.unsqueeze(0)
    transport_cost = (plan * cost).sum()
    entropy_term = eps * (plan * (torch.log(plan + delta) - 1.0)).sum()
    if balanced:
        total_dist = transport_cost + entropy_term
        kl_term = torch.tensor(0.0, device=device)
    else:
        p_sum = plan.sum(dim=1)
        q_sum = plan.sum(dim=0)
        kl_p_a = (p_sum * torch.log((p_sum + delta) / (a + delta)) - p_sum + a).sum()
        kl_q_b = (q_sum * torch.log((q_sum + delta) / (b + delta)) - q_sum + b).sum()
        kl_term = lam * (kl_p_a + kl_q_b)
        total_dist = transport_cost + entropy_term + kl_term

    if return_components:
        return {
            "total": total_dist,
            "transport": transport_cost,
            "entropy": entropy_term,
            "kl_p": kl_term * 0.5,
            "kl_q": kl_term * 0.5,
        }
    return total_dist


def compute_debiased_sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    w_x: Optional[torch.Tensor] = None,
    w_y: Optional[torch.Tensor] = None,
    eps: float = 0.01,
    lam: float = 1.0,
    max_iters: int = 100,
    p: float = 2.0,
) -> torch.Tensor:
    """Compute the debiased Sinkhorn divergence.

    Args:
        x: First support with shape `(n, d)`.
        y: Second support with shape `(m, d)`.
        w_x: Optional first weights.
        w_y: Optional second weights.
        eps: Entropic blur parameter.
        lam: Unbalanced mass KL penalty.
        max_iters: Number of iterations.
        p: Ground-cost power.

    Returns:
        Nonnegative scalar tensor after nonnegative clamping.
    """

    uot_xy = compute_entropic_uot_loss(x, y, w_x, w_y, eps, lam, max_iters, p)
    uot_xx = compute_entropic_uot_loss(x, x, w_x, w_x, eps, lam, max_iters, p)
    uot_yy = compute_entropic_uot_loss(y, y, w_y, w_y, eps, lam, max_iters, p)
    return torch.clamp(uot_xy - 0.5 * uot_xx - 0.5 * uot_yy, min=0.0)


def _sinkhorn_with_cost_scale(
    x: torch.Tensor,
    y: torch.Tensor,
    w_x: Optional[torch.Tensor],
    w_y: Optional[torch.Tensor],
    eps: float,
    lam: float,
    max_iters: int,
    p: float,
    return_components: bool,
    transport_mode: str,
    cost_scale: float,
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the inner solver with cost divided by `cost_scale ** 2`."""

    cost = torch.cdist(x, y, p=p) ** p
    cost = cost / (cost_scale**2)
    cost = torch.clamp(cost, max=1e3)
    kernel = torch.exp(-cost / eps)
    kernel = torch.clamp(kernel, min=1e-30, max=1e30)
    device = x.device
    n, _ = x.shape
    m, _ = y.shape
    a = torch.ones(n, device=device) / n if w_x is None else w_x
    b = torch.ones(m, device=device) / m if w_y is None else w_y
    u = torch.ones(n, device=device)
    v = torch.ones(m, device=device)
    delta = 1e-10

    if transport_mode == "ot":
        for _ in range(max_iters):
            kv = kernel @ v
            u = a / (kv + delta)
            ktu = kernel.t() @ u
            v = b / (ktu + delta)
            u = torch.clamp(u, min=1e-30, max=1e30)
            v = torch.clamp(v, min=1e-30, max=1e30)
    else:
        alpha = lam / (lam + eps)
        for _ in range(max_iters):
            kv = kernel @ v
            u = torch.pow(a / (kv + delta), alpha)
            ktu = kernel.t() @ u
            v = torch.pow(b / (ktu + delta), alpha)
            u = torch.clamp(u, min=1e-30, max=1e30)
            v = torch.clamp(v, min=1e-30, max=1e30)

    plan = u.unsqueeze(1) * kernel * v.unsqueeze(0)
    transport_cost = (plan * cost).sum()
    entropy_term = eps * (plan * (torch.log(plan + delta) - 1.0)).sum()
    if transport_mode == "ot":
        total_dist = transport_cost + entropy_term
        kl_term = torch.tensor(0.0, device=device)
    else:
        p_sum = plan.sum(dim=1)
        q_sum = plan.sum(dim=0)
        kl_p_a = (p_sum * torch.log((p_sum + delta) / (a + delta)) - p_sum + a).sum()
        kl_q_b = (q_sum * torch.log((q_sum + delta) / (b + delta)) - q_sum + b).sum()
        kl_term = lam * (kl_p_a + kl_q_b)
        total_dist = transport_cost + entropy_term + kl_term

    if return_components:
        return {
            "total": total_dist,
            "transport": transport_cost,
            "entropy": entropy_term,
            "kl_p": kl_term * 0.5,
            "kl_q": kl_term * 0.5,
        }
    return total_dist


def compute_diffusion_step_sinkhorn_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    w_x: Optional[torch.Tensor] = None,
    w_y: Optional[torch.Tensor] = None,
    eps: float = 0.01,
    lam: float = 1.0,
    max_iters: int = 100,
    p: float = 2.0,
    debiased: bool = False,
    return_components: bool = False,
    transport_mode: str = "uot",
    use_bias_correction: bool = True,
    cost_delta: float = 1.0,
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the diffusion-step Sinkhorn training loss.

    Purpose:
        Provides the training-path UOT/Sinkhorn entry point, including the
        optional debiased self-similarity correction used by the current
        implementation.

    Args:
        x: First support with shape `(n, d)`.
        y: Second support with shape `(m, d)`.
        w_x: Optional first event-mass tensor with shape `(n,)`.
        w_y: Optional second event-mass tensor with shape `(m,)`.
        eps: Entropic blur parameter.
        lam: Mass variation penalty.
        max_iters: Number of direct-domain iterations.
        p: Ground-cost power.
        debiased: Accepted for interface compatibility; the active branch is
            controlled by `use_bias_correction`.
        return_components: If true, return a dictionary. For the
            bias-corrected branch, this package wraps the exact scalar under
            `total` so the target training path can consume it.
        transport_mode: `uot` or `ot`.
        use_bias_correction: Whether to compute self-similarity
            corrected scalar.
        cost_delta: Denominator for the temporal cost scale.

    Returns:
        A scalar tensor or a component dictionary. The `total` field in the
        wrapped bias-corrected dictionary is exactly the scalar returned by the scalar branch.
    """

    del debiased
    if transport_mode == "ot":
        n = x.shape[0]
        m = y.shape[0]
        if n != m:
            min_len = min(n, m)
            x = x[:min_len]
            y = y[:min_len]
            if w_x is not None:
                w_x = w_x[:min_len]
                w_x = w_x / w_x.sum()
            if w_y is not None:
                w_y = w_y[:min_len]
                w_y = w_y / w_y.sum()

    if use_bias_correction:
        uot_xy = _sinkhorn_with_cost_scale(
            x, y, w_x, w_y, eps, lam, max_iters, p, False, transport_mode, cost_delta
        )
        uot_xx = _sinkhorn_with_cost_scale(
            x, x, w_x, w_x, eps, lam, max_iters, p, False, transport_mode, cost_delta
        )
        uot_yy = _sinkhorn_with_cost_scale(
            y, y, w_y, w_y, eps, lam, max_iters, p, False, transport_mode, cost_delta
        )
        total = torch.clamp(uot_xy - 0.5 * uot_xx - 0.5 * uot_yy, min=0.0)
        if return_components:
            zero = total.new_tensor(0.0)
            return {
                "total": total,
                "transport": uot_xy,
                "entropy": zero,
                "kl_p": zero,
                "kl_q": zero,
            }
        return total

    return _sinkhorn_with_cost_scale(
        x, y, w_x, w_y, eps, lam, max_iters, p, return_components, transport_mode, cost_delta
    )


__all__ = [
    "compute_debiased_sinkhorn_divergence",
    "compute_diffusion_step_sinkhorn_loss",
    "compute_entropic_uot_loss",
]
