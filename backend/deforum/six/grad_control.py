from functools import partial
from typing import Any, Callable, List, Union
import torch
from loguru import logger
from torch import Tensor
from .k_sampling.k_sampling_utils import append_dims


class GradController:
    def __init__(
        self,
        decode_fn,
        clamp_func: Callable[[Tensor], Tensor] = None,
        consolidate_fn: Callable[[Tensor, Tensor, Tensor], Tensor] = lambda x, y, z: x
        + y * z,
        cond_fns: List[Callable[[Any], Tensor]] = [],
        verbose: bool = False,
        wrt_x: bool = False,
        **kwargs,
    ):
        self.decode_fn = decode_fn
        self.cond_fns = cond_fns if cond_fns is not None else []
        self.clamp_func = clamp_func
        self.consolidate_fn = consolidate_fn
        self.verbose = verbose
        self.wrt_x = wrt_x
        if self.wrt_x:
            self.cond_grad_fn = self._cond_grad_fn_x
        else:
            self.cond_grad_fn = self._cond_grad_fn_x0_pred

    def _cond_calculate(
        self,
        x: Tensor,
        sigma: Tensor,
        denoised: Tensor,
        loss_fn: Callable[[Any],Tensor],
        scale: Union[int, float],
        **kwargs,
    ):
        with torch.enable_grad():
            denoised_sample = self.decode_fn(denoised).requires_grad_()
            loss = loss_fn(denoised_sample, sigma, **kwargs) * scale
            grad = -torch.autograd.grad(loss, x if self.wrt_x else denoised)[0]
        if self.verbose:
            logger.info(f"Loss: {loss.item()}")
        return grad

    def _clamp_and_clean_grad(self, grad: Tensor, sigma: Tensor) -> Tensor:
        # Clean:
        grad = torch.nan_to_num(
            grad, nan=0.0, posinf=float("inf"), neginf=-float("inf")
        )
        # Clamp
        if self.clamp_func is not None:
            grad: Tensor = self.clamp_func(grad, sigma)
        return grad

    def add_cond_fn(self, loss_fn, scale, return_fn=False):
        cond_fn = partial(self._cond_calculate, loss_fn=loss_fn, scale=scale)
        self.cond_fns.append(cond_fn)
        if return_fn:
            return cond_fn

    def _cond_grad_fn_x(
        self,
        x: Tensor,
        sigma: Tensor,
        inner_model: Callable[[Tensor, Tensor, Tensor], Tensor],
        **kwargs,
    ) -> Tensor:
        grad = torch.zeros_like(x)
        for cond_fn in self.cond_fns:
            with torch.enable_grad():
                x: Tensor = x.detach().requires_grad_()
                denoised: Tensor = inner_model(x, sigma, **kwargs)
                cond_grad: Tensor = cond_fn(
                    x, sigma, denoised=denoised, **kwargs
                ).detach()
            grad += cond_grad

        grad = self._clamp_and_clean_grad(grad, sigma)
        x.copy_(self.consolidate_fn(x.detach(), grad, append_dims(sigma, x.ndim)))
        cond_denoised = inner_model(x, sigma, **kwargs)
        return cond_denoised

    def _cond_grad_fn_x0_pred(
        self,
        x: Tensor,
        sigma: Tensor,
        inner_model: Callable[[Tensor, Tensor, Tensor], Tensor] = None,
        **kwargs,
    ) -> Tensor:
        grad = torch.zeros_like(x)
        for cond_fn in self.cond_fns:
            with torch.no_grad():
                denoised: Tensor = inner_model(x, sigma, **kwargs)
            with torch.enable_grad():
                cond_grad: Tensor = cond_fn(
                    x,
                    sigma,
                    denoised=denoised.detach().requires_grad_(),
                    **kwargs,
                ).detach()
            grad += cond_grad
        grad = self._clamp_and_clean_grad(grad, sigma)
        x.copy_(self.consolidate_fn(x.detach(), grad, append_dims(sigma, x.ndim)))
        return self.consolidate_fn(denoised.detach(), grad, append_dims(sigma, x.ndim))

    def clear_cond_fns(self):
        self.cond_fns = []