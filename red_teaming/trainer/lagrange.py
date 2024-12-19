from typing import Callable

import torch
from torch import nn


class LagrangeMultiplier(nn.Module):
    def __init__(
        self,
        name: str,
        initial_value: float,
        lr: float,
        momentum: float,
        max_value: float,
        threshold: float,
        transform_fn: Callable,
        inverse_fn: Callable,
        device: str,
        is_main_process: bool,
    ):
        super(LagrangeMultiplier, self).__init__()
        self._name = name
        # only update lagrange multiplier in rank 0 process
        self.transformed_lambda = torch.nn.Parameter(
            inverse_fn(torch.tensor(initial_value, device=device)),
            requires_grad=is_main_process,
        )
        if is_main_process:
            self.lambda_optimizer = torch.optim.SGD(
                [self.transformed_lambda], lr=lr, momentum=momentum,
            )
        else:
            self.lambda_optimizer = None
        self.transformed_lambda_max = inverse_fn(torch.tensor(max_value)) if max_value else None
        self.inverse_fn = inverse_fn
        self.transform_fn = transform_fn
        self.threshold = threshold

    @property
    def name(self):
        return self._name

    def get_multiplier(self):
        return self.transform_fn(self.transformed_lambda)

    def update_lambda(self, episode_cost: torch.Tensor):
        lambda_loss = -(episode_cost - self.threshold) * self.get_multiplier()
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        if self.transformed_lambda_max is not None:
            with torch.no_grad():
                if self.transformed_lambda_max.device != episode_cost.device:
                    self.transformed_lambda_max = self.transformed_lambda_max.to(episode_cost.device)
                self.transformed_lambda.clamp_(max=self.transformed_lambda_max)
