"""Launch official SAM3 training with an autograd-safe ViT MLP path."""

from __future__ import annotations

import runpy

import torch


def install_training_safe_vit_mlp() -> None:
    """Keep SAM3's fused inference path but use regular layers with gradients."""
    from sam3.model import vitdet

    def forward(self, x):
        if torch.is_grad_enabled():
            x = self.act(self.fc1(x))
        else:
            x = vitdet.addmm_act(type(self.act), self.fc1, x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    vitdet.Mlp.forward = forward


if __name__ == "__main__":
    install_training_safe_vit_mlp()
    runpy.run_module("sam3.train.train", run_name="__main__")
