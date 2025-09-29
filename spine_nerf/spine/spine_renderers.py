import torch
from torch import nn
from torchtyping import TensorType

class CLIPRenderer(nn.Module):
    """Calculate CLIP embeddings along ray."""

    @classmethod
    def forward(
        cls,
        embeds: TensorType["bs":..., "num_samples", "num_classes"],
        weights: TensorType["bs":..., "num_samples", 1],
    ) -> TensorType["bs":..., "num_classes"]:
        """Calculate semantics along the ray."""
        output = torch.sum(weights * embeds, dim=-2)
        output = output / torch.linalg.norm(output, dim=-1, keepdim=True) # can experiment without this line
        return output


class MeanRenderer(nn.Module):
    """Calculate average of embeddings along ray."""

    @classmethod
    def forward(
        cls,
        embeds: TensorType["bs":..., "num_samples", "num_classes"],
        weights: TensorType["bs":..., "num_samples", 1],
    ) -> TensorType["bs":..., "num_classes"]:
        """Calculate semantics along the ray."""
        output = torch.sum(weights * embeds, dim=-2)
        return output


class ContactRenderer(nn.Module):
    """Calculate average of embeddings along ray."""

    @classmethod
    def forward(
        cls,
        embeds: TensorType["bs":..., "num_samples", "3"],
        weights: TensorType["bs":..., "num_samples", 1],
    ) -> TensorType["bs":..., "2"]:
        """Calculate semantics along the ray."""
        # normalize
        embeds = torch.nn.functional.normalize(embeds, p=2.0, dim=-1)
        
        # output
        output = torch.sum(weights * embeds, dim=-2)
        output = output[..., :2]
        
        # normalize
        output = torch.nn.functional.normalize(output, p=2.0, dim=-1)
        
        return output


