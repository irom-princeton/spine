
import torch

from torch import Tensor
import scipy
import numpy as np
from nerfstudio.utils.misc import torch_compile

# rotation_conversion_utils from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
@torch_compile()
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

@torch_compile()
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)

@torch_compile()
def _lie_algebra_to_rotmat(lie_algebra: Tensor) -> Tensor:
    """Convert lie algebra to rotation matrix."""
    # TODO: vectorize
    assert (
        len(lie_algebra.shape) == 1 or lie_algebra.shape[0] == 1,
        "Higher-dimension rotation map not implemented!"
    )
    
    # reshape
    if len(lie_algebra.shape) > 1:
        lie_algebra = lie_algebra.reshape(-1)
    
    # construct the skew-symmetric matrix
    skew = torch.zeros((3, 3)).to(lie_algebra.device)
    skew[0, 1] = -lie_algebra[2]
    skew[0, 2] =  lie_algebra[1]
    skew[1, 0] =  lie_algebra[2]
    skew[1, 2] = -lie_algebra[0]
    skew[2, 0] = -lie_algebra[1]
    skew[2, 1] =  lie_algebra[0]

    # skew = torch.tensor(
    #     [[0,                -lie_algebra[2],   lie_algebra[1]],
    #     [lie_algebra[2],    0,               -lie_algebra[0]],
    #     [-lie_algebra[1],   lie_algebra[0],   0]]
    # ).to(lie_algebra.device)
    
    # matrix exponential
    R = torch.linalg.matrix_exp(skew)
    
    return R


def _rotmat_to_lie_algebra(rot_mat: Tensor) -> Tensor:
    """Convert rotation matrix to lie algebra."""
    #  matrix logarithm
    skew = scipy.linalg.logm(rot_mat.squeeze().detach().cpu().numpy())
    
    # TODO: implement better error-checking
    skew = np.real(skew)
    
    # construct the skew-symmetric matrix
    lie_algebra = torch.zeros((3,)).to(rot_mat.device)
    
    # unpack components
    lie_algebra[0] = skew[2, 1]
    lie_algebra[1] = skew[0, 2]
    lie_algebra[2] = skew[1, 0]
    
    return lie_algebra


def kl_divergence_loss_fn(mu, logvar):
    loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    loss = loss.mean()
    
    return loss


def _compute_GMM_entropy(weights: Tensor, cov: Tensor) -> Tensor:
    """
    Compute the (approximate) entropy of a Gaussian Mixture Model (GMM)
    
    Args:
    weights: (B, M) -> Batch size (B), M Gaussians in GMM
    cov: (B, M, D) ->  Batch size (B), D-dimensional variance for M GMM components
    """
    # input type-checking
    weights = weights.float()
    cov = cov.float()
    
    # TODO: Revisit implementation
    # uses the expected entropy across the GMM components
    # assumes independent Gaussian distributions
    entropy = 1.5 * (1 + torch.log(2 * torch.tensor(np.pi))) + 0.5 * torch.log(torch.prod(
        cov,
        dim=-1,
        keepdim=False,
    ))
    
    # take the expectation
    entropy = ((weights / weights.sum()) * entropy).sum(dim=-1)
    
    return entropy
    