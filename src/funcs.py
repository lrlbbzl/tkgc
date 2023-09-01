import torch
from torch import Tensor

def givens_rotation(r, x, transpose=False):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate
        transpose: whether to transpose the rotation matrix

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    if transpose:
        x_rot = givens[:, :, 0:1] * x - givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    else:
        x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


def euc_distance(x: Tensor, y: Tensor, eval_mode=False) -> Tensor:
    """calculate eucidean distance

    Args:
        x (Tensor): shape:(N1, d), the x tensor 
        y (Tensor): shape (N2, d) if eval_mode else (N1, d), the y tensor
        eval_mode (bool, optional): whether or not use eval model. Defaults to False.

    Returns:
        if eval mode: (N1, N2)
        else: (N1, 1)
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)

    if eval_mode:
        y2 = y2.t()
        xy = x @ y.t()
    else:
        assert x.shape[0] == y.shape[0], 'The shape of x and y do not match.'
        xy = torch.sum(x * y, dim=-1, keepdim=True)

    return x2 + y2 - 2 * xy