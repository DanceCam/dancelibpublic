import torch
import numpy as np


def compute_mad(x, med=None):
    if isinstance(x, torch.Tensor):
        if med is None:
            med = torch.nanmedian(x, dim=1, keepdim=True).values
        mad = torch.nanmedian(torch.abs(x - med), dim=1, keepdim=True).values + 1e-20
    elif isinstance(x, np.ndarray):
        if med is None:
            med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med)) + 1e-20
    else:
        raise TypeError("Input must be a PyTorch Tensor or NumPy array")

    return med, mad


def median_mad(data):
    if isinstance(data, torch.Tensor):
        x = data.clone().reshape(data.shape[0], -1, 1)
        med = None

        for _ in range(4):  # Number of iterations
            med, mad = compute_mad(x, med)
            mask = torch.abs(x - med) > 3.0 * mad
            x[mask] = torch.tensor(float('nan'))

        med = torch.nanmedian(x, dim=1).values
        avg = torch.nanmean(x, dim=1)
        adj_med = (3.5 * med - 2.5 * avg).reshape(-1, 1, 1, 1)

        return adj_med, mad.reshape(-1, 1, 1, 1)

    elif isinstance(data, np.ndarray):
        x = data.flatten()
        med = None

        for _ in range(4):  # Number of iterations
            med, mad = compute_mad(x, med)
            mask = np.abs(x - med) > 3.0 * mad
            x[mask] = np.nan

        med = np.nanmedian(x)
        avg = np.nanmean(x)
        adj_med = 3.5 * med - 2.5 * avg

        return adj_med, mad

    else:
        raise TypeError("Input must be a PyTorch Tensor or NumPy array")