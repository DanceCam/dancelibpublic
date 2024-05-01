import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from blended_tiling import TilingModule

from utils import anscombe_transform, inv_anscombe_transform, median_mad

def infer_frames(video_frames, model, input_channels, tile_size, tile_overlap, device='cpu', deep_supervision=False):
    """
    Infers a series of frames from a video stream using a given model and tile configuration.
    Accepts video frames in either NumPy or PyTorch tensor formats.

    Parameters:
    - video_frames : NumPy ndarray or torch.Tensor
      A batch of video frames of shape [number_of_frames, height, width].
    - model : torch.nn.Module
      The neural network model for inference.
    - input_channels : int
      The number of input channels the model expects (how many frames at a time).
    - tile_size : list of int
      The dimensions [height, width] of each tile.
    - tile_overlap : list of float
      The overlap percentage [height, width] for tiling.
    - device : str
      The device to perform computations on ('cpu' or 'cuda').
    - deep_supervision : bool
      Whether the model uses deep supervision.

    Returns:
    - inferred_stack : ndarray
      The inferred output as a numpy array.
    """

    # Convert NumPy arrays to torch.Tensor if necessary and fix byte order issue
    if isinstance(video_frames, np.ndarray):
        # Ensure the array is in native byte order
        if np.dtype(video_frames.dtype).byteorder == '>':
            video_frames = video_frames.byteswap().newbyteorder()
        video_frames = torch.from_numpy(video_frames)

    # Ensure the tensor is in the correct device and format
    video_frames = video_frames.float().to(device)

    # Determine full size from video frames
    full_size = list(video_frames.shape[1:])  # Assuming [number_of_frames, height, width]

    # Initialize the tiling module with dynamic size
    tiling_module = TilingModule(
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        base_size=full_size
    )
    
    inferred_tiles_sequence = []

    # Iterate over each frame slice
    for i in tqdm(range(len(video_frames) - input_channels + 1)):
        # Slice and reshape frames for tiling
        tiles = tiling_module.split_into_tiles(video_frames[i:i+input_channels].unsqueeze(0))
        with torch.no_grad():
            med, _ = median_mad(tiles)
            tiles -= med
            input_sequence = nn.functional.relu(tiles, inplace=False)
            cinput_sequence = anscombe_transform(input_sequence)
            if deep_supervision:
                inferred_tiles = model(cinput_sequence.reshape(-1, input_channels, *tile_size))[0]
            else:
                inferred_tiles = model(cinput_sequence.reshape(-1, input_channels, *tile_size))
            inferred_tiles = inv_anscombe_transform(inferred_tiles)
            inferred_tiles_sequence.append(inferred_tiles.squeeze())

            # Clear GPU cache
            del input_sequence, cinput_sequence, inferred_tiles
            torch.cuda.empty_cache()

    # Stack and average the inferred tiles
    inferred_tiles_sequence = torch.stack(inferred_tiles_sequence)
    mean_inferred_tiles_sequence = inferred_tiles_sequence.mean(dim=0)
    
    # Rebuild the full image from the inferred tiles
    inferred_stack = tiling_module.rebuild_with_masks(mean_inferred_tiles_sequence.unsqueeze(1)).squeeze().detach().cpu().numpy()

    return inferred_stack
