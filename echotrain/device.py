"""Pick the device to run on: an NVIDIA GPU (CUDA), an Apple Silicon GPU (MPS) or the CPU."""
import torch


def pick_device(requested="cuda:0"):
    """Return the requested device if it is available, otherwise the best one that is.

    requested: a torch device string from the config, e.g. "cuda:0", "cuda:1", "mps" or "cpu".
    "cpu" is always honoured. Otherwise the order is CUDA, then MPS, then CPU, so a config
    written for an NVIDIA machine also uses the GPU of an Apple Silicon Mac.
    """
    requested = str(requested)
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(requested if requested.startswith("cuda") else "cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
