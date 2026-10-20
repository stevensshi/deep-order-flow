"""The device of a fit and the settings that change its arithmetic, recorded as torch reports them.

On the CPU no ``torch.cuda`` function is called. On CUDA, cuDNN's algorithm search follows
``machine.cudnn_benchmark``; the two TF32 switches are set only when the configured value differs from torch's
current one; the allocator is capped at ``machine.gpu_memory_limit_bytes`` when that is set. Deterministic
algorithms are switched on or off on every device (on the CPU they give the default weights).
"""

from __future__ import annotations

import torch

from dof.deep.config import MachineConfig
from dof.deep.errors import DeviceError


def setup_device(machine: MachineConfig, log=print) -> torch.device:
    torch.use_deterministic_algorithms(machine.deterministic)
    if machine.torch_threads:
        torch.set_num_threads(machine.torch_threads)
    if machine.device != "cuda":
        log(f"torch {torch.__version__}, device cpu")
        return torch.device("cpu")
    available = torch.cuda.is_available()
    log(f"torch {torch.__version__}, cuda available {available}, device "
        f"{torch.cuda.get_device_name(0) if available else 'none'}")
    if not available:
        raise DeviceError("CUDA is not available")
    torch.backends.cudnn.benchmark = machine.cudnn_benchmark
    if torch.backends.cudnn.allow_tf32 != machine.cudnn_allow_tf32:
        torch.backends.cudnn.allow_tf32 = machine.cudnn_allow_tf32
    if torch.backends.cuda.matmul.allow_tf32 != machine.matmul_allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = machine.matmul_allow_tf32
    if machine.gpu_memory_limit_bytes is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, machine.gpu_memory_limit_bytes / total), 0)
    return torch.device("cuda")


def device_facts(device: torch.device) -> dict:
    """The settings in force, read from torch, not from the configuration."""
    cuda = device.type == "cuda"
    limit = None
    if cuda:
        limit = int(torch.cuda.get_per_process_memory_fraction(0) * torch.cuda.get_device_properties(0).total_memory)
    return {"type": device.type, "name": torch.cuda.get_device_name(0) if cuda else "cpu", "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if cuda else None,
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic": torch.are_deterministic_algorithms_enabled(), "torch_threads": torch.get_num_threads(),
            "memory_limit_bytes": limit}
