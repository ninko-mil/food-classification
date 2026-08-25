import os
import random
import platform

import numpy as np
import torch
import torchvision


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_environment_info():
    info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

        properties = torch.cuda.get_device_properties(0)
        info["gpu_memory_gb"] = round(
            properties.total_memory / (1024 ** 3),
            2
        )
    else:
        info["gpu_name"] = "CPU"
        info["gpu_count"] = 0
        info["gpu_memory_gb"] = 0

    return info