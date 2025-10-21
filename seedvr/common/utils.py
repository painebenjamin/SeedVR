import inspect
import time
import torch
import torch.nn.functional as F
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, Dict, Any

from functools import lru_cache
from typing import Any
from urllib.request import urlopen

import torch
import torch.nn.functional as F

def filter_kwargs_for_method(
    method: Callable,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Filter kwargs for a method to only include valid parameters.
    """
    try:
        sig = inspect.signature(method)
    except Exception as e:
        return kwargs
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    return {k: v for k, v in kwargs.items() if k in valid_params}

def safe_pad_operation(x, padding, mode='constant', value=0.0):
    """Safe padding operation that handles Half precision only for problematic modes"""
    # Modes qui nécessitent le fix Half precision
    problematic_modes = ['replicate', 'reflect', 'circular']
    
    if mode in problematic_modes:
        try:
            return F.pad(x, padding, mode=mode, value=value)
        except RuntimeError as e:
            if "not implemented for 'Half'" in str(e):
                original_dtype = x.dtype
                return F.pad(x.float(), padding, mode=mode, value=value).to(original_dtype)
            else:
                raise e
    else:
        # Pour 'constant' et autres modes compatibles, pas de fix nécessaire
        return F.pad(x, padding, mode=mode, value=value)


def safe_interpolate_operation(x, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    """Safe interpolate operation that handles Half precision for problematic modes"""
    # Modes qui peuvent causer des problèmes avec Half precision
    problematic_modes = ['bilinear', 'bicubic', 'trilinear']
    
    if mode in problematic_modes:
        try:
            return F.interpolate(
                x, 
                size=size, 
                scale_factor=scale_factor, 
                mode=mode, 
                align_corners=align_corners,
                recompute_scale_factor=recompute_scale_factor
            )
        except RuntimeError as e:
            if ("not implemented for 'Half'" in str(e) or 
                "compute_indices_weights" in str(e)):
                original_dtype = x.dtype
                return F.interpolate(
                    x.float(), 
                    size=size, 
                    scale_factor=scale_factor, 
                    mode=mode, 
                    align_corners=align_corners,
                    recompute_scale_factor=recompute_scale_factor
                ).to(original_dtype)
            else:
                raise e
    else:
        # Pour 'nearest' et autres modes compatibles, pas de fix nécessaire
        return F.interpolate(
            x, 
            size=size, 
            scale_factor=scale_factor, 
            mode=mode, 
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor
        )


@lru_cache
def torch_dtype_from_string(torch_type: str) -> torch.dtype:
    """
    Converts a string to a torch DType.
    """
    try:
        return {
            "complex128": torch.complex128,
            "cdouble": torch.complex128,
            "complex": torch.complex64,
            "complex64": torch.complex64,
            "cfloat": torch.complex64,
            "cfloat64": torch.complex64,
            "cf64": torch.complex64,
            "double": torch.float64,
            "float64": torch.float64,
            "fp64": torch.float64,
            "float": torch.float32,
            "full": torch.float32,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "fp8": torch.float8_e4m3fn,
            "float8": torch.float8_e4m3fn,
            "float8_e4m3": torch.float8_e4m3fn,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "fp84": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
            "fp85": torch.float8_e5m2,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "int16": torch.int16,
            "short": torch.int16,
            "int": torch.int32,
            "int32": torch.int32,
            "long": torch.int64,
            "int64": torch.int64,
            "bool": torch.bool,
            "bit": torch.bool,
            "1": torch.bool,
        }[torch_type[6:] if torch_type.startswith("torch.") else torch_type]
    except KeyError:
        raise ValueError(f"Unknown torch type '{torch_type}'")


@lru_cache
def get_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    """
    Gets the torch data type from a string.
    """
    if dtype is None:
        return torch.float32
    elif isinstance(dtype, str):
        return torch_dtype_from_string(dtype)
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise TypeError(f"Expected str or torch.dtype, got {type(dtype)}")


def get_torch_device(device: str | torch.device | None | int) -> torch.device:
    """
    Gets the torch device from a string or torch.device.
    """
    if device is None:
        return torch.device("cpu")
    elif isinstance(device, str):
        if device == "cuda":
            if torch.distributed.is_initialized():
                local_rank = torch.distributed.get_rank()
                return torch.device(f"cuda:{local_rank}")
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    elif isinstance(device, int):
        return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, torch.device):
        return device
    else:
        raise TypeError(f"Expected str, torch.device, int, or None, got {type(device)}")


@contextmanager
def no_init_weights() -> Iterator[None]:
    """
    Context manager to globally disable weight initialization to speed up loading large models.
    """
    torch_init_functions = {
        "uniform_": torch.nn.init.uniform_,
        "normal_": torch.nn.init.normal_,
        "trunc_normal_": torch.nn.init.trunc_normal_,
        "constant_": torch.nn.init.constant_,
        "xavier_uniform_": torch.nn.init.xavier_uniform_,
        "xavier_normal_": torch.nn.init.xavier_normal_,
        "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
        "kaiming_normal_": torch.nn.init.kaiming_normal_,
        "uniform": torch.nn.init.uniform,
        "normal": torch.nn.init.normal,
        "xavier_uniform": torch.nn.init.xavier_uniform,
        "xavier_normal": torch.nn.init.xavier_normal,
        "kaiming_uniform": torch.nn.init.kaiming_uniform,
        "kaiming_normal": torch.nn.init.kaiming_normal,
    }

    def _skip_init(*args: Any, **kwargs: Any) -> None:
        pass

    for name in torch_init_functions.keys():
        setattr(torch.nn.init, name, _skip_init)
    try:
        yield
    finally:
        # Restore the original initialization functions
        for name, init_func in torch_init_functions.items():
            setattr(torch.nn.init, name, init_func)


def read_from_url(url: str, timeout: int = 10, retries: int = 3) -> bytes:
    """
    Read from a URL with retries and timeout.
    """
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=timeout) as response:
                return response.read()
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(2 ** attempt)