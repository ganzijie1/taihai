"""DIAGNOSTIC_ONLY WSL2 verification for JAX and PyTorch CUDA residency."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accelerated_solver_infrastructure import JaxResidentRuntime, TorchResidentRuntime


def main() -> None:
    gpu_devices = jax.devices("gpu")
    if not gpu_devices:
        raise RuntimeError("JAX did not discover a WSL2 CUDA device")
    jax_runtime = JaxResidentRuntime("gpu")
    state = jax_runtime.put("residency", np.arange(4096, dtype=np.float64))
    update = jax_runtime.compile("residency.update", lambda value: 0.999 * value + 0.001)
    for _ in range(8):
        state = update(state)
    jax_runtime.buffers["residency"] = state
    state.block_until_ready()

    torch_runtime = TorchResidentRuntime("cuda:0")
    tensor = torch_runtime.put("residency", np.arange(4096, dtype=np.float64))
    pointer = int(tensor.data_ptr())
    for _ in range(8):
        tensor.mul_(0.999).add_(0.001)
    torch.cuda.synchronize()
    if int(tensor.data_ptr()) != pointer:
        raise AssertionError("PyTorch CUDA allocation was not stable")

    report = {
        "scope": "DIAGNOSTIC_ONLY",
        "jax": jax_runtime.diagnostics(),
        "torch": torch_runtime.diagnostics(),
        "jax_value_checksum": float(jnp.sum(state)),
        "torch_value_checksum": float(tensor.sum().item()),
    }
    target = Path(__file__).resolve().parents[1] / "outputs" / "wsl_jax_cuda_diagnostic.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DIAGNOSTIC_ONLY_WSL_JAX_CUDA: PASS")


if __name__ == "__main__":
    main()
