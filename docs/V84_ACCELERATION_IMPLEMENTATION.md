# V8.4 acceleration implementation

This note records executable infrastructure only. It does not contain or update
Taiwan Strait outcome probabilities.

## Implemented items

1. MFG density and WFR engagement mass stay on CUDA throughout the monthly
   fixed-point solve. Diagnostics are reduced on device and copied once per
   month; full density and mass are copied together at the monthly boundary.
2. The 25-point tridiagonal solve defaults to a Triton-compiled batched Thomas
   implementation. Batched PCR remains available as an explicit backend and
   numerical cross-check. CPU and installations without Triton retain the
   eager Thomas fallback.
3. The formal runner uses persistent process workers. Each scenario/intervention
   cell imports the model and initializes CUDA once; merging and auditing happen
   in the parent process. The default is one worker per physical GPU.
4. The fourteen V8.4 ablations are an explicit variant axis. Exogenous random
   innovations and frozen econometric draws are shared path by path, while all
   endogenous stocks and decisions remain variant-specific. A cell run writes
   all fourteen checkpoints.
5. State arrays follow `[variant, path, ...]` at the batch boundary and are
   flattened only for vectorized model kernels. Output aggregation restores the
   variant axis before computing statistics.

## Environment

The native Windows CUDA environment is pinned in
`requirements-windows-cuda.txt` to PyTorch 2.6.0 CUDA 12.4 and
`triton-windows` 3.2.0.post21. Triton for Windows is a community package;
PyTorch and CUDA remain unchanged by its installation.

## Verification

Run:

```powershell
$env:PYTHONPATH='work'
python work/diagnose_taiwan_v84_acceleration.py
python work/prepare_taiwan_v84.py
python work/prepare_taiwan_v84_ablations.py
python scripts/taiwan_model_audit_manifest.py outputs/taiwan_v84_traceability.json
```

The diagnostic artifact is
`outputs/taiwan_v84_acceleration_diagnostic.json`. It is explicitly marked
`DIAGNOSTIC_ONLY` and checks solver equivalence, conservation, common random
numbers, variant-batch equivalence, CUDA residency, and worker persistence.
