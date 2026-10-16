# Taiwan Strait Coupled Model

This repository contains the reproducible code path for the Taiwan Strait
evolution and campaign model. V7.7 is the current frozen source specification
and formal-result baseline. Its 30 registered scenario/case cells completed at
128 paths and 360 months, and the independent result audit passed.

The model is a monthly, same-path coupled system. Its active layers include
campaign and alliance intervention, continuous multi-population MFG, competing
risks, social contagion, finance, energy, trade, 140-node MR-CGE, open-economy
DSGE, disequilibrium inventories and backlogs, non-financial SFC, and a
four-party social accounting matrix (SAM). Later source versions add
econometric calibration, information/network/Riemannian geometry, three-sector
structural transformation, continuous state-action MFG, causal optimal
transport, stochastic geometry, symplectic/port-Hamiltonian dynamics and a
  hybrid large-deviation importance sampler. V6.8--V7.7 additionally activate
functional-analytic checks, martingale optimal transport, Tinbergen
reachability and tropical bottlenecks, combinatorial/continuous/Finsler
networks, hypergraph and state-action MFG, exact finite-complex online TDA,
information-bottleneck active observation, FGW/Ollivier/Hodge network
geometry, nested CES production and a signed inertial coupled oscillator.

## Current integrity status

- V5.3 formal-result baseline: retained as historical comparison
- V7.7 traceability and equation-family coverage: 100%
- V7.7 pre-simulation integrity, code-coupling and runtime audits: PASS
- Frozen formal design: 128 paths, 360 months, 30 conditional cells
- V7.7 result rows: 150; runtime diagnostic cells: 30
- V7.7 result SHA-256: `5209782A9E0E06CB19AB1D3A3BDA232FEE7C4A0AB67C878348B2979D4CB5493B`

Passing diagnostics establishes accounting and software consistency. It does
not establish empirical truth or classified-data-level predictive accuracy.

## Layout

- `work/`: frozen V7.7 executable dependency tree and public GIS cache; paths
  match the hashes recorded by the accepted specification
- `src/`: historical source mirror, diagnostics, audits, ablations and
  sensitivity scripts retained for version comparison
- `data/`: compact aggregated IO/MRIO inputs required by the public code path
- `artifacts/`: historical baselines plus V6.8--V7.7 traceability, audits,
  diagnostics, frozen specifications and the audited V7.7 formal result
- `work/gis_data/`: cached aggregate public terrain and road-network inputs
- `requirements-frozen-v77.txt`: exact core V7.7 numerical environment
- `requirements-wsl-jax-cuda.txt`: frozen WSL2 GPU environment
- `requirements-windows-cuda.txt`: Windows PyTorch CUDA environment

## Prepare the artifact layout

The research workspace stores generated records under `outputs/`, while this
repository keeps published records under `artifacts/`:

```powershell
New-Item -ItemType Directory -Force outputs
Copy-Item artifacts/*.json outputs/
```

## Run the V7.7 integrity gate

```powershell
python work/prepare_taiwan_v77.py
```

This command rebuilds diagnostic, traceability, static-coupling and frozen
specification artifacts. It does not produce substantive outcome probabilities.

## Reproduce the complete V7.7 simulation

Use the checkpointed entry point. Each completed scenario is written
atomically and is reusable only when paths, months, scenario and frozen-spec
hash all match:

```powershell
$env:TAIWAN_ALLOW_COMPLETE_RUN='1'
$env:TAIWAN_V77_PATHS='128'
$env:TAIWAN_V77_MONTHS='360'
python work/simulate_taiwan_v77_checkpointed.py
python work/audit_taiwan_v77_results.py
```

The checkpointed runner accepts a checkpoint only when its scenario, case,
path count, horizon and frozen-specification hash match. The result audit checks
150 conditional rows, 30 diagnostic cells, probability bounds, the complete
model gate and registered solver residuals.

## Result interpretation

`artifacts/taiwan_v77_result_summary.json` reports conditional broad-control
rates by intervention regime and horizon. Registered trade scenarios are
equal-weight sensitivity cases, not estimated scenario probabilities. A zero
Monte Carlo estimate means no qualifying terminal event was observed in the
finite weighted sample; it is not a proof of zero real-world probability.

The model is a public-data conditional mechanism study, not a classified
operational forecast. Conflict onset, intervention regime, conditional campaign
outcome and unconditional joint outcome must not be conflated.

## Reproduce the V5.3 formal baseline

The complete run is intentionally guarded. It can take hours on a CPU:

```powershell
$env:TAIWAN_ALLOW_COMPLETE_RUN='1'
$env:TAIWAN_V53_PATHS='128'
$env:TAIWAN_V53_MONTHS='360'
$env:TAIWAN_EQUILIBRIUM_SOLVER='picard'
python src/simulate_taiwan_fully_closed_v53.py
```

The code resolves input data relative to the repository root. Optional JAX,
PyTorch and GeomLoss paths degrade to documented CPU fallbacks when unavailable.
GPU execution must be benchmarked against identical seeds and error tolerances;
device discovery alone is not evidence of a valid speedup.

## Data and identification boundary

The compact IO files are derived from OECD ICIO 2022 and CEADS China MRIO.
Production, intermediate use, value added, final demand and trade control totals
are observed or derived public accounts. Wartime adjustment speeds, household
group splits where harmonized microdata are unavailable, and rare-event
parameters are scenario priors and require sensitivity analysis.

## Reproducibility

The accepted formal baseline is frozen in
`artifacts/taiwan_v77_frozen_specification.json`. Any source change requires a
new diagnostic pass, manifest audit and freeze before an outcome run can be
treated as comparable. The full result, compact summary and independent audit
are stored as `taiwan_v77_simulation.json`, `taiwan_v77_result_summary.json`
and `taiwan_v77_result_audit.json` under `artifacts/`.
